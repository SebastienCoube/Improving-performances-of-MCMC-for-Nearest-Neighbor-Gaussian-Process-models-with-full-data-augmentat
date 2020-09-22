mcmc_nngp_predict_field = function(mcmc_nngp_list, predicted_locs, burn_in = .5, n_cores = 1, m = 10)
{
  # constructing NNarray of observed and predicted locations
  locs = rbind(mcmc_nngp_list$locs, predicted_locs)
  NNarray = GpGp::find_ordered_nn(locs, m = m) 
  NNarray_non_NA = !is.na(NNarray)
  sparse_chol_row_idx = row(NNarray)[NNarray_non_NA]
  sparse_chol_column_idx = NNarray[NNarray_non_NA]
  # extracting info from mcmc nngp list
  space_time_model = mcmc_nngp_list$space_time_model
  vecchia_approx = mcmc_nngp_list$vecchia_approx
  # grabbig non thinned-oud indices of field samples
  stored_idx = mcmc_nngp_list$records$chain_1$saved_field
  stored_idx = stored_idx[stored_idx>(burn_in*max(stored_idx))]
  n_samples = length(stored_idx)
  predicted_field_samples = parallel::mclapply(mc.cores = n_cores, 
    X = mcmc_nngp_list$records, 
                               FUN = function(chain)
                               {
                                 # pre allocationg matrix 
                                 predicted_field_samples = matrix(0, n_samples, nrow(predicted_locs))
                                 # store whether the nngp factor needs to be recomputed bc of shape parameters change
                                 need_change_nngp_factor = !duplicated(chain$params$shape[stored_idx,])
                                 # i_predict = the index of the prediction made
                                 for(i_predict in seq(n_samples))
                                 {
                                   # i_chain = the corresponding index on the whole chain
                                   i_chain = stored_idx[i_predict]
                                   # i_chain = the corresponding index on the non thinned-out field
                                   i_field = match(i_chain, chain$saved_field)
                                   # compute nngp factor
                                   if(need_change_nngp_factor[i_predict])
                                   {
                                     shape = sapply(seq(length(space_time_model$covfun$shape_params)), function(j)
                                     {
                                       if(substr(space_time_model$covfun$shape_params[j], 1, 3)=="log")return(exp(chain$params$shape[i_chain, j]))
                                       else if(substr(space_time_model$covfun$shape_params[j], 1, 6)=="qlogis")return(1.5*plogis(chain$params$shape[i_chain, j]))
                                     })
                                     compressed_sparse_chol = GpGp::vecchia_Linv(covparms = c(1, shape, 0), covfun_name = space_time_model$covfun$stationary_covfun, locs, NNarray)
                                     sparse_chol = Matrix::sparseMatrix(i = sparse_chol_row_idx, j = sparse_chol_column_idx, x = compressed_sparse_chol[NNarray_non_NA], triangular = T)
                                   }
                                   # predict field
                                   spat_sd = exp(.5*chain$params$log_scale[i_chain])
                                   predicted_field_samples[i_predict,] = 
                                     as.vector(
                                        spat_sd * Matrix::solve(
                                        sparse_chol, 
                                         c(
                                           (1/spat_sd)*as.vector(sparse_chol[1:vecchia_approx$n_locs, 1:vecchia_approx$n_locs]%*% (chain$params$field[i_field,]-chain$params$beta_0[i_chain])),
                                           rnorm(nrow(predicted_locs))
                                           )
                                         )
                                       )[-seq(vecchia_approx$n_locs)]
                                 }
                                 return(predicted_field_samples)
                               })
  rbined_samples = do.call(rbind, predicted_field_samples)
  predicted_field_summary = get_summary(rbined_samples)
  return(list("predicted_locs" = predicted_locs,"predicted_field_samples" = predicted_field_samples, "predicted_field_summary" = predicted_field_summary))
}
#mcmc_nngp_predict_field(mcmc_nngp_list = mcmc_nngp_list, predicted_locs =  as.matrix(expand.grid(seq(-60, 60, .5), seq(-60, 60, .5))), burn_in = .5, n_cores = 3, m = 5)
#mcmc_nngp_list$locs




mcmc_nngp_predict_fixed_effects = function(mcmc_nngp_list, X_predicted, burn_in = .5, n_cores = 1,  match_field_thinning = T, add_intercept = F)
{
  # should the fixed effects be matched with field thinning ? 
  if(match_field_thinning)  stored_idx = mcmc_nngp_list$records$chain_1$saved_field
  if(!match_field_thinning)  stored_idx = seq(mcmc_nngp_list$records$chain_1$iterations[nrow(mcmc_nngp_list$records$chain_1$iterations), 1])
  # get stored idx
  stored_idx = stored_idx[stored_idx>(burn_in*max(stored_idx))]
  n_samples = length(stored_idx)
  # expand matrix from X predicted
  model_matrix_X_predicted = model.matrix(~., X_predicted)
  # retrieve the names of the variables in order to match them in beta if the regressors are not all used 
  fixed_effects_names =  colnames(model_matrix_X_predicted)
  fixed_effects_names[1] = "beta_0"
  # should the intercept be integrated in the fixed effects ?  If not, remove it
  if(!add_intercept) 
  {
    fixed_effects_names = fixed_effects_names[-1]
    model_matrix_X_predicted = matrix(model_matrix_X_predicted[,-1], nrow(model_matrix_X_predicted))
  }
  # match fixed effects name between predicted input and  
  beta_subset = match(fixed_effects_names, c("beta_0", colnames(mcmc_nngp_list$X$X)))
  predicted_fixed_effects_samples = parallel::mclapply(mc.cores = n_cores, X = mcmc_nngp_list$records, 
                                               FUN = function(chain)
                                               {
                                                 # convert sampled fixed effects into non centered fixed effects
                                                 beta_matrix = cbind(chain$params$beta_0, chain$params$beta)[stored_idx,]
                                                 # de-centering of beta_0
                                                 if(ncol(beta_matrix)>1) beta_matrix[,1]  = beta_matrix[,1] - beta_matrix[,-1]%*%matrix(mcmc_nngp_list$X$X_mean, ncol = 1)
                                                 #print(chain)
                                                 
                                                 predicted_fixed_effects_samples = beta_matrix[, beta_subset] %*%  t(model_matrix_X_predicted)
                                                 gc()
                                                 predicted_fixed_effects_samples 
                                               })
  rbined_samples = do.call(rbind, predicted_fixed_effects_samples)
  predicted_fixed_effects_summary = get_summary(rbined_samples)
  return(list("X_predicted" = X_predicted, "predicted_fixed_effects_samples" = predicted_fixed_effects_samples, "predicted_fixed_effects_summary" = predicted_fixed_effects_summary))
}
#mcmc_nngp_predict_field(mcmc_nngp_list = mcmc_nngp_list, predicted_locs =  as.matrix(expand.grid(seq(-60, 60, .5), seq(-60, 60, .5))), burn_in = .5, n_cores = 3, m = 5)
#mcmc_nngp_list$locs
# This one is doing : chromatic update on blocked Gaussian field + blocked parameters-field

###############################
# Vecchia likelihood function #
###############################

#Gaussian likelihood from Vecchia linv, for a 0-mean GP
ll_compressed_sparse_chol = function(Linv, field, NNarray, log_scale)
{
  chol_field = GpGp::Linv_mult(Linv = Linv, z = field, NNarray)
  sum(log(Linv[NNarray[,1],1]))-nrow(NNarray)*0.5*log_scale-0.5*sum(chol_field^2)/exp(log_scale)
}

mcmc_nngp_update_Gaussian = function(locs, X, observed_field, space_time_model, 
                                     vecchia_approx, states, 
                                     n_iterations_update, 
                                     n_cores = NULL, field_thinning = 1, 
                                     ancillary = T, n_chromatic = 10,
                                     iterations)
{
  
  if(is.null(n_cores))n_cores = min(parallel::detectCores()-1, length(states))
  
  
  return(parallel::mclapply(mc.cores = n_cores, 
                            seq(length(states)), FUN = function(i)
                            {
                              ######################
                              # Initializing stuff #
                              ######################
                              ################
                              # Setting seed #
                              ################
                              iter_start = iterations[nrow(iterations), 1]
                              # set seed 
                              set.seed(iter_start+i)
                              #########################################
                              # Initializing chain storage structures #
                              #########################################
                              state = states[[i]]
                              # this part re-creates a small portion of the $records objects of each chain. It fills it with chain state during the run, and then updates each chain with the new values
                              records = list()
                              # storing chain results
                              records$beta_0 = matrix(0, n_iterations_update, 1)
                              colnames(records$beta_0) = "beta_0"
                              if(!is.null(X$X))
                              {  
                                records$beta = matrix(0, n_iterations_update, ncol(X$X))
                                colnames(records$beta) = names(state$params$beta)
                              }  
                              records$log_scale = matrix(0, n_iterations_update, 1)
                              records$log_noise_variance = matrix(0, n_iterations_update, 1)
                              records$shape = matrix(0, n_iterations_update, length(space_time_model$covfun$shape_params))
                              colnames(records$shape) = space_time_model$covfun$shape_params
                              #matrix instead of vector since there is n field parameters = better thin the field
                              records$field = matrix(0, nrow = round(n_iterations_update*field_thinning), ncol = vecchia_approx$n_locs)
                              
                              # acceptance records for  covariance parameters
                              acceptance_records = list()
                              acceptance_records$covariance_acceptance_sufficient = rep(0, n_iterations_update)
                              acceptance_records$covariance_acceptance_ancillary = rep(0, n_iterations_update)
                              
                              ###############################################
                              # Initializing Vecchia factor and field value #
                              ###############################################
                              # Initializing compressed inverse Cholesky using GpGp package. This form is mostly used in parameter updating
                              shape = sapply(seq(length(space_time_model$covfun$shape_params)), function(j)
                              {
                                if(substr(space_time_model$covfun$shape_params[j], 1, 3)=="log")return(exp(state$params$shape[j]))
                                else if(substr(space_time_model$covfun$shape_params[j], 1, 6)=="qlogis")return(.5+.5*plogis(state$params$shape[j]))
                              })
                              compressed_sparse_chol = GpGp::vecchia_Linv(covparms = c(1, shape, 0), covfun_name = space_time_model$covfun$stationary_covfun, locs, vecchia_approx$NNarray)
                              sparse_chol = Matrix::sparseMatrix(i = vecchia_approx$sparse_chol_row_idx, j = vecchia_approx$sparse_chol_column_idx, x = compressed_sparse_chol[vecchia_approx$NNarray_non_NA], triangular = T)
                              precision_diag = as.vector((compressed_sparse_chol[vecchia_approx$NNarray_non_NA]^2)%*%Matrix::sparseMatrix(i = seq(length(vecchia_approx$sparse_chol_column_idx)), j = vecchia_approx$sparse_chol_column_idx, x = rep(1, length(vecchia_approx$sparse_chol_row_idx))))
                              current_p = rnorm(vecchia_approx$n_locs,0,1)
                              
                              if(length(X$locs)>0)
                              {
                                beta_interweaved_precision = as.matrix(Matrix::crossprod(sparse_chol%*%cbind(1, X$X[vecchia_approx$hctam_scol_1,X$locs])))
                                beta_interweaved_covmat = solve(beta_interweaved_precision, tol = min(rcond(beta_interweaved_precision),.Machine$double.eps))
                                beta_interweaved_covmat_chol = chol(beta_interweaved_covmat)
                                sparse_chol_X_locs = as.matrix(sparse_chol%*%cbind(1, X$X[vecchia_approx$hctam_scol_1,X$locs]))
                              }
                              
                              if(!is.null(X$X))mu = state$params$beta_0 + X$X%*%state$params$beta
                              else mu = rep(state$params$beta_0, vecchia_approx$n_obs)
                              
                              
                              
                              residuals_sum_matrix = Matrix::sparseMatrix(i = vecchia_approx$locs_match, j = seq(vecchia_approx$n_obs), x = 1)
                              
                              # setting true covparms            
                              #compressed_sparse_chol = GpGp::vecchia_Linv(covparms = c(x1, 1, 0), covfun_name = space_time_model$covfun$stationary_covfun, locs, vecchia_approx$NNarray)
                              #sparse_chol = Matrix::sparseMatrix(i = vecchia_approx$sparse_chol_row_idx, j = vecchia_approx$sparse_chol_column_idx, x = compressed_sparse_chol[vecchia_approx$NNarray_non_NA], triangular = T)                                                                
                              #precision_diag = as.vector((compressed_sparse_chol[vecchia_approx$NNarray_non_NA]^2)%*%Matrix::sparseMatrix(i = seq(length(vecchia_approx$sparse_chol_column_idx)), j = vecchia_approx$sparse_chol_column_idx, x = rep(1, length(vecchia_approx$sparse_chol_row_idx))))                              
                              # setting true field
                              #state$params$field = true_field[mcmc_nngp_list$vecchia_approx$hctam_scol_1]                                
                              #################
                              # Gibbs sampler #
                              #################
                              for(iter in seq(1, n_iterations_update))
                              {
                                #setting fake covarms
                                #state$params$log_scale = rnorm(1, 0, .01)                                 
                                #state$params$shape = rnorm(1, 0, .01)     
 
 
                                ###########################################
                                # covariance parameters : range and scale with ancillary augmentation
                                ###########################################
                                #scale and range are block updated
                                # proposing new values
                                innovation =  rnorm(length(state$params$shape)+1, 0, exp(.5*state$transition_kernels$covariance_params_ancillary$logvar))
                                new_log_scale = state$params$log_scale + innovation[1]
                                new_shape = state$params$shape+innovation[seq(2, length(innovation))]
                                # computing new Vecchia spare Cholesky factor, new conditional mean, new precision
                                # Vecchia factor
                                shape = sapply(seq(length(new_shape)), function(j)
                                {
                                  if(substr(space_time_model$covfun$shape_params[j], 1, 3)=="log")return(exp(new_shape[j]))
                                  else if(substr(space_time_model$covfun$shape_params[j], 1, 6)=="qlogis")return(.5+.5*plogis(new_shape[j]))
                                })
                                new_compressed_sparse_chol = GpGp::vecchia_Linv(covparms = c(1, shape, 0), covfun_name = space_time_model$covfun$stationary_covfun, locs, vecchia_approx$NNarray)
                                new_sparse_chol = Matrix::sparseMatrix(i = vecchia_approx$sparse_chol_row_idx, j = vecchia_approx$sparse_chol_column_idx, x = new_compressed_sparse_chol[vecchia_approx$NNarray_non_NA], triangular = T)
                                new_field = state$params$field
                                #print(as.matrix(Matrix::solve(sparse_chol)))
                                new_field = state$params$beta_0 + exp(.5 * (new_log_scale - state$params$log_scale)) * as.vector(Matrix::solve(new_sparse_chol, sparse_chol    %*% (state$params$field - state$params$beta_0)))
                                GP_ratio =  0
                                field_response_ratio = 
                                  sum(dnorm(x = observed_field, mean = new_field          [vecchia_approx$locs_match] + mu -state$params$beta_0, sd = exp(0.5*state$params$log_noise_variance), log = T)-
                                      dnorm(x = observed_field, mean = state$params$field [vecchia_approx$locs_match] + mu -state$params$beta_0, sd = exp(0.5*state$params$log_noise_variance), log = T))
                                #print(field_response_ratio+GP_ratio+pc_prior_ratio+transition_ratio)
                                if(field_response_ratio+GP_ratio > log(runif(1)))        
                                {
                                  #parameter updating
                                  state$params$shape = new_shape
                                  state$params$log_scale = new_log_scale
                                  state$params$field = new_field
                                  #updating Vecchia cholesky
                                  compressed_sparse_chol = new_compressed_sparse_chol
                                  sparse_chol = new_sparse_chol
                                  precision_diag = as.vector((compressed_sparse_chol[vecchia_approx$NNarray_non_NA]^2)%*%Matrix::sparseMatrix(i = seq(length(vecchia_approx$sparse_chol_column_idx)), j = vecchia_approx$sparse_chol_column_idx, x = rep(1, length(vecchia_approx$sparse_chol_row_idx))))
                                  # updating ecceptance records
                                  acceptance_records$covariance_acceptance_ancillary[iter] = 1
                                  if(length(X$locs)>0)
                                  {
                                    beta_interweaved_precision = as.matrix(Matrix::crossprod(sparse_chol%*%cbind(1, X$X[vecchia_approx$hctam_scol_1,X$locs])))
                                    beta_interweaved_covmat = solve(beta_interweaved_precision, tol = min(rcond(beta_interweaved_precision),.Machine$double.eps))
                                    beta_interweaved_covmat_chol = chol(beta_interweaved_covmat)
                                    sparse_chol_X_locs = as.matrix(sparse_chol%*%cbind(1, X$X[vecchia_approx$hctam_scol_1,X$locs]))
                                  }
                                }
                                if(iter_start %in% seq(0, 2000) & iter/25 == iter %/% 25)
                                {
                                  if(mean(acceptance_records$covariance_acceptance_ancillary[seq(iter-24, iter)])<.05) state$transition_kernels$covariance_params_ancillary$logvar = state$transition_kernels$covariance_params_ancillary$logvar - rnorm(1, .4, .05)
                                  if(mean(acceptance_records$covariance_acceptance_ancillary[seq(iter-24, iter)])>.15)  state$transition_kernels$covariance_params_ancillary$logvar = state$transition_kernels$covariance_params_ancillary$logvar + rnorm(1, .4, .05)
                                }
                                
                                
                                ########################################################################
                                # covariance parameters : scale and range with sufficient augmentation #
                                ########################################################################
                                #scale and range are block updated
                                # proposing new values
                                innovation =  rnorm(length(state$params$shape)+1, 0, exp(.5*state$transition_kernels$covariance_params_sufficient$logvar))
                                new_log_scale = state$params$log_scale+innovation[1]
                                if(exp(new_log_scale)<var(observed_field))
                                {
                                  new_shape = state$params$shape+innovation[seq(2, length(innovation))]
                                  if(length(innovation) == 2)new_shape = state$params$shape+innovation[seq(2, length(innovation))]
                                  
                                  # computing new Vecchia spare Cholesky factor, new conditional mean, new precision
                                  # Vecchia factor
                                  shape = sapply(seq(length(new_shape)), function(j)
                                  {
                                    if(substr(space_time_model$covfun$shape_params[j], 1, 3)=="log")return(exp(new_shape[j]))
                                    else if(substr(space_time_model$covfun$shape_params[j], 1, 6)=="qlogis")return(.5+.5*plogis(new_shape[j]))
                                  })
                                  new_compressed_sparse_chol = GpGp::vecchia_Linv(covparms = c(1, shape, 0), covfun_name = space_time_model$covfun$stationary_covfun, locs, vecchia_approx$NNarray)
                                  new_sparse_chol = Matrix::sparseMatrix(i = vecchia_approx$sparse_chol_row_idx, j = vecchia_approx$sparse_chol_column_idx, x = new_compressed_sparse_chol[vecchia_approx$NNarray_non_NA], triangular = T)
                                  #print(as.matrix(Matrix::solve(sparse_chol)))
                                  # GP prior simulated field -- covariance parameters
                                  GP_ratio =  0
                                  GP_ratio = 
                                    ll_compressed_sparse_chol(Linv= new_compressed_sparse_chol, log_scale = new_log_scale,          field = state$params$field- state$params$beta_0, NNarray =  vecchia_approx$NNarray) - 
                                    ll_compressed_sparse_chol(Linv= compressed_sparse_chol,     log_scale = state$params$log_scale, field = state$params$field- state$params$beta_0, NNarray =  vecchia_approx$NNarray)
                                  
                                  #print(field_response_ratio+GP_ratio+pc_prior_ratio+transition_ratio)
                                  if(GP_ratio > log(runif(1)))        
                                  {
                                    #parameter updating
                                    state$params$shape = new_shape
                                    state$params$log_scale = new_log_scale
                                    #updating Vecchia cholesky
                                    compressed_sparse_chol = new_compressed_sparse_chol
                                    sparse_chol = new_sparse_chol
                                    precision_diag = as.vector((compressed_sparse_chol[vecchia_approx$NNarray_non_NA]^2)%*%Matrix::sparseMatrix(i = seq(length(vecchia_approx$sparse_chol_column_idx)), j = vecchia_approx$sparse_chol_column_idx, x = rep(1, length(vecchia_approx$sparse_chol_row_idx))))
                                    # updating ecceptance records
                                    acceptance_records$covariance_acceptance_sufficient[iter] = 1
                                    if(length(X$locs)>0)
                                    {
                                      beta_interweaved_precision = as.matrix(Matrix::crossprod(sparse_chol%*%cbind(1, X$X[vecchia_approx$hctam_scol_1,X$locs])))
                                      beta_interweaved_covmat = solve(beta_interweaved_precision, tol = min(rcond(beta_interweaved_precision),.Machine$double.eps))
                                      beta_interweaved_covmat_chol = chol(beta_interweaved_covmat)
                                      sparse_chol_X_locs = as.matrix(sparse_chol%*%cbind(1, X$X[vecchia_approx$hctam_scol_1,X$locs]))
                                    }
                                  }
                                }
                                if(iter_start %in% seq(0, 2000) & iter/25 == iter %/% 25)
                                {
                                  if(mean(acceptance_records$covariance_acceptance_sufficient[seq(iter-24, iter)])<.05) state$transition_kernels$covariance_params_sufficient$logvar = state$transition_kernels$covariance_params_sufficient$logvar - rnorm(1, .2, .05)
                                  if(mean(acceptance_records$covariance_acceptance_sufficient[seq(iter-24, iter)])>.15)  state$transition_kernels$covariance_params_sufficient$logvar = state$transition_kernels$covariance_params_sufficient$logvar + rnorm(1, .2, .05)
                                }
                                ##############
                                # Field mean #
                                ##############
                                
                                # updating just beta_0
                                if(length(X$locs) == 0 | is.null(X$X))
                                {
                                  beta_covmat = c(solve(as.vector(Matrix::crossprod(sparse_chol%*%rep(1, vecchia_approx$n_locs))))*exp(state$params$log_scale))
                                  beta_mean = exp(-state$params$log_scale)*as.vector(Matrix::crossprod(sparse_chol%*%state$params$field, sparse_chol%*%rep(1, vecchia_approx$n_locs)))*beta_covmat
                                  state$params$beta_0 = beta_mean + sqrt(as.vector(beta_covmat))*rnorm(1)
                                }
                                # if other regression coefficients
                                if(!is.null(X$X))
                                {
                                  # sampling beta
                                  #beta_mean =  crossprod(observed_field-state$params$field[vecchia_approx$locs_match], X$X) %*%X$solve_XTX
                                  beta_mean =  crossprod(observed_field-state$params$field[vecchia_approx$locs_match] + state$params$beta_0, cbind(1, X$X)) %*%X$solve_1XT1X
                                  #state$params$beta   = c(beta_mean + exp(.5*state$params$log_noise_variance) * as.vector(t(X$chol_solve_XTX)%*%rnorm(ncol(X$X))))
                                  innovation   = c(beta_mean + exp(.5*state$params$log_noise_variance) * as.vector(t(X$chol_solve_1XT1X)%*%rnorm(ncol(X$X)+1)))
                                  state$params$field = state$params$field - state$params$beta_0 + innovation[1]
                                  state$params$beta_0 = innovation[1]
                                  state$params$beta = innovation[-1]
                                  # interweaving centered sampling in case of location data to improve beta sampling
                                  if(length(X$locs)>0)
                                  {
                                    # creating resampled field X + beta_0 + X_locs Beta_locs
                                    other_field = state$params$field + X$X[vecchia_approx$hctam_scol_1,X$locs]%*%matrix(state$params$beta[X$locs], ncol = 1)
                                    beta_mean = beta_interweaved_covmat%*%as.vector(Matrix::crossprod(sparse_chol%*%other_field, sparse_chol_X_locs))
                                    innovation = as.vector(beta_mean) +exp(.5*state$params$log_scale)*t(beta_interweaved_covmat_chol)%*%rnorm(length(X$locs)+1)
                                    state$params$beta_0 = innovation[1]
                                    state$params$beta[X$locs] = innovation[-1]
                                    state$params$field  = other_field - X$X[vecchia_approx$hctam_scol_1,X$locs]%*%matrix(state$params$beta[X$locs], ncol = 1)
                                  }
                                }
                                
                                if(!is.null(X$X))mu = state$params$beta_0 + X$X%*%state$params$beta
                                else mu = rep(state$params$beta_0, vecchia_approx$n_obs)
                                
                                
                                
                                ############################
                                # Field : chromatic sampling #
                                #############################
                                for(i in seq(n_chromatic))
                                {
                                  
                                  residuals_sum = residuals_sum_matrix%*%(observed_field - mu)
                                  for(color_idx in unique(vecchia_approx$coloring))
                                  {
                                    selected_locs = which(vecchia_approx$coloring==color_idx)
                                    posterior_precision = exp(-state$params$log_scale) * precision_diag[selected_locs] + exp(-state$params$log_noise_variance)*vecchia_approx$obs_per_loc[selected_locs]
                                    #conditional mean
                                    cond_mean = state$params$beta_0 - # un conditional mean
                                      (1/posterior_precision) *
                                      ( 
                                        as.vector(Matrix::crossprod(sparse_chol[,selected_locs], sparse_chol%*%((state$params$field - state$params$beta_0)*(vecchia_approx$coloring!=color_idx))))*exp(-state$params$log_scale) # rest of the simulated Gaussian variables
                                        - exp(-state$params$log_noise_variance)*residuals_sum[selected_locs]# Gaussian observations 
                                      )
                                    # field sampling
                                    state$params$field [selected_locs] = as.vector(cond_mean +rnorm(length(selected_locs))/sqrt(posterior_precision))
                                  }
                                }
                                
                              ##################
                              # Noise variance #
                              ##################
                              
                              sum_squared_residuals = sum((observed_field - state$params$field[vecchia_approx$locs_match] - mu + state$params$beta_0)^2)
                              field_precision_field = sum((sparse_chol %*% (state$params$field -state$params$beta_0))^2)
                              for(i in seq(10))
                              {
                                innovation = rnorm(1, 0, .01)
                                if(exp(state$params$log_noise_variance + innovation)<var(observed_field))
                                {
                                  if(-.5 * vecchia_approx$n_obs*innovation - .5 * sum_squared_residuals * (exp(-state$params$log_noise_variance - innovation) - exp(-state$params$log_noise_variance)) > log(runif(1)))
                                  {
                                    state$params$log_noise_variance = state$params$log_noise_variance + innovation
                                  }
                                }
                              }
                              
                                

                                
                                
                                #print(iter)
                                
                                ######################
                                # Saving chain state #
                                ######################
                                
                                if(!is.null(X$X))records$beta[iter,] = state$params$beta
                                records$beta_0[iter] = state$params$beta_0
                                
                                records$log_noise_variance[iter] = state$params$log_noise_variance
                                records$log_scale[iter] = state$params$log_scale
                                records$shape[iter,] = state$params$shape
                                if(round(iter*field_thinning) == (iter*field_thinning)) records$field[iter*field_thinning,] = state$params$field
                                
                                
                              }
                              return(list("state" = state, "records" = records))
                            }))
}



get_summary = function(samples)
{
  out = t(apply(samples, 2, function(x)c(mean(x), quantile(x, c(0.025, 0.5, 0.975)), sd(x))))
  colnames(out) = c("mean", "q0.025", "median", "q0.975", "sd")
  out
}


mcmc_nngp_estimate = function(mcmc_nngp_list, burn_in = .5)
{
  
  iter = mcmc_nngp_list$records$chain_1$iterations[nrow(mcmc_nngp_list$records$chain_1$iterations),1]
  res = list()
  #########################
  # COVARIANCE PARAMETERS #
  #########################
  res$covariance_params = list()
  # getting MCMC samples of covariance parameters
    # getting variables names
  covparms_names = intersect(c("log_scale", "log_noise_variance", "shape"), names(mcmc_nngp_list$records$chain_1$params))
    # getting samples
  samples = do.call(rbind, lapply(mcmc_nngp_list$records, function(chain)
  {
    cnames = unname(unlist(sapply(covparms_names, function(x)
    {
      if(x!="shape")x
      else colnames(chain$params$shape)
    })))
    out = do.call(cbind, chain$params[covparms_names])[(burn_in*iter):iter,]
    colnames(out) = cnames
    out
  }))
  # getting estimates of sampled parameters, with log or logistic transform
  res$covariance_params$sampled_covparams = get_summary(samples)
  # getting estimates with GpGp parametrization 
    # transforming samples
  samples[,grep("log_", colnames(samples))] = exp(samples[,grep("log_", colnames(samples))])
  samples[,grep("qlogis_", colnames(samples))] = 1.5* plogis(samples[,grep("qlogis_", colnames(samples))])
    # changing names
  colnames(samples)  = unname(sapply(colnames(samples), function(x)
  {
    if(length(grep("log_", x))!=0) return(substr(x, 5, 1000))
    if(length(grep("qlogis_", x))!=0) return(substr(x, 8, 1000))
    else return( x)
  }))
  res$covariance_params$GpGp_covparams = get_summary(samples)
  # getting estimates with INLA parametrization
    # exponential covfun case
  if(length(grep("exponential", mcmc_nngp_list$space_time_model$covfun$stationary_covfun))!=0)
  {
    samples[,grep("range", colnames(samples))] = samples[,grep("range", colnames(samples))] * 2
  }
    # Matérn covfun case
  if(length(grep("matern", mcmc_nngp_list$space_time_model$covfun$stationary_covfun))!=0)
  {
    samples[,grep("range", colnames(samples))] = samples[,grep("range", colnames(samples))] * sqrt(8 * samples[,grep("smoothness", colnames(samples))])
    samples[,grep("smoothness", colnames(samples))] = NULL
  }
    # passing noise variance to precision, field scale to sd
  samples[,grep("noise", colnames(samples))] = 1/samples[,grep("noise", colnames(samples))]
  samples[,grep("scale", colnames(samples))] = (samples[,grep("scale", colnames(samples))])^.5
    # changing names
  colnames(samples)[grep("scale", colnames(samples))] = "sd_for_spatial"
  colnames(samples)[grep("noise", colnames(samples))] = "precision_of_Gaussian_obs"
  res$covariance_params$INLA_covparams = get_summary(samples)
  
  #################
  # FIXED EFFECTS #
  #################
  
  beta_names = setdiff(names(mcmc_nngp_list$records$chain_1$params), c("field", covparms_names))
  samples = do.call(rbind, lapply(mcmc_nngp_list$records, function(chain)
  {
    out = do.call(cbind, chain$params[beta_names])[(burn_in*iter):iter,]
    if(!is.matrix(out))out = matrix(out, ncol  =1)
    if(ncol(out)>1) out[,1] = out[,1] - out[,-1]%*%matrix(mcmc_nngp_list$X$X_mean, ncol = 1)
    out
  }))

  res$fixed_effects = get_summary(samples)
  res$fixed_effects =cbind(res$fixed_effects, "zero_out_of_ci" = ((sign(res$fixed_effects[,"q0.025"])*sign(res$fixed_effects[,"q0.975"]))>0))
  
  
  #########
  # FIELD #
  #########
  
  samples = do.call(rbind, lapply(mcmc_nngp_list$records, function(chain)
  {
    out = chain$params$field[mcmc_nngp_list$records$chain_1$saved_field > iter*burn_in,]
    out = out - chain$params$beta_0[mcmc_nngp_list$records$chain_1$saved_field[which(mcmc_nngp_list$records$chain_1$saved_field > iter*burn_in)]]
    out
  }))
  res$field = get_summary(samples)
  res
}

#test = mcmc_nngp_estimate(mcmc_nngp_list)
#test$field


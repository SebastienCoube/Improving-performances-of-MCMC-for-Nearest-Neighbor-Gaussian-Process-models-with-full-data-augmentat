mcmc_nngp_initialize = 
  function(observed_locs, #spatial locations
           observed_field, # Response variable
           X_obs = NULL, # Covariates per observation
           X_locs = NULL, # Covariates per location
           m = 10, #number of Nearest Neighbors
           reordering = "maxmin", #Reordering
           stationary_covfun = "exponential_isotropic", response_model = "Gaussian", # covariance model and reponse model
           n_chains = 3,  # number of MCMC chains
           seed = 1
  )
  {
    
    # time  
    t_begin = Sys.time()
    # seed
    set.seed(seed)
    # cleansing RAM
    gc()
    
    ###############
    # Re-ordering #
    ###############
    
    
    locs = rbind(observed_locs)
    # remove duplicated locations
    locs = locs[duplicated(locs)==F,]
    if(reordering[1] == "maxmin")locs_reordering = GpGp::order_maxmin(locs, lonlat = !identical((grep("sphere", "sphere")) , integer(0)))
    if(reordering[1] == "random")locs_reordering = sample(seq(nrow(locs)))
    if(reordering[1] == "coord")locs_reordering = GpGp::order_coordinate(locs = locs, coordinate = reordering[,2])
    if(reordering[1] == "dist_to_point")locs_reordering = GpGp::order_dist_to_point(locs, loc0 = reordering[,2], lonlat = !identical((grep("sphere", "sphere")) , integer(0)))
    if(reordering[1] == "middleout")locs_reordering = GpGp::order_middleout(locs, lonlat = !identical((grep("sphere", "sphere")) , integer(0)))
    locs = locs[locs_reordering,]
    # extracting number of locations as shortcut
    n = nrow(locs)

    
    ####################
    # space-time model #
    ####################
    
      #################
      # link function #
      #################
    
    space_time_model = list()
    space_time_model$response_model = response_model
    
      ###############
      # Hyperpriors #
      ###############
    
    
      #######################
      # Covariance function #
      #######################
    
    #extracting number of covariance parameters minus scale and nugget
    #for example : isotropic Matérn, extracts 2 = range + smoothness
    #isotropic exponential, extracts 1  = range
    if(stationary_covfun == "exponential_isotropic") shape_params = c("log_range")
    if(stationary_covfun == "exponential_sphere") shape_params = c("log_range")
    if(stationary_covfun == "exponential_scaledim") shape_params = paste("log_range", seq(ncol(locs)), sep = "_")
    if(stationary_covfun == "exponential_spacetime") shape_params = c("log_range_1", "log_range_2")
    if(stationary_covfun == "matern_isotropic") shape_params = c("log_range", "qlogis_smoothness")
    if(stationary_covfun == "matern_sphere") shape_params = c("log_range", "qlogis_smoothness")
    if(stationary_covfun == "matern_scaledim") shape_params = c(paste("log_range", seq(ncol(locs)), sep = "_"), "qlogis_smoothness")
    if(stationary_covfun == "matern_spacetime") shape_params = c("log_range_1", "log_range_2", "qlogis_smoothness")
    
    space_time_model$covfun$stationary_covfun = stationary_covfun
    space_time_model$covfun$shape_params = shape_params
    
    #########################
    # Vecchia approximation #
    #########################
    
    # This object gathers the NNarray table used by GpGp package and related objects
    
    vecchia_approx = list()
    # storing numbers
    vecchia_approx$n_locs = n
    vecchia_approx$n_obs = length(observed_field)
    # matching observed locations with reordered, unrepeated locations
    locs_match = match(split(observed_locs, row(observed_locs)), split(locs, row(locs)))
    vecchia_approx$locs_match = locs_match
    # doing reversed operation : for a given unrepeated location, tell which observations correspond
    vecchia_approx$hctam_scol = split(seq(vecchia_approx$n_obs), locs_match)
    vecchia_approx$hctam_scol_1 = sapply(vecchia_approx$hctam_scol, function(x)x[1])
    # count how many observations correspond to one location
    vecchia_approx$obs_per_loc = unlist(sapply(vecchia_approx$hctam_scol, length))
    #extracting NNarray =  nearest neighbours for Vecchia approximation
    vecchia_approx$NNarray = GpGp::find_ordered_nn(locs, m)
    
    #computations from vecchia_approx$NNarray in order to create sparse Cholesky using Matrix::sparseMatrix
    #non_NA indices from vecchia_approx$NNarray
    vecchia_approx$NNarray_non_NA = !is.na(vecchia_approx$NNarray)
    #column idx of the uncompressed sparse Cholesky factor
    vecchia_approx$sparse_chol_column_idx = vecchia_approx$NNarray[vecchia_approx$NNarray_non_NA]
    #row idx of the uncompressed sparse Cholesky factor
    vecchia_approx$sparse_chol_row_idx = row(vecchia_approx$NNarray)[vecchia_approx$NNarray_non_NA]
    # adjacency matrix of MRF
    vecchia_approx$MRF_adjacency_mat =  Matrix::crossprod(Matrix::sparseMatrix(i = vecchia_approx$sparse_chol_row_idx, j = vecchia_approx$sparse_chol_column_idx, x = 1))
    
    # stupid trick to coerce adjacency matrix format...
    vecchia_approx$MRF_adjacency_mat@x = rep(1, length(vecchia_approx$MRF_adjacency_mat@x))
    vecchia_approx$MRF_adjacency_mat[1, 2] = 0
    vecchia_approx$MRF_adjacency_mat[1, 2] = 1
    vecchia_approx$MRF_adjacency_mat@x = rep(1, length(vecchia_approx$MRF_adjacency_mat@x))
    vecchia_approx$coloring = naive_greedy_coloring(vecchia_approx$MRF_adjacency_mat)
    
    ##############
    # Regressors #
    ##############
    
    X = list()
    X$arg = list("X_locs" = X_locs, "X_obs" = X_obs)
    if(is.null(X_locs)&is.null(X_obs))X$X = NULL
    if(!is.null(X_locs)&is.null(X_obs))X$X = X_locs
    if(is.null(X_locs)&!is.null(X_obs))X$X = X_obs
    if(!is.null(X_locs)&!is.null(X_obs))X$X = cbind(X_locs, X_obs)
    if(!is.null(X$X))
    {
      X$X = model.matrix(~., X$X)
      cnames =  colnames(X$X)[-1]
      X$X = matrix(X$X[,-1], nrow = nrow(X$X))
      colnames(X$X) = cnames
      #X$X_sparse = Matrix::sparseMatrix(i = row(X$X)[X$X!=0], j = col(X$X)[X$X!=0], x = X$X[X$X!=0])
      X$locs = seq(ncol(X_locs))
      X$X_mean = c()
      for(i in seq(ncol(X$X)))X$X_mean = c(X$X_mean,  mean(X$X[,i])) 
      for(i in seq(ncol(X$X)))X$X[,i] = X$X[,i] - mean(X$X[,i])
      X$solve_XTX = solve(crossprod(X$X))
      X$chol_solve_XTX = chol(X$solve_XTX)
    }
    
    ################
    # Chain states #
    ################
    
    states = list()
    # creating sub-list : each sub-list is a chain
    for(i in seq(n_chains))
    {
      states[[paste("chain", i, sep = "_")]] = list()
    }
      #########################
      # Covariance parameters #
      #########################
    for(i in seq(n_chains))
    {
    if(stationary_covfun == "exponential_isotropic") states[[i]]$params$shape = c(sample(log(max(dist(locs[1:100,])))-log(seq(2000, 20000, 1)), 1))
    if(stationary_covfun == "exponential_sphere") states[[i]]$params$shape = c(sample(log(max(dist(locs[1:100,])))-log(seq(2000, 20000, 1)), 1))
    if(stationary_covfun == "exponential_scaledim") states[[i]]$params$shape = sapply(seq(ncol(locs)), function(j)return(sample(log(max(dist(locs[1:100,j])))-log(seq(2000, 20000, 1)), 1)))
    if(stationary_covfun == "exponential_spacetime") states[[i]]$params$shape = c(sample(log(max(dist(locs[1:100,-ncol(locs)])))-log(seq(2000, 20000, 1)), 1), sample(log(max(dist(locs[1:100,ncol(locs)])))-log(seq(2000, 20000, 1)), 1))
    if(stationary_covfun == "matern_isotropic") states[[i]]$params$shape = c(sample(log(max(dist(locs[1:100,])))-log(seq(2000, 20000, 1)), 1), rnorm(1))
    if(stationary_covfun == "matern_sphere") states[[i]]$params$shape = c(sample(log(max(dist(locs[1:100,])))-log(seq(2000, 20000, 1)), 1), rnorm(1))
    if(stationary_covfun == "matern_scaledim") states[[i]]$params$shape = c(sapply(seq(ncol(locs)), function(j)sample(log(max(dist(locs[1:100,j])))-log(seq(2000, 20000, 1)), 1)), rnorm(1))
    if(stationary_covfun == "matern_spacetime") states[[i]]$params$shape =  c(sample(log(max(dist(locs[1:100,-ncol(locs)])))-log(seq(2000, 20000, 1)), 1), sample(log(max(dist(locs[1:100,ncol(locs)])))-log(seq(2000, 20000, 1)), 1), rnorm(1))
    }
    
    
    #################
    # Gaussian case #
    #################
    
    
    if(response_model == "Gaussian")
    {
      # beta starting values
      if(!is.null(X$X))naive_ols =  lm(observed_field~X$X)
      else naive_ols =  lm(observed_field~NULL)
      
      
      
      # for each chain, creating sub-lists in order to stock all the stuff that is related to one chain, including : 
      # transition_kernel_sd is a list that stocks the (current) automatically-tuned transition kernels standard deviations
      # params is a list that stocks the (current) parameters of the model, including covariance parameters, the value of the sampled field, etc
      for(i in seq(n_chains))
      {
        # Starting points for transition kernels, will be adaptatively tuned
        states[[i]]$transition_kernels = list()
        states[[i]]$transition_kernels$covariance_params_sufficient$logvar = -2
        states[[i]]$transition_kernels$covariance_params_ancillary$logvar = -2
        states[[i]]$transition_kernels$log_noise_variance$logvar = -1
        #starting points for regression coeffs
        perturb = t(chol(vcov(naive_ols)))%*%rnorm(length(naive_ols$coefficients))
        states[[i]]$params$beta_0 = naive_ols$coefficients[1] + perturb[1]
        if(!is.null(X$X))states[[i]]$params$beta = naive_ols$coefficients[-1] + perturb[-1]
        #starting points for covariance parameters
        states[[i]]$params$log_scale = log(rbeta(1, shape1 = 10, shape2 = 10) * var(naive_ols$residuals)) 
        states[[i]]$params$log_noise_variance = log(rbeta(1, shape1 = 10, shape2 = 10) * var(naive_ols$residuals))
        #field
        shape = sapply(seq(length(space_time_model$covfun$shape_params)), function(j)
        {
          if(substr(space_time_model$covfun$shape_params[j], 1, 3)=="log")return(exp(states[[i]]$params$shape[j]))
          else if(substr(space_time_model$covfun$shape_params[j], 1, 6)=="qlogis")return(.4+.7*plogis(states[[i]]$params$shape[j]))
        })
        Linv = GpGp::vecchia_Linv(covparms = c(1, shape, 0), covfun_name = stationary_covfun, locs = locs, NNarray = vecchia_approx$NNarray)
        Linv = 
          Matrix::sparseMatrix(
            i = vecchia_approx$sparse_chol_row_idx, 
            j = vecchia_approx$sparse_chol_column_idx, 
            x = Linv[vecchia_approx$NNarray_non_NA], 
            triangular = T)
        states[[i]]$params$field  = states[[i]]$params$beta_0 + sqrt(exp(states[[i]]$params$log_scale)) * as.vector(Matrix::solve(Linv, rnorm(vecchia_approx$n_locs)))
      }
    }

    
    
    
   #######################
   # Chain records setup #
   #######################
   
   # records is a list that stocks the recorded parameters of the model, including covariance parameters, the value of the sampled field, etc. In terms of RAM, those are the biggest bit !
   # iteration is a 2-colums matrix that records the iteration at the end of each chains join and the associated CPU time
   records = list()
   for(i in seq(n_chains))
   {
     records[[paste("chain", i, sep = "_")]] = list()
     records[[i]]$iterations = matrix(c(0, Sys.time()-t_begin), ncol = 2)
     colnames(records[[i]]$iterations) = c("iteration", "time")
     records[[i]]$params =  list()
     
   }
   diagnostics = list()
   diagnostics$Gelman_Rubin_Brooks = list()
   
   
   ##########
   
   print(paste("Setup done,", as.numeric(Sys.time()- t_begin, units = "secs"), "s elapsed" ))
   return(list("locs" = locs, "X" = X, "observed_field" = observed_field, "observed_locs" = observed_locs, "space_time_model" = space_time_model, 
               "vecchia_approx" = vecchia_approx, "states" = states, "records" = records, "diagnostics" = diagnostics,
               "t_begin" = t_begin, "seed" = seed))
}  

Gelman_Rubin_Brooks = function(records, burn_in = .5, n=NULL)
{
  if(is.null(n)) n = nrow(records$chain_1$params$beta_0)
  samples = lapply(records, 
                   function(x)
                   {
                     res = do.call(cbind, lapply(x$params[setdiff(names(x$params), "field")], function(y)y[seq(burn_in*n, n),]))
                     res
                   })
  
  m = length(samples)
  # computing within and between variance matrices for higher level parameters  
  within_variance = lapply(samples, var)
  within_variance = Reduce("+", within_variance)/m
  means = sapply(samples, function(x)apply(x, 2, mean))
  between_variance = var(t(means))
  # multivariate diagnostic
  MPSRF = (n - 1) / (n) + (m + 1) / m * svd(solve(within_variance, tol = rcond(within_variance))%*%between_variance)$d[1]
  names(MPSRF) = "Multivariate"
  # univariate diagnostics
  Individual_PSRF = ((m+1)/(m)) * ((n-1)/(n)) * (diag(between_variance)/diag(within_variance)) +(n+1)/(n)
  names(Individual_PSRF) = colnames(samples[[1]])
  return(list("R_hat" = c(MPSRF, Individual_PSRF), "within_variance" = within_variance))
}


raw_chains_plots_one_param = function(records, name = "beta_0", begin = 1, end = NULL, main = NULL)
{
  if(is.null(end))end =  nrow(records[[1]]$params$beta_0)
  if(is.null(main))main = name
  to_be_plotted = lapply(records, function(record)record$params[[name]][seq(begin, end)])
  plot(seq(begin, end), type = "n", xlab = "iteration", ylab = name, main = main, 
       ylim = c(min(unlist(to_be_plotted)), max(unlist(to_be_plotted))))
  col = 1
  # loop over chains
  for(x in to_be_plotted)
  {
    lines(seq(begin, end), x, col = col)
    col = col+1
  }
}

raw_chains_plots_covparms = function(records, burn_in = .5)
{
  iter = nrow(records[[1]]$params$beta_0)
  par(mfrow = c(length(colnames(records[[1]]$params$shape))+3, 1))
  for(name in c(setdiff(names(records[[1]]$params), c("shape", "beta", "field"))))
  {
    to_be_plotted = lapply(records, function(record)record$params[[name]][seq(burn_in*(iter-1), iter-1)])
    plot(seq(burn_in*(iter-1), iter-1), seq(burn_in*(iter-1), iter-1), type = "n", xlab = "iteration", ylab = name, main = name, 
         ylim = c(min(unlist(to_be_plotted)), max(unlist(to_be_plotted))))
    col = 1
    # loop over chains
    for(x in to_be_plotted)
    {
      lines(seq(burn_in*(iter-1), iter-1), x, col = col)
      col = col+1
    }
  }
  for(i in seq(length(colnames(records[[1]]$params$shape))))
  {
    name = colnames(records[[1]]$params$shape)[i]
    to_be_plotted = lapply(records, function(record)record$params$shape[seq(burn_in*(iter-1), iter-1),i])
    plot(seq(burn_in*(iter-1), iter-1), seq(burn_in*(iter-1), iter-1), type = "n", xlab = "iteration", ylab = name, main = name, 
         ylim = c(min(unlist(to_be_plotted)), max(unlist(to_be_plotted))))
    col = 1
    # loop over chains
    for(x in to_be_plotted)
    {
      lines(seq(burn_in*(iter-1), iter-1), x, col = col)
      col = col+1
    }
  }
  par(mfrow = c(1, 1))
}


raw_chains_plots_beta = function(records, burn_in = .5)
{
  iter = nrow(records[[1]]$params$beta_0)
  par(mfrow = c(length(colnames(records[[1]]$params$shape))+3, 1))
  # loop over regression coeffs
  if("beta" %in% names(records[[1]]$params))
  {
    par(mfrow = c(min(ncol(records[[1]]$params$beta), 4), 1))
    for(i in seq(ncol(records[[1]]$params$beta)))
    {
      to_be_plotted = lapply(records, function(record)record$params$beta[seq(burn_in*(iter-1), iter-1), i])
      plot(seq(burn_in*(iter-1), iter-1), seq(burn_in*(iter-1), iter-1), type = "n", xlab = "iteration", ylab = colnames(records[[1]]$params$beta)[i], main = colnames(records[[1]]$params$beta)[i], 
           ylim = c(min(unlist(to_be_plotted)), max(unlist(to_be_plotted))))
      col = 1
      # loop over chains
      for(x in to_be_plotted)
      {
        lines(seq(burn_in*(iter-1), iter-1), x, col = col)
        col = col+1
      }
    }
  }
  
  par(mfrow = c(1, 1))
}



ESS = function(records, burn_in = .5)
{
  samples = lapply(records, 
                   function(x)
                   {
                     res = do.call(cbind, lapply(x$params[setdiff(names(x$params), "field")], function(y)y[seq(burn_in*nrow(y), nrow(y)),]))
                     res
                   })
  ESS = t(sapply(samples, function(x)apply(x, 2,coda::effectiveSize)))
  ESS = rbind(ESS, apply(ESS, 2, sum))
  ESS
}





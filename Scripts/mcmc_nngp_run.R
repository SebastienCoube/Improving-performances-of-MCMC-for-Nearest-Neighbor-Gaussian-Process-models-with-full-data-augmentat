mcmc_nngp_run = function(mcmc_nngp_list, 
                         Gelman_Rubin_Brooks_stop = c(1.1, 1.1), burn_in = .5, # MCMC parameters
                         n_cores = NULL, field_thinning = 1, n_iterations_update = 200, #run parameters
                         ancillary = T, n_chromatic = 10,    # MCMC parameters : ancillary covariance update and number of chromatic updates
                         save_name = NULL, n_cycles = 1, plot_beta = F)
{
  
  cycle = 1
  while(cycle <= n_cycles)
  {
    print(paste("cycle =", cycle))
    if(mcmc_nngp_list$space_time_model$response_model == "Gaussian")
    {
      res = mcmc_nngp_update_Gaussian(locs = mcmc_nngp_list$locs, X = mcmc_nngp_list$X, observed_field = mcmc_nngp_list$observed_field, 
                                      space_time_model = mcmc_nngp_list$space_time_model, vecchia_approx = mcmc_nngp_list$vecchia_approx, 
                                      states = mcmc_nngp_list$states, iterations = mcmc_nngp_list$records[[1]]$iterations, 
                                      n_iterations_update = n_iterations_update, n_cores = n_cores, field_thinning = field_thinning, 
                                      ancillary = ancillary, n_chromatic = n_chromatic 
      )
    }
    
    for(i in seq(length(mcmc_nngp_list$records)))
    {
      mcmc_nngp_list$states[[i]] = res[[i]]$state
      iter_start = mcmc_nngp_list$records[[i]]$iterations[nrow(mcmc_nngp_list$records[[i]]$iterations), 1]
      saved_field = seq(n_iterations_update)[round(seq(n_iterations_update)*field_thinning) == (seq(n_iterations_update)*field_thinning)]
      mcmc_nngp_list$records[[i]]$saved_field = c(mcmc_nngp_list$records[[i]]$saved_field, iter_start + saved_field)
      mcmc_nngp_list$records[[i]]$iterations = rbind(mcmc_nngp_list$records[[i]]$iterations, c(mcmc_nngp_list$records[[i]]$iterations[nrow(mcmc_nngp_list$records[[i]]$iterations),1] + n_iterations_update , as.numeric(Sys.time() - mcmc_nngp_list$t_begin, unit = "secs")))
      for(name in names(res[[i]]$records))
      {
        mcmc_nngp_list$records[[i]]$params[[name]] = rbind(mcmc_nngp_list$records[[i]]$params[[name]], res[[i]]$records[[name]])
      }
    }
    
    # diagnostics
    if(plot_beta)raw_chains_plots_beta(mcmc_nngp_list$records, burn_in)
    raw_chains_plots_covparms(mcmc_nngp_list$records, burn_in)
    mcmc_nngp_list$diagnostics$Gelman_Rubin_Brooks[[cycle]] = Gelman_Rubin_Brooks(mcmc_nngp_list$records, burn_in)
    mcmc_nngp_list$diagnostics$ESS[[cycle]] = ESS(mcmc_nngp_list$records, burn_in)
    print("Gelman-Rubin-Brooks R-hat : ")
    print(mcmc_nngp_list$diagnostics$Gelman_Rubin_Brooks[[cycle]]$R_hat)
    if(mcmc_nngp_list$diagnostics$Gelman_Rubin_Brooks[[cycle]]$R_hat[1]<Gelman_Rubin_Brooks_stop[1] |
       all(mcmc_nngp_list$diagnostics$Gelman_Rubin_Brooks[[cycle]]$R_hat[-1]<Gelman_Rubin_Brooks_stop[2]))
    {
      break
    }
    cycle = cycle+1
  }
  
  
  return(mcmc_nngp_list)
}

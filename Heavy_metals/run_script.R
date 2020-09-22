remove(list = ls()) ; gc()
source("Scripts/mcmc_nngp_initialize.R")
source("Scripts/mcmc_nngp_diagnose.R")
source("Scripts/mcmc_nngp_run.R")
source("Scripts/Coloring.R")

t1 = Sys.time()
mcmc_nngp_list = mcmc_nngp_initialize(
  observed_locs =  readRDS(file = "Heavy_metals/processed_data.RDS")$observed_locs, 
  observed_field = readRDS(file = "Heavy_metals/processed_data.RDS")$observed_field, 
  X_locs =         readRDS(file = "Heavy_metals/processed_data.RDS")$X_locs, 
  stationary_covfun = "exponential_sphere", m = 5)

source("Scripts/mcmc_nngp_update_Gaussian.R")
mcmc_nngp_list = mcmc_nngp_run(mcmc_nngp_list, n_cores = 3, field_thinning = .5, Gelman_Rubin_Brooks_stop = c(1.00, 1.00), n_cycles = 12)
t1 = Sys.time()-t1
saveRDS(file = "Heavy_metals/myfit.RDS", object = list("t1" = t1, "res" = mcmc_nngp_list))





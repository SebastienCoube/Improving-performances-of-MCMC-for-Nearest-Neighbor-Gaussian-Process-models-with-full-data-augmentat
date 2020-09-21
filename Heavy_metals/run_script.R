remove(list = ls()) ; gc()
source("R_scripts/mcmc_Vecchia_initialize.R")
source("R_scripts/mcmc_Vecchia_diagnose.R")
source("R_scripts/mcmc_Vecchia_run.R")
source("R_scripts/DSATUR.R")

t1 = Sys.time()
mcmc_vecchia_list = mcmc_Vecchia_initialize(
  observed_locs =  readRDS(file = "Heavy_metals/processed_data.RDS")$observed_locs, 
  observed_field = readRDS(file = "Heavy_metals/processed_data.RDS")$observed_field, 
  X_locs =         readRDS(file = "Heavy_metals/processed_data.RDS")$X_locs, 
  stationary_covfun = "exponential_sphere",
  pc_prior_range = c(1, .5), pc_prior_sd = c(1, .5), m = 5)

source("R_scripts/mcmc_vecchia_update_Gaussian.R")
mcmc_vecchia_list = mcmc_Vecchia_run(mcmc_vecchia_list, n_cores = 3, n_blocks = 100, n_iterations_update = 500, field_thinning = .2, Gelman_Rubin_Brooks_stop = c(1.00, 1.00), n_cycles = 8)
t1 = Sys.time()-t1
saveRDS(file = "Heavy_metals/myfit.RDS", object = list("t1" = t1, "res" = mcmc_vecchia_list))





remove(list = ls())
gc()
res_my_fit  = readRDS("Heavy_metals/myfit.RDS")
original_dataset  = readRDS("Heavy_metals/processed_data.RDS")
source("./Scripts/mcmc_nngp_estimate.R")

########################################################
# Gelman Rubin diags for spNNGP and our implementation #
########################################################


pdf("Heavy_metals/grbdiag_myfit.pdf")
n_grbdiags = length(res_my_fit$res$records$chain_1$params$shape)/100
grb_diag = rep(0, n_grbdiags)
for(i in seq(n_grbdiags))
{
  grb_diag[i] = coda::gelman.diag(x = list(
    coda::as.mcmc(
      cbind(
        exp(res_my_fit$res$records$chain_1$params$log_scale[seq(100*i)]),
        exp(res_my_fit$res$records$chain_1$params$log_noise_variance[seq(100*i)]),
        exp(res_my_fit$res$records$chain_1$params$shape[seq(100*i)])
      )),
    coda::as.mcmc(
      cbind(
        exp(res_my_fit$res$records$chain_2$params$log_scale[seq(100*i)]),
        exp(res_my_fit$res$records$chain_2$params$log_noise_variance[seq(100*i)]),
        exp(res_my_fit$res$records$chain_2$params$shape[seq(100*i)])
      )),
    coda::as.mcmc(
      cbind(
        exp(res_my_fit$res$records$chain_3$params$log_scale[seq(100*i)]),
        exp(res_my_fit$res$records$chain_3$params$log_noise_variance[seq(100*i)]),
        exp(res_my_fit$res$records$chain_3$params$shape[seq(100*i)])
      ))
  ))$mpsrf
}

plot(seq(n_grbdiags) *100, ylim = c(1, 3.5), grb_diag, type = "l", xlab = "iteration", ylab = "Gelman-Rubin-Brooks diagnostic")

grb_diag = matrix(0, 3, n_grbdiags)
for(i in seq(n_grbdiags))
{
  grb_diag[,i] = coda::gelman.diag(x = list(
    coda::as.mcmc(
      cbind(
        exp(res_my_fit$res$records$chain_1$params$log_scale[seq(100*i)]),
        exp(res_my_fit$res$records$chain_1$params$log_noise_variance[seq(100*i)]),
        exp(res_my_fit$res$records$chain_1$params$shape[seq(100*i)])
      )),
    coda::as.mcmc(
      cbind(
        exp(res_my_fit$res$records$chain_2$params$log_scale[seq(100*i)]),
        exp(res_my_fit$res$records$chain_2$params$log_noise_variance[seq(100*i)]),
        exp(res_my_fit$res$records$chain_2$params$shape[seq(100*i)])
      )),
    coda::as.mcmc(
      cbind(
        exp(res_my_fit$res$records$chain_3$params$log_scale[seq(100*i)]),
        exp(res_my_fit$res$records$chain_3$params$log_noise_variance[seq(100*i)]),
        exp(res_my_fit$res$records$chain_3$params$shape[seq(100*i)])
      ))
  ))$psrf[,1]
}
lines(seq(n_grbdiags) *100, grb_diag[1,], col = 2)
lines(seq(n_grbdiags) *100, grb_diag[2,], col = 3)
lines(seq(n_grbdiags) *100, grb_diag[3,], col = 4)
legend("topright", fill = seq(4), legend = c("multivariate", "spatial effect variance", "noise variance", "spatial effect range"))
abline(h = 1)
dev.off()

# beta


pdf("Heavy_metals/grbdiag_myfit_beta.pdf", width = 5, height = 5)
grb_diag = rep(0, n_grbdiags - 10 )
for(i in seq(11, n_grbdiags))
{
  grb_diag[i - 10] = coda::gelman.diag(x = list(
    coda::as.mcmc(
      cbind(
        res_my_fit$res$records$chain_1$params$beta_0[seq(100*i),],
        res_my_fit$res$records$chain_1$params$beta[seq(100*i),]
      )),
    coda::as.mcmc(
      cbind(
        res_my_fit$res$records$chain_2$params$beta_0[seq(100*i),],
        res_my_fit$res$records$chain_2$params$beta[seq(100*i),]
      )),
    coda::as.mcmc(
      cbind(
        res_my_fit$res$records$chain_3$params$beta_0[seq(100*i),],
        res_my_fit$res$records$chain_3$params$beta[seq(100*i),]
      ))
  ))$mpsrf
}



plot(seq(11, n_grbdiags) *100, grb_diag, type = "l", xlab = "iteration", ylab = "Gelman-Rubin-Brooks diagnostic", ylim = c(1, 2))
grb_diag = matrix(0, ncol(res_my_fit$res$records$chain_1$params$beta)+1, n_grbdiags - 10)
for(i in seq(11, n_grbdiags))
{
  grb_diag[,i-10] = coda::gelman.diag(x = list(
    coda::as.mcmc(
      cbind(
        res_my_fit$res$records$chain_1$params$beta_0[seq(100*i),],
        res_my_fit$res$records$chain_1$params$beta[seq(100*i),]
      )),
    coda::as.mcmc(
      cbind(
        res_my_fit$res$records$chain_2$params$beta_0[seq(100*i),],
        res_my_fit$res$records$chain_2$params$beta[seq(100*i),]
      )),
    coda::as.mcmc(
      cbind(
        res_my_fit$res$records$chain_3$params$beta_0[seq(100*i),],
        res_my_fit$res$records$chain_3$params$beta[seq(100*i),]
      ))
  ))$psrf[,1]
}
for(i in seq(ncol(res_my_fit$res$records$chain_1$params$beta)+1)){lines(seq(11, n_grbdiags)*100, grb_diag[i,], col = "lightgray")}

legend("topright", fill = c("black", "lightgray"), legend = c("multivariate", "individual (various chains)"))
abline(h = 1)
dev.off()


##############
# ESTIMATES #
##############

estimates = mcmc_nngp_estimate(res_my_fit$res)

fixed_effects_subset = estimates$fixed_effects[-c(grep(c("minotype"), row.names(estimates$fixed_effects)), grep(c("MAJOR1"), row.names(estimates$fixed_effects)), grep(c("glwd"), row.names(estimates$fixed_effects))),]
fixed_effects_subset = fixed_effects_subset[,-6]

covariance_parameters_estimates = estimates$covariance_params$GpGp_covparams
covariance_parameters_estimates[3,] = covariance_parameters_estimates[3,]*6371

estimates_out  =signif(rbind(covariance_parameters_estimates, fixed_effects_subset), 3)
xtable::xtable(estimates_out, digits = 5)


############################
# Predictions of residuals #
############################

# getting US map
usa <- maps::map("state", fill = TRUE)
IDs <- sapply(strsplit(usa$names, ":"), function(x) x[1])
usa <- maptools::map2SpatialPolygons(usa, IDs=IDs, proj4string=sp::CRS("+proj=longlat +datum=WGS84"))
# overlaying 5 Km spatial grid on US map
grid.list <- c("dairp.asc", "dmino.asc", "dquksig.asc", "dTRI.asc", "gcarb.asc",
               "geomap.asc", "globedem.asc", "minotype.asc", "nlights03.asc", "sdroads.asc",
               "twi.asc", "vsky.asc", "winde.asc", "glwd31.asc")
gridmaps <- rgdal::readGDAL("Heavy_metals/usgrids5km/dairp.asc")

names(gridmaps)[1] <- sub(".asc", "", grid.list[1])
for(i in grid.list[-1]) {
  gridmaps@data[sub(".asc", "", i[1])] <- rgdal::readGDAL(paste("Heavy_metals/usgrids5km/",i, sep = ""))$band1
}
AEA <- "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0
+ellps=GRS80 +datum=NAD83 +units=m +no_defs"
sp::proj4string(gridmaps) = sp::CRS(AEA)
usa_aea = sp::spTransform(usa, sp::CRS(AEA))
gridmaps_overlay = sp::over(gridmaps, usa_aea)
# transforming gridmaps into WGS84 data
gridmaps_WGS84 = sp::spTransform(gridmaps, sp::CRS("+proj=longlat +datum=WGS84"))
predicted_coords = sp::coordinates(gridmaps_WGS84)[!is.na(gridmaps_overlay),]

sp::plot(res_my_fit$res$locs, pch = 20, cex = .00001, col = 2, main = "sampling sites", xlab = "longitude", ylab = "latitude")
sp::plot(usa, add = T)

source("Scripts/mcmc_nngp_predict.R")
#prediction = mcmc_nngp_predict_field(res_my_fit$res, predicted_locs = predicted_coords, burn_in = .5, n_cores = 3, m = 5)
#saveRDS(prediction, "Heavy_metals/prediction_field.RDS")
prediction = readRDS("Heavy_metals/prediction_field.RDS")

pdf("Heavy_metals/latent_mean.pdf", width = 7, height = 4)
gridmaps$predicted_lead_mean = NA
gridmaps$predicted_lead_mean[!is.na(gridmaps_overlay)] = prediction$predicted_field_summary[,"mean"]
sp::plot(gridmaps["predicted_lead_mean"], main = "")
sp::plot(usa_aea, add = T)
dev.off()

pdf("Heavy_metals/latent_sd.pdf", width = 7, height = 4)
gridmaps$predicted_lead_sd = NA
gridmaps$predicted_lead_sd[!is.na(gridmaps_overlay)] = prediction$predicted_field_summary[,"sd"]
sp::plot(gridmaps["predicted_lead_sd"])
sp::plot(usa_aea, add = T)
dev.off()

pdf("Heavy_metals/sampling_sites.pdf", width = 10, height = 7)
sp::plot(ngs.aea[!is.na(sp::over(ngs.aea, usa_aea)),], pch = 20, cex = .05, col = 2)
sp::plot(usa_aea, add = T)
dev.off()


#########################################
# Fixed effect example :  air pollution #
#########################################



source("Heavy_metals/mcmc_nngp_predict.R")
pollution_data = gridmaps[c("dairp", "dTRI")]@data
pollution_data = pollution_data[!is.na(gridmaps_overlay),]
pollution_data[,1] = (pollution_data[,1] - original_dataset$X_locs_mean["dairp"])/original_dataset$X_locs_sd["dairp"]
pollution_data[,2] = (pollution_data[,2] - original_dataset$X_locs_mean["dTRI"])/original_dataset$X_locs_sd["dTRI"]
prediction_pollution_effect = mcmc_nngp_predict_fixed_effects(res_my_fit$res, pollution_data, burn_in = .5, n_cores = 1, add_intercept = F, match_field_thinning = T)
gridmaps$predicted_pollution_effect_mean = NA
gridmaps$predicted_pollution_effect_mean[!is.na(gridmaps_overlay)] = prediction_pollution_effect$predicted_fixed_effects_summary[,"mean"]
gridmaps$predicted_pollution_effect_sd = NA
gridmaps$predicted_pollution_effect_sd[!is.na(gridmaps_overlay)] = prediction_pollution_effect$predicted_fixed_effects_summary[,"sd"]

sp::plot(gridmaps["predicted_pollution_effect_mean"])
sp::plot(usa_aea, add = T)
sp::plot(gridmaps["predicted_pollution_effect_sd"])
sp::plot(usa_aea, add = T)

remove(prediction_pollution_effect)
gc()

prediction_urbanization_effect = mcmc_nngp_predict_fixed_effects(res_my_fit$res, gridmaps[c("nlights03", "sdroads")]@data, burn_in = .5, n_cores = 3, add_intercept = F)
est
names(gridmaps)





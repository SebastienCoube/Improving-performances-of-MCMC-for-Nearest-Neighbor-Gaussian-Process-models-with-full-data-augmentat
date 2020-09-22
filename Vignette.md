This document is a vignette for the supplementary material of *Two MCMC
Strategies for Nearest Neighbor Gaussian Processes*. The plan is the
following :

-   Create a small synthetic toy example

-   Initialize a model to analyze this toy example and familiarize with
    the objects that are created

-   Run the model, monitor convergence

-   Estimate the parameters

-   Predict the latent field and fix effects

-   Highlight the importance of interweaving with a bad regressor
    parametrization of the same toy example

Toy example creation
====================

Let’s start by creating a 1-dimension toy example

``` r
# let's set a seed...
set.seed(1)
# let's sample some spatial locations
locs = cbind(500*runif(2000), 1) # note that the second dimension does not move ! 
locs[1, 2] = 1.01 # actually, we have to change one point of the second dimension otherwise the nearest neighbor algorithms do not work
# let's sample a random field with exponential covariance
field = sqrt(10)* t(chol(GpGp::exponential_isotropic(c(1, 5, 0), locs)))%*%rnorm(2000)
# let's visualize the field
plot(locs[,1], field, main = "latent field")
```

![](Vignette_files/figure-markdown_github/generate_toy_example-1.png)

``` r
# let's add a fix effect
X = matrix(c(locs[,1], rnorm(2000)), ncol = 2) # the first regressor actually is equal to the spatial location
colnames(X) = c("slope", "white_noise")
beta = c(.01, rnorm(1)) #  simulate beta
beta_0 = rnorm(1) # add an intercept
#let's plot the fix effects
plot(locs[,1], X%*%beta + beta_0, main = "fix effects")
```

![](Vignette_files/figure-markdown_github/generate_toy_example-2.png)

``` r
# let's add noise with variance 5
noise = sqrt(5) * rnorm(2000)
plot(locs[,1], noise, main = "noise")
```

![](Vignette_files/figure-markdown_github/generate_toy_example-3.png)

``` r
# let's combine the latent field, the fix effect, and the Gaussian noise
observed_field = c(as.matrix(field+noise+ X%*%beta + beta_0))
plot(locs[,1], observed_field, main = "observed field")
```

![](Vignette_files/figure-markdown_github/generate_toy_example-4.png)

Initialization
==============

How to initialize a model ?
---------------------------

Now, let’s do the setup to work on the toy example. We will use
mcmc\_nngp\_initialize, a function that takes the observed data, Vecchia
approximation, and the specified covariance model as arguments and
outputs a big list with necessary information to run the model. The
initial chain states are guessed using the size of the domain
(covariance range), the variance of the signal (covariance scale and
noise variance), a naive OLS of the observed signal on the regressors
(fix effects of the models), and a perturbation is added in order to
overdisperse the starting points for Gelman-Rubin-Brooks diagnostics.
The list is generated using various arguments :

-   The covariance model of the latent field. The (stationary)
    covariance function is indicated with a string. The stationary
    functions of the GpGp package are used.

-   The Vecchia approximation design. The maxmin order is always used
    for locations ordering. the number of neighbors m is set to 10 by
    default but can be changed. If needed, the reference set can be
    restricted to a certain number of obervations using
    n\_reference\_set.

-   Regressors. Two types of regressors can be provided : X\_obs and
    X\_locs. The first can vary within a spatial location, while the
    second cannot : for example, smoking or alchool consumption can vary
    between the members of a household, while asbestos contamination
    cannot. The formers can only be passed as X\_obs, while the latter
    can be passed as X\_locs or X\_obs. The format is data.frame.
    Passing the regressor at the same time in the two slots will cause
    problems. When it is possible, a regressor should be apssed as
    X\_locs, we will see why later.

-   A seed, set to 1 if not precised

``` r
source("Scripts/mcmc_nngp_initialize.R")
source("Scripts/Coloring.R")
# Now, let's initialize the list. This creates the chains, guesses the initial states, does the Nearest Neighbor search for NNGP, etc.... 
mcmc_nngp_list = mcmc_nngp_initialize(observed_locs = locs, observed_field = observed_field, 
                                            stationary_covfun = "exponential_isotropic", 
                                            X_locs = as.data.frame(X), X_obs = NULL,
                                            m = 5,  
                                            seed = 1)
```

    ## [1] "Setup done, 1.23651695251465 s elapsed"

What is there in the list we just created ?
-------------------------------------------

This section explores the object mcmc\_vecchia\_list we just created in
order to familiarize with the objects that are stored in it.

### Some (reordered, without duplicate) spatial locations

mcmc\_nngp\_list$observed\_locs is the set of locations given as an
input. It can have duplicates. mcmc\_nngp\_list$locs is the set of
spatial locations with no duplicates. It is reordered using the Maxmin
order (see Guinness, *Permutation and grouping methods for sharpening
Gaussian process approximations*)

``` r
head(mcmc_nngp_list$observed_locs)
```

    ##          [,1] [,2]
    ## [1,] 132.7543 1.01
    ## [2,] 186.0619 1.00
    ## [3,] 286.4267 1.00
    ## [4,] 454.1039 1.00
    ## [5,] 100.8410 1.00
    ## [6,] 449.1948 1.00

``` r
head(mcmc_nngp_list$locs)
```

    ##           [,1] [,2]
    ## [1,] 473.98318    1
    ## [2,] 322.15788    1
    ## [3,] 497.64053    1
    ## [4,] 261.23064    1
    ## [5,]  17.08532    1
    ## [6,] 212.43027    1

### Regressors and various objects extracted from them

mcmc\_vecchia\_list$X contains various information about the fix effects
design.

mcmc\_vecchia\_list$X$X is the combination of X\_obs and X\_locs, but it
is not reordered like locs and there can be duplicates. it is centered.

mcmc\_vecchia\_list$X$X\_means indicates the means of the original
regressors, in order to re-transform the regression coefficients samples
once the model is fit. This allows to estimate the regression
coefficients and the effect of the covariates on the signal.

mcmc\_nngp\_list$X$locs indicates which columns of X come from X\_locs.
In our case, since no X\_locs was given, it indicates all the columns of
X.

The rest of the elements are pre-computed cross-products that are used
in the model fitting

``` r
head(mcmc_nngp_list$X$X)
```

    ##           slope white_noise
    ## [1,] -114.74263   0.7407856
    ## [2,]  -61.43501   0.3882794
    ## [3,]   38.92972   1.2980678
    ## [4,]  206.60693  -0.8018877
    ## [5,] -146.65600  -1.6009550
    ## [6,]  201.69788   0.9349216

### Information about the covariance model

mcmc\_nngp\_list$space\_time\_model gives info about the covariance
function, its parameters, their hyperpriors.

``` r
print(mcmc_nngp_list$space_time_model)
```

    ## $response_model
    ## [1] "Gaussian"
    ## 
    ## $covfun
    ## $covfun$stationary_covfun
    ## [1] "exponential_isotropic"
    ## 
    ## $covfun$shape_params
    ## [1] "log_range"

### The design of Vecchia approximation

mcmc\_nngp\_list$vecchia\_approx gives info about the Vecchia
approximation design, the reordering of locations, and some
miscellaneaous stuff such as the number of spatial locations and useful
shorthands.

#### The Nearest Neighbor Array (NNarray) that defines Vecchia approximation’s Directed Acyclic Graph, and some shorthands directly derived from it.

``` r
# Nearest Neighbor Array
head(mcmc_nngp_list$vecchia_approx$NNarray) 
```

    ##      [,1] [,2] [,3] [,4] [,5] [,6]
    ## [1,]    1   NA   NA   NA   NA   NA
    ## [2,]    2    1   NA   NA   NA   NA
    ## [3,]    3    1    2   NA   NA   NA
    ## [4,]    4    2    1    3   NA   NA
    ## [5,]    5    4    2    1    3   NA
    ## [6,]    6    4    2    5    1    3

``` r
# indicator of the non-NA coefficients in NNarray
head(mcmc_nngp_list$vecchia_approx$NNarray_non_NA) 
```

    ##      [,1]  [,2]  [,3]  [,4]  [,5]  [,6]
    ## [1,] TRUE FALSE FALSE FALSE FALSE FALSE
    ## [2,] TRUE  TRUE FALSE FALSE FALSE FALSE
    ## [3,] TRUE  TRUE  TRUE FALSE FALSE FALSE
    ## [4,] TRUE  TRUE  TRUE  TRUE FALSE FALSE
    ## [5,] TRUE  TRUE  TRUE  TRUE  TRUE FALSE
    ## [6,] TRUE  TRUE  TRUE  TRUE  TRUE  TRUE

``` r
 # the column indices in the Vecchia factor = the non-NA entries of NNarray
print(mcmc_nngp_list$vecchia_approx$sparse_chol_column_idx[1:100])
```

    ##   [1]   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
    ##  [19]  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
    ##  [37]  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
    ##  [55]  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72
    ##  [73]  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90
    ##  [91]  91  92  93  94  95  96  97  98  99 100

``` r
# the row indices in the Vecchia factor = the row indices of the non-NA entries of NNarray
print(mcmc_nngp_list$vecchia_approx$sparse_chol_row_idx[1:100]) 
```

    ##   [1]   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
    ##  [19]  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
    ##  [37]  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
    ##  [55]  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72
    ##  [73]  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90
    ##  [91]  91  92  93  94  95  96  97  98  99 100

#### A sparseMatrix objects that stores the adjacency matrix of the Markov graph that is induced by Vecchia approximation.

Is tis obtained by moralization of the Vecchia Directed Acyclic Graph.
It will be used for chromatic sampling.

``` r
print(mcmc_nngp_list$vecchia_approx$MRF_adjacency_mat[1:30, 1:30]) 
```

    ## 30 x 30 sparse Matrix of class "dgCMatrix"
    ##                                                                  
    ##  [1,] 1 1 1 1 1 1 1 1 1 1 1 . . . . . . 1 . . . . 1 . . . 1 . . .
    ##  [2,] 1 1 1 1 1 1 1 1 1 1 1 1 1 . 1 1 . . . . . 1 . . 1 . . . . .
    ##  [3,] 1 1 1 1 1 1 . 1 1 1 . . . . . . . . . . . . 1 . . . 1 . . .
    ##  [4,] 1 1 1 1 1 1 1 1 1 . 1 1 1 1 . . 1 . . . . 1 . . 1 . . . . .
    ##  [5,] 1 1 1 1 1 1 1 . . . . . 1 1 . . 1 . 1 1 . . . . . . . . 1 .
    ##  [6,] 1 1 1 1 1 1 1 1 1 . . . 1 1 . . 1 . 1 . 1 1 . 1 . 1 . . . .
    ##  [7,] 1 1 . 1 1 1 1 . 1 . . . 1 1 . . 1 . 1 1 1 . . 1 . 1 . 1 . .
    ##  [8,] 1 1 1 1 . 1 . 1 1 1 1 1 . . 1 1 . 1 . . . . 1 . . . 1 . . 1
    ##  [9,] 1 1 1 1 . 1 1 1 1 1 1 1 1 . 1 1 . . . . . 1 . . 1 . . . . .
    ## [10,] 1 1 1 . . . . 1 1 1 1 . . . . . . 1 . . . . 1 . . . 1 . . .
    ## [11,] 1 1 . 1 . . . 1 1 1 1 1 . . 1 1 . 1 . . . . 1 . . . 1 . . 1
    ## [12,] . 1 . 1 . . . 1 1 . 1 1 . . 1 1 . 1 . . . 1 . . 1 . . . . .
    ## [13,] . 1 . 1 1 1 1 . 1 . . . 1 1 . . 1 . . . 1 1 . 1 . 1 . . . .
    ## [14,] . . . 1 1 1 1 . . . . . 1 1 . . 1 . 1 1 1 . . 1 . . . . 1 .
    ## [15,] . 1 . . . . . 1 1 . 1 1 . . 1 1 . 1 . . . . 1 . . . 1 . . 1
    ## [16,] . 1 . . . . . 1 1 . 1 1 . . 1 1 . 1 . . . . . . . . . . . .
    ## [17,] . . . 1 1 1 1 . . . . . 1 1 . . 1 . 1 1 1 . . 1 . 1 . 1 . .
    ## [18,] 1 . . . . . . 1 . 1 1 1 . . 1 1 . 1 . . . . 1 . . . 1 . . 1
    ## [19,] . . . . 1 1 1 . . . . . . 1 . . 1 . 1 1 1 . . 1 . . . . 1 .
    ## [20,] . . . . 1 . 1 . . . . . . 1 . . 1 . 1 1 1 . . 1 . . . . 1 .
    ## [21,] . . . . . 1 1 . . . . . 1 1 . . 1 . 1 1 1 . . 1 . 1 . 1 1 .
    ## [22,] . 1 . 1 . 1 . . 1 . . 1 1 . . . . . . . . 1 . . 1 . . . . .
    ## [23,] 1 . 1 . . . . 1 . 1 1 . . . 1 . . 1 . . . . 1 . . . 1 . . 1
    ## [24,] . . . . . 1 1 . . . . . 1 1 . . 1 . 1 1 1 . . 1 . 1 . 1 1 .
    ## [25,] . 1 . 1 . . . . 1 . . 1 . . . . . . . . . 1 . . 1 . . . . .
    ## [26,] . . . . . 1 1 . . . . . 1 . . . 1 . . . 1 . . 1 . 1 . 1 . .
    ## [27,] 1 . 1 . . . . 1 . 1 1 . . . 1 . . 1 . . . . 1 . . . 1 . . 1
    ## [28,] . . . . . . 1 . . . . . . . . . 1 . . . 1 . . 1 . 1 . 1 . .
    ## [29,] . . . . 1 . . . . . . . . 1 . . . . 1 1 1 . . 1 . . . . 1 .
    ## [30,] . . . . . . . 1 . . 1 . . . 1 . . 1 . . . . 1 . . . 1 . . 1

#### Vector/Lists of indices that put in relation the observed, possibly redundant observed\_locs and the reordered, non-redundant locs.

The first one is a vector that matches the rows of observed\_locs with
the rows of locs. It allows to use locs to recreate observed\_locs (see
below). Its length is equal to the number of rows of
mcmc\_nngp\_list$observed\_locs and its values range from 1 to the
number of rows of mcmc\_nngp\_list$locs. This means that there are
potentially redundant indices in this vector (if and only if redundant
spatial locations are observed).

``` r
print(mcmc_nngp_list$vecchia_approx$locs_match[1:100])
```

    ##   [1] 1770 1953 1917 1103  487 1070  323 1853 1936 1472  230  267 1378 1534 1059
    ##  [16]  151 1873   71  761 1066  575 1194 1524 1823 1696 1459 1570 1926  273 1758
    ##  [31]  618 1929 1181  783 1664 1752 1501 1689 1727 1315  769 1381 1425  533  601
    ##  [46] 1158  529 1994  737 1156 1111 1807 1987   29 1800  202  231  920  948 1574
    ##  [61]  466  379  297 1848  291 1128 1408  531 1267  406  586  883  390  139 1999
    ##  [76] 1015  456 1137 1620 1760 1730 1262  823 1882 1767  252  412 1384 1814 1710
    ##  [91] 1168  923  695  272 1361  919 1240 1747  888 1714

``` r
head(mcmc_nngp_list$observed_locs)
```

    ##          [,1] [,2]
    ## [1,] 132.7543 1.01
    ## [2,] 186.0619 1.00
    ## [3,] 286.4267 1.00
    ## [4,] 454.1039 1.00
    ## [5,] 100.8410 1.00
    ## [6,] 449.1948 1.00

``` r
head(mcmc_nngp_list$locs[mcmc_nngp_list$vecchia_approx$locs_match,])
```

    ##          [,1] [,2]
    ## [1,] 132.7543 1.01
    ## [2,] 186.0619 1.00
    ## [3,] 286.4267 1.00
    ## [4,] 454.1039 1.00
    ## [5,] 100.8410 1.00
    ## [6,] 449.1948 1.00

The second one is a list of vectors that matches the rows of locs to the
rows of observed\_locs. It is a list and not a vector because there can
be duplicates in observed\_locs, so various rows of observed\_locs can
be matched to the same row of locs. As a result, the list’s length is
the same as the number of rows of locs, but the sum of the lengths of
the elements is equal to the number of rows of observed\_locs. Its
(difficult to pronounce) name is the reverse of the first index vector
name : locs\_match \|\| hctam\_scol

``` r
# It's a list
print(mcmc_nngp_list$vecchia_approx$hctam_scol[1:10])
```

    ## $`1`
    ## [1] 214
    ## 
    ## $`2`
    ## [1] 177
    ## 
    ## $`3`
    ## [1] 1818
    ## 
    ## $`4`
    ## [1] 911
    ## 
    ## $`5`
    ## [1] 1862
    ## 
    ## $`6`
    ## [1] 1855
    ## 
    ## $`7`
    ## [1] 1687
    ## 
    ## $`8`
    ## [1] 1149
    ## 
    ## $`9`
    ## [1] 1389
    ## 
    ## $`10`
    ## [1] 109

The third is a vector obtained by selecting only the first element of
each vector of the previous list. It allows to recreate locs from
observed\_locs. It has the same length as locs.

``` r
print(mcmc_nngp_list$vecchia_approx$hctam_scol_1[1:100])
```

    ##    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16 
    ##  214  177 1818  911 1862 1855 1687 1149 1389  109  251  502 1070  156  580  735 
    ##   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31   32 
    ##  259  714  461 1352 1067 1537 1154  618  585  556 1882  113   54 1630  754  755 
    ##   33   34   35   36   37   38   39   40   41   42   43   44   45   46   47   48 
    ## 1967 1436  475  516 1157 1264 1886 1307 1673 1738  701 1659  783  476  423 1472 
    ##   49   50   51   52   53   54   55   56   57   58   59   60   61   62   63   64 
    ##  796  322  490  860 1473 1134  614  302 1563 1242  874  682  857 1051 1373 1803 
    ##   65   66   67   68   69   70   71   72   73   74   75   76   77   78   79   80 
    ## 1515  767  187 1379 1466 1327   18  628 1246 1954  312 1851 1503  286  396  931 
    ##   81   82   83   84   85   86   87   88   89   90   91   92   93   94   95   96 
    ##  478  645 1001 1010 1400  489  126 1532 1905 1435 1404 1152  533  944 1250  890 
    ##   97   98   99  100 
    ##  788  732 2000 1509

``` r
head(mcmc_nngp_list$observed_locs[mcmc_nngp_list$vecchia_approx$hctam_scol_1,]) 
```

    ##           [,1] [,2]
    ## [1,] 473.98318    1
    ## [2,] 322.15788    1
    ## [3,] 497.64053    1
    ## [4,] 261.23064    1
    ## [5,]  17.08532    1
    ## [6,] 212.43027    1

``` r
head(mcmc_nngp_list$locs)
```

    ##           [,1] [,2]
    ## [1,] 473.98318    1
    ## [2,] 322.15788    1
    ## [3,] 497.64053    1
    ## [4,] 261.23064    1
    ## [5,]  17.08532    1
    ## [6,] 212.43027    1

The Markov Chain states
-----------------------

mcmc\_nngp\_list$states is a list with 2 or more sublists, each
corresponding to one chain. For each chain, the transition kernels are
adapted in the first hundred iterations. They are stored in one sublist.
The other sublist contains the state of the model current parameters.

``` r
print(mcmc_nngp_list$states$chain_1$transition_kernels) # the transition kernels
```

    ## $covariance_params_sufficient
    ## $covariance_params_sufficient$logvar
    ## [1] -2
    ## 
    ## 
    ## $covariance_params_ancillary
    ## $covariance_params_ancillary$logvar
    ## [1] -2
    ## 
    ## 
    ## $log_noise_variance
    ## $log_noise_variance$logvar
    ## [1] -1

``` r
print(mcmc_nngp_list$states$chain_1$params$beta_0)# intercept
```

    ## (Intercept) 
    ##     2.88572

``` r
print(mcmc_nngp_list$states$chain_1$params$beta)# other regression coefficients
```

    ##       X$Xslope X$Xwhite_noise 
    ##    0.003415492   -1.646223863

``` r
print(mcmc_nngp_list$states$chain_1$params$log_scale)# log of the scale parameter
```

    ## [1] 1.744899

``` r
print(mcmc_nngp_list$states$chain_1$params$shape) # other covariance parameters : log-range, smoothness
```

    ## [1] -2.83484

``` r
print(mcmc_nngp_list$states$chain_1$params$log_noise_variance) # log-variance of the Gaussian noise
```

    ## [1] 1.975535

``` r
print(mcmc_nngp_list$states$chain_1$params$field[seq(100)])# latent field
```

    ##   [1] -0.29683954  2.96964453  4.31028042  3.40020352  1.69947380  3.81411024
    ##   [7]  3.33643351  3.22393691  0.33349991  4.75841789  8.89586943 -0.41394711
    ##  [13]  1.60159764  0.89021745  5.53731391  3.38270822  1.03282706  2.68300514
    ##  [19] -0.13157718  3.14527834  1.98641321  3.77292524  5.18444521  2.92069576
    ##  [25]  5.91340613  6.64042396  4.47413533  1.40515403  8.24330977 -3.17585285
    ##  [31] -0.01246357  0.04652107  0.91310742  1.99617089  2.81133644  6.86003740
    ##  [37]  2.80643458  5.24685774  4.39357958  2.45139310  3.19248503  0.62205815
    ##  [43]  5.66393831  4.19478179 -2.41933187  5.38971675  9.49883707  3.74628455
    ##  [49]  0.67788626  3.85645926  4.48229267  2.04625598  2.42280888  1.25150564
    ##  [55]  1.99265569  2.92310534 -2.93486149  2.39181101  4.15543708  3.30782569
    ##  [61]  3.23123245  2.45633767  8.82814332  5.36992341  2.55148786 -0.24210552
    ##  [67]  3.34961465  3.68635597 -1.41751586  0.60719243  5.24759683  1.18257661
    ##  [73]  4.44251162  6.55219895  0.35298164  2.60918835  8.13635013  2.98265358
    ##  [79] -1.59361787  2.41936024  2.96167474 -1.06563059  4.88272898  7.55102008
    ##  [85]  0.86864038  2.83183263  1.13696151  2.77236465  6.02777251  2.75878449
    ##  [91]  6.44769815  3.71925996  3.13790800  3.96588174  3.74443281  4.78442725
    ##  [97]  3.60061654  2.23913770  3.48656899  6.08800091

Records of the chain states
---------------------------

There is one record per chain. For each chain, iterations is a two
column matrix that gives info about the number of iterations done and
the time from setup. params is a list that keeps the chain states. For
now, they are empty since the chains were not run. When the chains start
running, the records start filling.

``` r
print(mcmc_nngp_list$records) 
```

    ## $chain_1
    ## $chain_1$iterations
    ##      iteration     time
    ## [1,]         0 1.235978
    ## 
    ## $chain_1$params
    ## list()
    ## 
    ## 
    ## $chain_2
    ## $chain_2$iterations
    ##      iteration     time
    ## [1,]         0 1.236422
    ## 
    ## $chain_2$params
    ## list()
    ## 
    ## 
    ## $chain_3
    ## $chain_3$iterations
    ##      iteration     time
    ## [1,]         0 1.236476
    ## 
    ## $chain_3$params
    ## list()

Diagnostics
-----------

Gelman-Rubin-Brooks diagnostics are computed while the chain runs. They
are stocked here. For now, there is nothing.

``` r
print(mcmc_nngp_list$diagnostics) 
```

    ## $Gelman_Rubin_Brooks
    ## list()

Miscellaneous
-------------

``` r
print(mcmc_nngp_list$t_begin) # the time setup was done 
```

    ## [1] "2020-09-22 13:36:53 CEST"

``` r
print(mcmc_nngp_list$seed) # the seed 
```

    ## [1] 1

Let’s fit the model
===================

Now, we will fit the model. The “states”, “records” and “diagnostics”
part of mcmc\_vecchia\_list are updated while the rest does not change.
The chains will run in parallel and join each other once in a while.
When the whains are joined, Gelman-Rubin-Brooks diagnostics are
computed, the chains are plotted We use the function mcmc\_vecchia\_run
with arguments :

-   mcmc\_vecchia\_list, the object we just created and examined

-   n\_cores : the number of cores used

-   n\_iterations\_update : the number of iterations between each join
    of the chains.

-   n\_cycles : the number of updates cycles that are done. This means
    that the Gibbs sampler is iterated n\_cycles \*
    n\_iterations\_update

-   burn\_in : a proportion between 0 and 1 of the discarded states
    before computing Gelman-Rubin-Brooks diagnostics and plotting the
    chains

-   field\_thinning : a proportion between 0 (excluded) and 1 of the
    field samples that are saved.

-   Gelman\_Rubin\_Brooks\_stop : a vector of two numbers bigger than 1,
    an automatic stop using Gelman-Rubin-Brooks diagnostics. Univariate
    and multivariate Gelman-Rubin-Brooks diagnostics are computed on the
    hugh-level parameters (covariance, noise variance, fix effects). If
    either the multivariate or all univariate diagnostics fall below
    thir respective thresold, the function stops and the rest of the
    scheduled iterations is not done. If it is set to c(1, 1), all the
    epochs are done.

-   ancillary : whether ancillary covariance parameters updates are
    done. True by default and better left True all the time.

-   n\_chromatic : number of chromatic update per iterations, better to
    do a couple from our experience

burn-in : run the chains (almost) wihout saving the field (field\_thinning = 0.01)
----------------------------------------------------------------------------------

``` r
source("Scripts/mcmc_nngp_diagnose.R")
source("Scripts/mcmc_nngp_run.R")
source("Scripts/mcmc_nngp_update_Gaussian.R")
mcmc_nngp_list =  mcmc_nngp_run(mcmc_nngp_list, n_cores = 3, 
                                      n_cycles = 5, n_iterations_update = 200,  ancillary = T, n_chromatic = 5, 
                                      burn_in = .5, field_thinning = 0.01, Gelman_Rubin_Brooks_stop = c(1.00, 1.00))
```

    ## [1] "cycle = 1"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           9.130233           1.071222           1.035781           1.020273 
    ##          log_scale log_noise_variance              shape 
    ##           1.202292           1.075488           1.102439 
    ## [1] "cycle = 2"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           8.632773           1.009672           1.016464           1.020748 
    ##          log_scale log_noise_variance              shape 
    ##           3.358741           1.093527           3.188239 
    ## [1] "cycle = 3"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           5.010261           1.004661           1.007641           1.017572 
    ##          log_scale log_noise_variance              shape 
    ##           1.181133           1.263868           1.922709 
    ## [1] "cycle = 4"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           4.287606           1.004215           1.007968           1.007166 
    ##          log_scale log_noise_variance              shape 
    ##           1.381336           1.321231           1.894245 
    ## [1] "cycle = 5"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           2.945314           1.002223           1.009088           1.001459 
    ##          log_scale log_noise_variance              shape 
    ##           1.434431           1.061791           1.294592

Run the chains until all individual Gelman-Rubin-Brooks diagnostics drop below 1.05
-----------------------------------------------------------------------------------

``` r
mcmc_nngp_list =  mcmc_nngp_run(mcmc_nngp_list, n_cores = 3,
                                      n_cycles = 1000, n_iterations_update = 100,  
                                      burn_in = .5, field_thinning = .2, Gelman_Rubin_Brooks_stop = c(1.00, 1.05))
```

    ## [1] "cycle = 1"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           4.395414           1.002914           1.010965           1.002501 
    ##          log_scale log_noise_variance              shape 
    ##           1.325413           1.049621           1.312379 
    ## [1] "cycle = 2"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           4.351953           1.002649           1.007953           1.004993 
    ##          log_scale log_noise_variance              shape 
    ##           1.255406           1.105365           1.439968 
    ## [1] "cycle = 3"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           3.198014           1.004167           1.006240           1.003271 
    ##          log_scale log_noise_variance              shape 
    ##           1.311836           1.129868           1.581264 
    ## [1] "cycle = 4"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.782138           1.002550           1.004306           1.002831 
    ##          log_scale log_noise_variance              shape 
    ##           1.080554           1.092401           1.200965 
    ## [1] "cycle = 5"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.611165           1.002089           1.002277           1.003096 
    ##          log_scale log_noise_variance              shape 
    ##           1.060195           1.081875           1.221441 
    ## [1] "cycle = 6"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.504148           1.000875           1.001665           1.005696 
    ##          log_scale log_noise_variance              shape 
    ##           1.036385           1.083714           1.174879 
    ## [1] "cycle = 7"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.407107           1.000904           1.001380           1.004294 
    ##          log_scale log_noise_variance              shape 
    ##           1.015215           1.061181           1.110709 
    ## [1] "cycle = 8"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.196063           1.000566           1.000794           1.004396 
    ##          log_scale log_noise_variance              shape 
    ##           1.007447           1.051480           1.054739 
    ## [1] "cycle = 9"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.126061           1.000547           1.001047           1.005393 
    ##          log_scale log_noise_variance              shape 
    ##           1.009078           1.015847           1.015056

Run the chains 1000 more iterations just to be sure
---------------------------------------------------

``` r
mcmc_nngp_list =  mcmc_nngp_run(mcmc_nngp_list, n_cores = 3,
                                      n_cycles = 10, n_iterations_update = 100,  
                                      burn_in = .5, field_thinning = .2, Gelman_Rubin_Brooks_stop = c(1.00, 1.00))
```

    ## [1] "cycle = 1"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.117261           1.000625           1.001022           1.001880 
    ##          log_scale log_noise_variance              shape 
    ##           1.023800           1.002952           1.012217 
    ## [1] "cycle = 2"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.366930           1.000508           1.000913           1.001293 
    ##          log_scale log_noise_variance              shape 
    ##           1.079828           1.003541           1.075229 
    ## [1] "cycle = 3"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.331625           1.001267           1.000561           1.001470 
    ##          log_scale log_noise_variance              shape 
    ##           1.120416           1.012439           1.172954 
    ## [1] "cycle = 4"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.221022           1.000880           1.000483           1.000573 
    ##          log_scale log_noise_variance              shape 
    ##           1.113126           1.020907           1.163202 
    ## [1] "cycle = 5"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.253131           1.001153           1.000465           1.000559 
    ##          log_scale log_noise_variance              shape 
    ##           1.127849           1.026441           1.186422 
    ## [1] "cycle = 6"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.333779           1.001180           1.000513           1.000727 
    ##          log_scale log_noise_variance              shape 
    ##           1.099803           1.018580           1.147709 
    ## [1] "cycle = 7"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.449764           1.001553           1.000534           1.000610 
    ##          log_scale log_noise_variance              shape 
    ##           1.106582           1.021749           1.182535 
    ## [1] "cycle = 8"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.168451           1.001749           1.000480           1.000741 
    ##          log_scale log_noise_variance              shape 
    ##           1.107167           1.012239           1.132251 
    ## [1] "cycle = 9"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.191262           1.001208           1.000484           1.000652 
    ##          log_scale log_noise_variance              shape 
    ##           1.074294           1.004777           1.081977 
    ## [1] "cycle = 10"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           1.398074           1.000605           1.000690           1.000467 
    ##          log_scale log_noise_variance              shape 
    ##           1.060026           1.005558           1.088994

Chains plotting
===============

Normally, plotting is done each time the chains join. This allows to
monitor the progress of the fitting along with Gelman-Rubin-Brooks
diagnostics. Here, we de-activated it in the Rmarkdown options in order
to keep the document readable. Let’s plot the chains. We must input the
records of the chains, and the burn in (the proportion of observations
that are discarded).

``` r
raw_chains_plots_covparms(records = mcmc_nngp_list$records, burn_in = 0.01, n_chains = 1)
```

![](Vignette_files/figure-markdown_github/unnamed-chunk-13-1.png)![](Vignette_files/figure-markdown_github/unnamed-chunk-13-2.png)![](Vignette_files/figure-markdown_github/unnamed-chunk-13-3.png)![](Vignette_files/figure-markdown_github/unnamed-chunk-13-4.png)

``` r
raw_chains_plots_beta(records = mcmc_nngp_list$records, burn_in = 0.01, n_chains = 1)
```

![](Vignette_files/figure-markdown_github/unnamed-chunk-14-1.png)![](Vignette_files/figure-markdown_github/unnamed-chunk-14-2.png)

Parameters estimation
=====================

The function takes as arguments the list we created and updated
previously, and a burn-in proportion between 0 and 1. Each estimation
includes the mean, quantiles 0.025, 0.5, 0.975, and the standard
deviation. There are estimations of :

-   The covariance parameters (in various parametrizations :
    log-transformed, GpGp, INLA)

-   The intercept and the other regression coefficients (that correspond
    to the inputed X, not the centered X that is used during fitting)

-   The latent field

``` r
source("Scripts/mcmc_nngp_estimate.R")
estimations  = mcmc_nngp_estimate(mcmc_nngp_list, burn_in = .5)
print(estimations$covariance_params$GpGp_covparams)
```

    ##                     mean   q0.025    median    q0.975        sd
    ## scale          11.297571 8.284917 11.113068 15.874456 1.9145750
    ## noise_variance  5.317137 4.901220  5.302875  5.837934 0.2333184
    ## range           5.974293 4.031685  5.786091  9.174549 1.3106910

``` r
print(estimations$fixed_effects)
```

    ##                        mean       q0.025       median      q0.975          sd
    ## beta_0          2.160527761  0.145642483  2.162140291  4.14821057 1.015581836
    ## X$Xslope        0.003080794 -0.003739017  0.002977385  0.01005237 0.003513744
    ## X$Xwhite_noise -1.597578383 -1.707732234 -1.596760420 -1.49170653 0.055823760
    ##                zero_out_of_ci
    ## beta_0                      1
    ## X$Xslope                    0
    ## X$Xwhite_noise              1

``` r
head(estimations$field)
```

    ##            mean     q0.025     median     q0.975       sd
    ## [1,]  4.0566791  1.5985379  4.0680578  6.5593256 1.294413
    ## [2,] -1.4316042 -3.6184625 -1.4375123  0.6084193 1.035347
    ## [3,] -3.0600557 -5.4171875 -3.0407646 -0.8044955 1.187430
    ## [4,]  2.0982390 -0.2480847  2.1455991  4.6216580 1.266173
    ## [5,] -2.7607445 -5.1636284 -2.8109128 -0.5583597 1.172682
    ## [6,] -0.6677489 -3.2092631 -0.6306473  1.7653234 1.261741

Prediction
==========

Prevision of the latent field
-----------------------------

The previsions of the latent field at unobserved locations demans to
have the chains and the predicted locations. Several cores can work in
parallel. Like before, m is the number of neighbors used to compute
Vecchia approximations. Prediction can be done only when the field state
is recorded, so a low field\_thinning parameter will result in scant
prediction samples. We can use burn\_in to precise the proportion of
samples that are left out of the original records.

Let’s take an example : if a chain did 1000 iterations with
field\_thinning = .5, only one state out of two will be saved. If
burn\_in = .2, only the states after iteration .2 × 1000 = 200 will be
used. Then, the predictions will rely on (1000 − 2 × 100) × .5 = 400
states.

The outputs are :

-   The locations where the prediction is done

-   Prediction samples

-   Prediction summaries (mean, quantiles 0.025, 0.5, 0.975, and
    standard deviation)

``` r
source("Scripts/mcmc_nngp_predict.R")
predicted_locs = cbind(seq(0, 500, .01), 1)
predicted_locs[1, 2] = 1.01
predictions = mcmc_nngp_predict_field(mcmc_nngp_list, predicted_locs = predicted_locs, n_cores = 3, m = 5)
plot(locs[,1], field)
lines( predicted_locs[,1], predictions$predicted_field_summary[,"mean"], col  = 2)
legend("topright", legend = c("true latent field", "prediction"), fill = c(1, 2))
```

![](Vignette_files/figure-markdown_github/unnamed-chunk-16-1.png)

Prevision of the fix effects
----------------------------

The previsions of the fix effects at unobserved locations demand to have
the chains and the values of the regressors at the predicted locations.
Like before, several cores can work in parallel and a burn in parameter
must be set. The names of X will be matched with the names of the
regressors given in the initialization, and X must be a data.frame.
Column subsets of the regressors are accepted, so that one can evaluate
the effect of one single regressor or a group. It is possible to match
field thinning, in order to produce samples of the fix effects only
where the latent field is recorded and combine the samples. The option
add\_intercept, FALSE by default, allows to add the intercept to the fix
effects.

The outputs are :

-   The regressors at the predicted locations

-   Samples from the predicted fix effects

-   Summary of the predictions

``` r
X_pred = as.data.frame(predicted_locs[,1])
names(X_pred) = "slope"
predictions = mcmc_nngp_predict_fixed_effects(mcmc_nngp_list = mcmc_nngp_list, X_predicted =  X_pred, burn_in = .5, n_cores = 3, match_field_thinning = F, add_intercept = T)
plot(locs[,1], observed_field)
lines( predicted_locs[,1], predictions$predicted_fixed_effects_summary[,"mean"], col  = 2)
legend("topright", legend = c("observed field", "predicted slope effect"), fill = c(1, 2))
```

![](Vignette_files/figure-markdown_github/unnamed-chunk-17-1.png)

Using interweaving for the regression coefficients
==================================================

Some regressors that have some kind of space coherence can interfere
with the latent field. The paper proposes a way to address the problem,
and it is implemented here. The problem is that it works for regressors
that do not vary within a spatial location. But since NNGP is by essence
a method on spatial points, data from spatial grids and areas are
immediately elegible.

Let’s fit the model just like before, but passing the regressors as
X\_obs.

We can see that the Gelman-Rubin-Brooks diagnostics of Xslope are
terrible. The raw chains confirm that Xslope does not mix, while
Xwhite\_noise does. This is because of the fact that Xslope has some
spatial coherence, while the other variable has not. The conclusion is
that whenever possible, variables you suspect to have a spatial
coherence should be input as X\_locs.

``` r
source("Scripts/mcmc_nngp_initialize.R")


# Now, let's initialize the list. This creates the chains, guesses the initial states, does the Nearest Neighbor search for NNGP, etc.... 
mcmc_nngp_list = mcmc_nngp_initialize(observed_locs = locs, observed_field = observed_field, 
                                            stationary_covfun = "exponential_isotropic", 
                                            X_obs = as.data.frame(X),
                                            m = 5, 
                                            seed = 1)
```

    ## [1] "Setup done, 0.448009490966797 s elapsed"

``` r
mcmc_nngp_list =  mcmc_nngp_run(mcmc_nngp_list, n_cores = 3,
                                      n_cycles = 5, n_iterations_update = 200,  
                                      burn_in = .5, field_thinning = 0.01, Gelman_Rubin_Brooks_stop = c(1.00, 1.00))
```

    ## [1] "cycle = 1"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##         170.014463           1.005320           3.552945           1.034448 
    ##          log_scale log_noise_variance              shape 
    ##           1.665247           1.185107           1.560121 
    ## [1] "cycle = 2"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##         112.565240           1.011866           2.805420           1.034671 
    ##          log_scale log_noise_variance              shape 
    ##           1.088062           1.062049           1.565563 
    ## [1] "cycle = 3"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##          59.046057           1.002437           3.748876           1.005206 
    ##          log_scale log_noise_variance              shape 
    ##           1.053912           1.054917           1.026351 
    ## [1] "cycle = 4"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##          89.968481           1.010518           2.186869           1.009562 
    ##          log_scale log_noise_variance              shape 
    ##           1.365080           1.017303           1.372133 
    ## [1] "cycle = 5"

    ## [1] "Gelman-Rubin-Brooks R-hat : "
    ##       Multivariate             beta_0           X$Xslope     X$Xwhite_noise 
    ##           4.973723           1.002074           1.024530           1.008644 
    ##          log_scale log_noise_variance              shape 
    ##           1.059062           1.004335           1.051531

``` r
raw_chains_plots_beta(mcmc_nngp_list$records, burn_in = .01, n_chains = 1)
```

![](Vignette_files/figure-markdown_github/unnamed-chunk-19-1.png)![](Vignette_files/figure-markdown_github/unnamed-chunk-19-2.png)

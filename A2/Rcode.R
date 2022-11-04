setwd('/Users/damarisdeng/BIM Assignments/BIM3008-Assignment/A2')
#q2
library(MASS)
# set seed and create data vectors
set.seed(0)
sample_size <- 100                                       
sample_meanvector <- c(10, 5)                                   
sample_covariance_matrix <- matrix(c(10, 5, 2, 9),
                                   ncol = 2)
# create bivariate normal distribution
sample_dist <- mvrnorm(n = sample_size,
                       mu = sample_meanvector, 
                       Sigma = sample_covariance_matrix)

# print top of distribution
head(sample_dist)
# calculate m and s from data
m = colMeans(sample_dist) # the output is 9.871285 5.013477
s = var(sample_dist)
print(s)
#output
#[,1]     [,2]
#[1,] 7.231733 3.641243
#[2,] 3.641243 8.255741

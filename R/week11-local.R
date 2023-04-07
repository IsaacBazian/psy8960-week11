## Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
library(tictoc)
library(parallel)
library(doParallel)
set.seed(2112)


## Data Import and Cleaning
gss_tbl_original <- read_sav(file = "../data/GSS2016.sav") %>% 
  filter(!is.na(MOSTHRS)) %>% 
  select(-HRS1, -HRS2) #This code drops the HRS1 and HRS2 variables, so we do not predict workhours from work hours

missing_75 <- colMeans(is.na(gss_tbl_original)) >= .75

gss_tbl <- gss_tbl_original[,!missing_75] %>% 
  sapply(as.numeric) %>% 
  as_tibble() %>% 
  rename(workhours = MOSTHRS)

## Visualization
ggplot(gss_tbl, aes(x = workhours)) +
  geom_histogram() +
  labs(x = "Number of Hours Worked Last Week", y = "Number of Respondents")

## Analysis
gss_shuffled_tbl <- gss_tbl[sample(nrow(gss_tbl)),]
split75 <- round(nrow(gss_shuffled_tbl) * .75)
gss_train_tbl <- gss_shuffled_tbl[1:split75,]
gss_test_tbl <- gss_shuffled_tbl[(split75 + 1):nrow(gss_shuffled_tbl),]
training_folds <- createFolds(gss_train_tbl$workhours, 10)

tic() #These calls of tic and toc record the runtime of each model train
modelOLS <- train(
  workhours ~ .,
  gss_train_tbl,
  method = "lm",
  metric = "Rsquared",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method="cv", indexOut = training_folds, number = 10, search = "grid", verboseIter=T)
)
tocOLS <- toc() #Runtime of each model assigned to an object like this for later Publication section

tic()
modelElasticNet <- train(
  workhours ~ .,
  gss_train_tbl,
  method = "glmnet",
  metric = "Rsquared",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method="cv", indexOut = training_folds, number = 10, search = "grid", verboseIter=T)
)
tocElasticNet <- toc()

tic()
modelRandomForest <- train(
  workhours ~ .,
  gss_train_tbl,
  method = "ranger",
  metric = "Rsquared",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method="cv", indexOut = training_folds, number = 10, search = "grid", verboseIter=T),
  tuneLength = 3
)
tocRandomForest <- toc()

tic()
modelXGB <- train(
  workhours ~ .,
  gss_train_tbl,
  method = "xgbLinear",
  metric = "Rsquared",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method="cv", indexOut = training_folds, number = 10, search = "grid", verboseIter=T),
  tuneLength = 3
)
tocXGB <- toc()





# The following code sets up parallelization, runs the same models again, and times them
local_cluster <- makeCluster(7)
registerDoParallel(local_cluster)

tic()
modelOLSPar <- train(
  workhours ~ .,
  gss_train_tbl,
  method = "lm",
  metric = "Rsquared",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method="cv", indexOut = training_folds, number = 10, search = "grid", verboseIter=T)
)
tocOLSPar <- toc()

tic()
modelElasticNetPar <- train(
  workhours ~ .,
  gss_train_tbl,
  method = "glmnet",
  metric = "Rsquared",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method="cv", indexOut = training_folds, number = 10, search = "grid", verboseIter=T)
)
tocElasticNetPar <- toc()

tic()
modelRandomForestPar <- train(
  workhours ~ .,
  gss_train_tbl,
  method = "ranger",
  metric = "Rsquared",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method="cv", indexOut = training_folds, number = 10, search = "grid", verboseIter=T),
  tuneLength = 3
)
tocRandomForestPar <- toc()

tic()
modelXGBPar <- train(
  workhours ~ .,
  gss_train_tbl,
  method = "xgbLinear",
  metric = "Rsquared",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method="cv", indexOut = training_folds, number = 10, search = "grid", verboseIter=T),
  tuneLength = 3
)
tocXGBPar <- toc()


#Stop parallelization
stopCluster(local_cluster)
registerDoSEQ()




## Publication
table1_tbl <- tibble(
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  cv_rsq = c(
    str_remove(format(round(modelOLS$results$Rsquared, 2), nsmall = 2), pattern = "^0"),
    str_remove(format(round(max(modelElasticNet$results$Rsquared), 2), nsmall = 2), pattern = "^0"),
    str_remove(format(round(max(modelRandomForest$results$Rsquared), 2), nsmall = 2), pattern = "^0"),
    str_remove(format(round(max(modelXGB$results$Rsquared), 2), nsmall = 2), pattern = "^0")
  ),
  ho_rsq = c(
    str_remove(format(round(cor(predict(modelOLS, gss_test_tbl, na.action = na.pass), gss_test_tbl$workhours)^2, 2), nsmall = 2), pattern = "^0"),
    str_remove(format(round(cor(predict(modelElasticNet, gss_test_tbl, na.action = na.pass), gss_test_tbl$workhours)^2, 2), nsmall = 2), pattern = "^0"),
    str_remove(format(round(cor(predict(modelRandomForest, gss_test_tbl, na.action = na.pass), gss_test_tbl$workhours)^2, 2), nsmall = 2), pattern = "^0"),
    str_remove(format(round(cor(predict(modelXGB, gss_test_tbl, na.action = na.pass), gss_test_tbl$workhours)^2, 2), nsmall = 2), pattern = "^0")
  )
) 


# This code makes a tibble of runtime for each model run sequentially and parallel
table2_tbl <- tibble(
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  original = c(tocOLS$callback_msg, tocElasticNet$callback_msg, tocRandomForest$callback_msg, tocXGB$callback_msg),
  parallelized = c(tocOLSPar$callback_msg, tocElasticNetPar$callback_msg, tocRandomForestPar$callback_msg, tocXGBPar$callback_msg)
)

#Q1: It seems that eXtreme Gradient Boosting benefited the most from parallelization,
#shaving roughly 25 seconds off the runtime. I imagine that this is because it
#has a relatively long runtime relative to OLS and Elastic Net, so there's actually
#space to notice considerable improvement. The Random Forest seemed to not improve
#much, despite also being a relatively long run time - based on some quick Googling,
#the 'ranger' method may already be using some kind of processing tricks that
#make parallelization less relatively impactful. I have since come back a few times,
#and on every occassion I find that the parallelized and non-parallelized RF
#is roughly the same run-time, and in fact sometimes the parallelized is slightly
#slower. I have also talked to some people using Mac machines, and they get different
#runtimes across most algos, including better reductions in run-time for the RF.
#I assume some kind of hardware thing may be going on, and I will not worry about
#it for now.

#Q2: The fastest parallelized model was the Elastic Net at roughly 1 second, and
#the slowest was the XGB at roughly 87 seconds, for a differences of roughly 86 seconds.
#However, it is worth noting that the OLS model bears the brunt of the startup time
#when initiating parallelization. XGB was already slower than EN before parallelization,
#by roughly 110 seconds, so the gap certainly narrowed, but XGB is still a more
#complex algorithm than EN due to more hyperparameters, so it's only natural that
#it would still take longer, even if they both benefit from parallelization.

#Q3: I would still recommend the Random Forest model in this case - it still
#produces the highest ho_rsq, which is the prediction metric we care about,
#and even with parallelization it is faster than XGB. Having said that, it would
#be worth comparing prediction and speed if different data is being used or more
#processing power is available - with enough cores and large enough datasets, 
#the XGB model may run much faster than the Random Forest, which would potentially
#be more valuable than marginal gains in predictive power (and if XGB is both faster
#and better at predicting, it becomes no contest)










## Script Settings and Resources
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
  select(-HRS1, -HRS2)

missing_75 <- colMeans(is.na(gss_tbl_original)) >= .75

gss_tbl <- gss_tbl_original[,!missing_75] %>% 
  sapply(as.numeric) %>% 
  as_tibble() %>% 
  rename(workhours = MOSTHRS)

## Analysis
gss_shuffled_tbl <- gss_tbl[sample(nrow(gss_tbl)),]
split75 <- round(nrow(gss_shuffled_tbl) * .75)
gss_train_tbl <- gss_shuffled_tbl[1:split75,]
gss_test_tbl <- gss_shuffled_tbl[(split75 + 1):nrow(gss_shuffled_tbl),]
training_folds <- createFolds(gss_train_tbl$workhours, 10)

tic()
modelOLS <- train(
  workhours ~ .,
  gss_train_tbl,
  method = "lm",
  metric = "Rsquared",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(method="cv", indexOut = training_folds, number = 10, search = "grid", verboseIter=T)
)
tocOLS <- toc()

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




local_cluster <- makeCluster(14)
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


stopCluster(local_cluster)
registerDoSEQ()




## Publication
Table3 <- tibble(
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


Table4 <- tibble(
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  supercomputer = c(tocOLS$callback_msg, tocElasticNet$callback_msg, tocRandomForest$callback_msg, tocXGB$callback_msg),
  "supercomputer-14" = c(tocOLSPar$callback_msg, tocElasticNetPar$callback_msg, tocRandomForestPar$callback_msg, tocXGBPar$callback_msg)
)


#Output tables as csv
write_csv(Table3, "table3.csv")
write_csv(Table4, "table4.csv")





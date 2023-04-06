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
  #tuneGrid = expand.grid(mtry = c(2, 10, 50, 100, 200), splitrule = c("variance", "extratrees"), min.node.size = 5) #This seems to run a little faster. Commented out, as just letting tuneLength run wasn't that bad in the end
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



Table2_tbl <- tibble(
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  original = c(tocOLS$callback_msg, tocElasticNet$callback_msg, tocRandomForest$callback_msg, tocXGB$callback_msg),
  parallelized = c("", "", "", "")
)











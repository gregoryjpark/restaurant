setwd("~/Dropbox/kaggle/restaurant/")
library(randomForest)
library(caret)
library(stringr)
#library(doMC) 
#registerDoMC(6)

# binarify() takes a dataframe (df) and a categorical variable (cat.var),
# and returns the dataframe with additional columns: one for each
# unique value of the categorical variable, each filled with 0/1 
# to indicate the observations value.

binarify <- function(df, cat.var, suffix="", all.levels=T){
  if(all.levels==T){
    num.levels <- length(unique(df[,cat.var])) 
  }
  else{
    num.levels <- length(unique(df[,cat.var])) - 1
  }

  for (i in c(1:num.levels))
  {
    df[,paste(cat.var, "_", 
              str_replace_all(names(table(df[,cat.var]))[i], pattern="\\s",
              replacement=""), suffix, sep="")] <- 
                   ifelse(df[,cat.var] == names(table(df[,cat.var]))[i], 1, 0)
  }
  return(df)
}

# expRMSE() is an evaluation metric passed to caret
# the model is trained on a log-transformed response,
# then back-transformed to the original scale for evaluation

expRMSE <- function(data, lev=NULL, model=NULL){
  out <- myRMSE(exp(data[,"obs"]), exp(data[,"pred"]))
  names(out) <- "expRMSE"
  out
}

# vanilla RMSE, expRMSE wraps around this
myRMSE <- function(a, p){
  sq.error <- (a - p)^2
  mean.sq.error <- mean(sq.error)
  rmse <- sqrt(mean.sq.error)
	return(rmse)
}

train_data <- read.csv("train.csv", stringsAsFactors=F)
test_data <- read.csv("test.csv", stringsAsFactors=F)

data_y <- train_data$revenue 
train_data$revenue <- NULL

data <- rbind(train_data, test_data)
train_index <- 1:nrow(train_data)
test_index <- (nrow(train_data)+1):nrow(data)

# process Open.Date column
data$Open.Date <- as.Date(data$Open.Date, format="%m/%d/%Y")
data$opendate_numeric <- as.numeric(data$Open.Date)
data$opendate_month <- as.numeric(format(data$Open.Date, "%m"))
data$opendate_day <- as.numeric(format(data$Open.Date, "%d"))
data$opendate_year <- as.numeric(format(data$Open.Date, "%Y"))

# add features based on City.Group, Type 
# 'Type' encoding based on Jamie Ross's benchmark
data <- binarify(data, "City.Group")
data$Type <- ifelse(data$Type == "DT" | data$Type == "MB", "Other", data$Type)
data <- binarify(data, "Type")
data$instabul <- ifelse(data$City=="İstanbul", 1, 0)
data$ankara <- ifelse(data$City=="Ankara", 1, 0)

# expand all 'P' features to binary encodings
pcols <- paste0("P", 1:37)

for (var in pcols){
   data <- binarify(data, var)
}

# separate training and test data
train_data_x <- data[train_index, c(6:ncol(data))]
test_data_x <- data[test_index, c(6:ncol(data))]

# Train random forest
set.seed(1)
# train_grid <- expand.grid(.mtry = seq(25, ncol(data_x), 25))
train_grid <- expand.grid(.mtry = 225)

train_params <- trainControl(method = "repeatedcv",
                             number = 50,
                             repeats = 5,
                             savePredictions = TRUE,
                             summaryFunction = expRMSE,
                             verboseIter = TRUE)
                       
train_model <- train(y = log(data_y), # log-transform response
                     x = train_data_x,
                     ntree=5000,
                     metric = "expRMSE",
                     maximize = FALSE,
                     tuneGrid = train_grid,
                     trControl = train_params)

# mtry = 225 is has lowest RMSE; RMSE ~ 1785000

predictions <- predict(train_model, newdata=test_data_x)

pred.df <- data.frame(Id = test_data$Id,
                      Prediction = exp(predictions))

write.table(pred.df, "tfi_submission.csv", sep=",", col.names=T, row.names=F)

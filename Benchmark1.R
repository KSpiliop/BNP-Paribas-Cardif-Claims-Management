###--- BNP-Paribas-Cardif-Claims-Management-Script ---###
###--- Environment Set-Up ---###

# Clear Workspace
rm(list=ls())

# Set Seed
set.seed(0602)

# Load Libraries
library(caret)
library(e1071)

###--- Load Data ---###

# Load Training Data
train = read.csv("C:/Users/Joe/Desktop/Data/train.csv")

# Load Test Data
test = read.csv("C:/Users/Joe/Desktop/Data/test.csv")

# Load Submission Sample
sample_submission <- read.csv("C:/Users/Joe/Desktop/Data/sample_submission.csv")

# Data Set stats
train_rows = nrow(train)
train_cols = ncol(train)
test_rows = nrow(test)
test_cols = ncol(test)

###--- Pre-processing ---###
# Convert 'target' to factor and put into labels vector
labels = as.factor(train$target)

# Delete 'target' from dataset
train = train[, -2]

# Calculate vector counting the number of missing values in each row
# train$missingVal = rowSums(is.na(train))
# test$missingVal = rowSums(is.na(test))

# Extract all numeric columns first
train.numeric = train[, sapply(train, is.numeric)]
test.numeric = test[, sapply(test, is.numeric)]

# Example of removing highly correlated variables
correl <- cor(train.numeric,use="pairwise.complete.obs")
correlated <- findCorrelation(correl,cutoff=0.9)
train.numeric <- train.numeric[,-correlated]
test.numeric <- test.numeric[,-correlated]

# Example of reducing dimensions
xTrans <- preProcess(train.numeric,method="pca",thres=0.90,verbose=TRUE)
train.numeric <- predict(xTrans, train.numeric)
test.numeric <- predict(xTrans, test.numeric)

###--- Model Training ---###

# Run NaiveBayes model 
# To use categorical variables: must find which factor levels are not in the training set
m = naiveBayes(train.numeric, labels, laplace = 0)
p = predict(m, test.numeric, type = "raw")

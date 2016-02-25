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

###--- Model Training ---###

# Run NaiveBayes model 
m = naiveBayes(train.numeric, labels, laplace = 0)
p = predict(m, test.numeric, type = "raw")

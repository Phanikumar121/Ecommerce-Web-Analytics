
#Setting working directory
setwd("C:\\Users\\PHANI KUMAR\\Desktop\\Machine Learning Projects\\5. ECOMMERCE CASE STUDY - CLASSIFICATION")
options(scipen = 999)

#Reading the files
mydata <- read.csv("train.csv")

#understanding my data
summary(mydata)
str(mydata)

#Various columns like region,source medium, country ,mobile cannot be used for analysis as they are hashed

####################################Data cleaning and pre engineering###################################

#User defined function to get descriptive statistics for continuous data
my_stats <- function(x){
  nmiss<- sum(x[x==-1])
  mean<- mean(x)
  max<- max(x)
  min<- min(x)
  p1<- quantile(x,0.01)
  p99<- quantile(x,0.99)
  outlier_cap<- max>p99|min<p1
  return(c(nmiss=nmiss,mean=mean,max=max,min=min,P1=p1,P99=p99,Cap=outlier_cap))
}

#User defined function to get descriptive statistics for categorical data

my_stats_categ <- function(x){
  nmiss=sum(x[x==-1])
  return(nmiss=nmiss)
}


#Splitting the main data into continuous and categorical data frames

continuous_vars <- c("metric1","metric2","metric6","metric3","metric4","metric5","page1_top",
            "page1_exits","page2_top","page2_exits","page3_top","page3_exits","page4_top",
            "page4_exits","page5_top","page5_exits","page6_top","page6_exits")

categorical_vars <- c("binary_var1","binary_var2","visited_page1","visited_page2","visited_page3",
            "visited_page4","visited_page5","visited_page6","target") 

#Applying udf to get descriptives of variables

descriptive_cont <- data.frame(sapply(mydata[continuous_vars],FUN = my_stats))

write.csv(descriptive_cont,"descriptive_stats.csv")

descriptive_categ <- data.frame(sapply(mydata[categorical_vars],FUN = my_stats_categ))

#In the continuous variable table, most of the variables related to page exits have large amount
#of missing values so it is better to remove them from the analysis

#There is no missing data in the categorical data
#The data does not have any outliers too

#########################################Building model#######################################

#splitingdata for building classification model 
set.seed(1137)
training_split <- sample(2,nrow(mydata),replace = TRUE,prob = c(0.7,0.3))

training<- mydata[training_split==1,]
testing<- mydata[training_split==2,]

#Checking if the data is split in equal proportions or not
prop.table(table(mydata$target))
prop.table(table(training$target))
prop.table(table(testing$target))

#Converting the categorical data into factors
training$visited_page1<- as.factor(training$visited_page1)
training$visited_page2<- as.factor(training$visited_page2)
training$visited_page3<- as.factor(training$visited_page3)
training$visited_page4<- as.factor(training$visited_page4)
training$visited_page5<- as.factor(training$visited_page5)
training$visited_page6<- as.factor(training$visited_page6)
training$binary_var1<- as.factor(training$binary_var1)
training$binary_var2<- as.factor(training$binary_var2)
training$target <- as.factor(training$target)

#Deleting the colums that have missing data and columns that are not required
remove <- c("unique_id","region","sourceMedium","device","country",
"dayHourMinute","page1_exits","page2_exits","page3_exits",
"page4_exits","page5_exits","page6_exits")

training[,remove] <- NULL
training <- training

#Model creation

#When i tried to create a model with the training data R threw me an error saying cant allocate
#this much size to a vector . So i though of to take less data from the training data

training<- training[1:20000, ]

library(caret)
library(randomForest)

names(training)
set.seed(333)
model1 <- randomForest(target ~ ., data = training)
print(model1)

#Now I fine tuned my randomforest model to see if i can reduce the out of bag error 
tuneRF(training[ ,-21],training[ ,21], stepFactor = 0.5,plot = TRUE, ntreeTry = 300, trace = TRUE,improve = 0.05)

model2<- randomForest(target ~ ., data = training, ntree=300,mtry=4,
                    Importance=TRUE,Proximity=TRUE)
print(model2)

#Lets predict the above two models on testing data
testing[,remove] <- NULL
testing <- testing

testing$visited_page1<- as.factor(testing$visited_page1)
testing$visited_page2<- as.factor(testing$visited_page2)
testing$visited_page3<- as.factor(testing$visited_page3)
testing$visited_page4<- as.factor(testing$visited_page4)
testing$visited_page5<- as.factor(testing$visited_page5)
testing$visited_page6<- as.factor(testing$visited_page6)
testing$binary_var1<- as.factor(testing$binary_var1)
testing$binary_var2<- as.factor(testing$binary_var2)
testing$target <- as.factor(testing$target)

#for model1
pred_test <- predict(model1,newdata = testing)
confusionMatrix(pred_test,testing$target,positive = "1")

#for model2
pred_test2 <- predict(model2,newdata = testing)
confusionMatrix(pred_test2,testing$target,positive = "1")

#There is negligible difference of accuracy between the two models but the data has very less no of "1"

#From the confusion matrix we can see that most of the data contains target value as "0".
#less than 10% of data is "1". The business problem given is to predict the number of "1" for unique ID.
#We need to focus mostly on predicting the no of "1" for unique ID
#But as our data has very less number of our predictor values , we need to go ahead with 
#Over sampling
#Over and Under sampling 
#Under sampling

###############################################Sampling#############################

library(ROSE)
library(e1071)

#Under sampling
under_sampling <- ovun.sample(target~.,data = training,method = "under",seed = 456,N = 5000)$data

set.seed(184)
randomforest_under <- randomForest(target~.,data = under_sampling,Importance=TRUE,Proximity=TRUE)

#Over sampling

over_sampling <- ovun.sample(target~.,data = training,method = "over",seed = 299,N = 35000)$data

set.seed(233)
randomforest_over <- randomForest(target~.,data = over_sampling,Importance=TRUE,Proximity=TRUE)

#Over and Under sampling

over_under_sampling <- ovun.sample(target~.,data = training,method = "both",seed = 677,N = 20000)$data

set.seed(966)
randonforest_over_under_sampling <- randomForest(target~.,data = over_under_sampling,Importance=TRUE,
                                                 Proximity=TRUE)
#Confusion matrix for the 3 different samples
confusionMatrix((predict(randomforest_under,newdata = testing)),testing$target,positive="1")
confusionMatrix((predict(randomforest_over,newdata = testing)),testing$target,positive="1")
confusionMatrix((predict(randonforest_over_under_sampling,newdata = testing)),testing$target,positive="1")

#So on the basis of the above confusion metrics, under sampled data is producing better result over all
#where sensitivity is 0.75742,specificity is 0.83752 and accurary is 0.8302
#This model predicted the no of "1" more correctly when compared to the other models

##############################Using the model for testing#########################

rf_pred_under <- data.frame(predict(randomforest_under,newdata = testing,type = "prob"))
rf_pred_under <- rf_pred_under[ ,2]
print(head(rf_pred_under))

library(ROCR)
rf_pred_under <- prediction(rf_pred_under,testing$target)

ROC <- performance(rf_pred_under,"tpr","fpr")
plot(ROC,xlab = "1-specificity",ylab = "sensitivity",main = "ROC curve")

#Calculating the AUC score
rf_AUC_under <- performance(rf_pred_under,"auc")

#Getting the AUC value
rf_AUC_under <- unlist(slot(rf_AUC_under,"y.values"))
rf_AUC_under <- round(rf_AUC_under,3)
legend(title = "AUC","bottomright",0.4,rf_AUC_under) 

#calculating Accuracy,F1score,Precision,Recall
rf_pred2_under <- predict(randomforest_under,newdata = testing)
confusionMatrix(rf_pred2_under,testing$target,positive = "1")

diag_calc <-table(rf_pred2_under,testing$target)

accuracy_pred_rf <- sum(diag(diag_calc))/sum(diag_calc)
misclass <- 1-accuracy_pred_rf

#F1 score

precision_rate <- posPredValue(rf_pred2_under,testing$target,positive = "1")
Recall <- sensitivity(rf_pred2_under,testing$target,positive = "1")

F1_score <- (2*precision_rate*Recall)/(precision_rate+Recall)

################################Predicting on the new testing data provided#######################

newdata <- read.csv("test.csv")

#Converting categorical variables into factors

newdata$visited_page1<- as.factor(newdata$visited_page1)
newdata$visited_page2<- as.factor(newdata$visited_page2)
newdata$visited_page3<- as.factor(newdata$visited_page3)
newdata$visited_page4<- as.factor(newdata$visited_page4)
newdata$visited_page5<- as.factor(newdata$visited_page5)
newdata$visited_page6<- as.factor(newdata$visited_page6)
newdata$binary_var1<- as.factor(newdata$binary_var1)
newdata$binary_var2<- as.factor(newdata$binary_var2)

#Predicting on the new data
testing_predictions <- predict(randomforest_under,newdata)

#Merging the data set
final_output_bind <- cbind(newdata,testing_predictions)
table(final_output_bind$testing_predictions)

library(dplyr)
final_testing_output_predictions <- select(final_output_bind,c(unique_id,testing_predictions))

write.csv(final_testing_output_predictions,"testingdata_output_predictions.csv")

############################End of classification######################################

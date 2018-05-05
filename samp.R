#rm enviroinment
rm(list = ls(all = T))

#setting working directory
getwd()
setwd("/Users/piyushmishra/Downloads/intern")

#loading data set
insurance= read.csv("train.csv", header = T, sep = ",")
str(insurance)
str(insurance, list.len = 128) 
insurance$Product_Info_2<-as.factor(insurance$Product_Info_2)
table(insurance$Product_Info_2)
##check the missing values
sum(is.na(insurance))
colSums(is.na(insurance))


#1. Pre-processing and cleaning the data
#REmoving columns which are having more than 60% data value as null

data<-insurance[, -which(colMeans(is.na(insurance)) > .5)]


#Verifying the columns which are removed
removedcoumns<-insurance[,which(colMeans(is.na(insurance)) > 0.5)]

#Family_Hist_3 Family_Hist_5 Medical_History_10 Medical_History_15 Medical_History_24 Medical_History_32 



#Setting missing values to mean value 
#Finding columns having missing values
missingvalues <- c(unlist(lapply(data, function(x) any(is.na(x)))))

#Finding columns having missing values
missingvalues

#Employment_Info_1   Employment_Info_4   Employment_Info_6 Insurance_History_5       Family_Hist_2 
#TRUE                TRUE                TRUE                TRUE                TRUE 
#Family_Hist_4   Medical_History_1 
#TRUE                TRUE             

library(DMwR)
data1 = knnImputation(data, k = 3)

data[(is.na(data))] <- 0
sum(is.na(data))
library(dummies)
data.new <- dummy.data.frame(data, sep = ".")
names(data.new)


str(data)

library(ROSE)
table(data.new$Response)
prop.table(table(insurance$Response))
library(caret)
x=cor(data.new[,!(names(data.new) %in% "Response")])
findCorrelation(x,names=T,cutoff = 0.80)
data1=data.new[,!(names(data.new)%in% c("Insurance_History_9","Medical_History_26", "Medical_History_36","Medical_Keyword_23", "Medical_Keyword_48" ,"Wt","Insurance_History_3","Insurance_History_4","Medical_History_37","Medical_History_23","Medical_History_19"))]

data1$Id<-NULL
summary(data1)

library(caret)
set.seed(123)
trainrows <- createDataPartition(y = data1$Response, p = 0.80, list = F)
train <- data1[trainrows,]
test <- data1[-trainrows,]

train$Response<-as.factor(train$Response)
prop.table(table(train$Response))

library(DMwR)
data.smote<-SMOTE(Response~., data = train,perc.over = 500,perc.under = 1500)
table(data.smote$Response)

prop.table(table(data.smote$Response))
data.smote$Response<-as.factor(data.smote$Response)
test$Response<-as.factor(test$Response)
##Decision Tree
library(C50)
c5_tree <- C5.0(Response ~ . , data.smote)
pred4<-predict(c5_tree, newdata = test)
confusionMatrix(test$Response, pred4)
c5_rules <- C5.0(Response ~ . , data.smote, rules = T)
summary(c5_rules)
##Random Forest
library(randomForest)

fit<- randomForest(Response~.,data=train)

summary(fit)
importance(fit)
pred<-predict(fit, newdata = test)
test_prediction1 <- matrix(pred)
pred1<-predict(fit, newdata = test1)

confusionMatrix(test$Response, pred)


##Upsampling 
samp<-upSample(x = train[,-128], y = train$Response)
table(samp$Class)

fit1<- randomForest(Class~.,data=samp)

pred1<-predict(fit1, newdata = test)
confusionMatrix(test$Response, pred1)

##Downsampling 
samp1<-downSample(x = train[,-128], y = train$Response)
table(samp1$Class)
fit2<- randomForest(Class~.,data=samp1)

pred2<-predict(fit2, newdata = test)
confusionMatrix(test$Response, pred2)

#Multinom MODEL BULDING
library(nnet)

multi_model <- multinom(Response~., data = train,maxit=1000)


summary(multi_model)
head(fitted(multi_model))

#variable imporance
library(caret)
mostImportantVariables <- varImp(multi_model)
mostImportantVariables$Variables <- row.names(mostImportantVariables)
mostImportantVariables <- mostImportantVariables[order(-mostImportantVariables$Overall),]
print(head(mostImportantVariables))


pred_model1 <- predict(multi_model, newdata = test1, type = "probs")
head(pred_model1)
test1$Product_Info_2<-as.numeric(test1$Product_Info_2)
pred_model2 <- predict(multi_model, type="class", newdata=test1)
head(pred_model2)

head(table(pred_model2))

confusionMatrix(test$Response, pred_model2)  

##KNN
library(MASS)
knn <- knn3(Response ~ ., data = train ,k=5)
summary(knn)
#Predict Output 
predicted= predict(knn,test,type = "class")
confusionMatrix(test$Response, predicted) 

##SVM
library(e1071)
train$Response<-as.numeric(train$Response)
model <-svm(Response ~ ., data = train)
pred_prob <- predict(fit, test, decision.values = TRUE, probability = TRUE)

##Naive Bayes

library(e1071)
data.classifier <- naiveBayes(data.smote[,-128],data.smote$Response)
data.predict <- predict(data.classifier, test)
confusionMatrix(test$Response, data.predict) 
##XGBoost
library(caret)
std_data=preProcess(train[,!(names(train)%in% "Response")],method = c("center","scale"))
train_data=predict(std_data,train)
test_data=predict(std_data,test)

library(xgboost)
train_xgb=xgb.DMatrix(data=as.matrix(train_data[,!(names(train_data) %in% "Response")]),
                      label=as.matrix(train_data[,names(train_data)%in%"Response"]))

test_xgb=xgb.DMatrix(data=as.matrix(test_data[,!(names(test_data) %in% "Response")]),
                     label=as.matrix(test_data[,names(test_data) %in% "Response"]))

set.seed(123)
params_list=list("objective"="multi:softmax", "num_class" = 9,"eta"=0.1,"max_depth" = 8,"gamma" = 0,"colsample_bytree" = 0.5,"eval_metric" = "mlogloss","subsample" = 1.0,"silent" = 1)

xgb_model_params=xgb.cv(data =train_xgb,params = params_list,nrounds = 1000,early_stopping_rounds = 50,nfold = 5 ,maximize = T) 



nround = 51
md <- xgb.train(data=train_xgb, params=params_list, nrounds=nround, nthread=6)

xgb_params_pred=predict(md,test_xgb)
xgb_params_pred
confusionMatrix(xgb_params_pred,test_data$Response)

test1=predict(std_data,test1)
test1=xgb.DMatrix(data=as.matrix(test1))
pred=predict(md,test1)
##insurance$BMI<-NULL
##insurance$Employment_Info_4<-NULL
##insurance$Employment_Info_6<-NULL
#insurance$Insurance_History_5<-NULL
#insurance$Family_Hist_2<-NULL
#insurance$Family_Hist_3<-NULL
#insurance$Family_Hist_4<-NULL
#insurance$Family_Hist_5<-NULL
insurance$Medical_History_10<-NULL
insurance$Medical_History_15<-NULL
insurance$Medical_History_24<-NULL
insurance$Medical_History_32<-NULL
##insurance$Medical_History_1<-NULL
library(imputeMissings)
##insurance<-impute(insurance,object = NULL,method = "median/mode",flag = FALSE)
insurance[(is.na(insurance))] <- 0
sum(is.na(insurance))

insurance$Product_Info_2 <- as.factor(insurance$Product_Info_2)
insurance$Product_Info_2 <- as.numeric(insurance$Product_Info_2)
insurance$Response <- as.factor(insurance$Response)
str(insurance)

test1 <- read.csv("test.csv", header = T, sep = ",")

sum(is.na(test1))
colSums(is.na(test1))
##test1$BMI<-NULL
##test1$Employment_Info_4<-NULL
##test1$Employment_Info_6<-NULL
#test1$Insurance_History_5<-NULL
#test1$Family_Hist_2<-NULL
#test1$Family_Hist_3<-NULL
#test1$Family_Hist_4<-NULL
#test1$Family_Hist_5<-NULL
test1$Medical_History_10<-NULL
test1$Medical_History_15<-NULL
test1$Medical_History_24<-NULL
test1$Medical_History_32<-NULL
##test1$Medical_History_1<-NULL
library(imputeMissings)
test1<-impute(test1,object = NULL,method = "median/mode",flag = FALSE)
test1[(is.na(test1))] <- 0
sum(is.na(test1))
test1$Product_Info_2<-as.factor(test1$Product_Info_2)
test1$Product_Info_2<-as.numeric(test1$Product_Info_2)

library(caret)
y=cor(test1[,!(names(test1) %in% "Response")])
findCorrelation(x,names=T,cutoff = 0.80)
test1=test1[,!(names(test1)%in% c("Insurance_History_9","Medical_History_26", "Medical_History_36","Medical_Keyword_23", "Medical_Keyword_48" ,"Wt","Insurance_History_3","Insurance_History_4","Medical_History_37","Medical_History_23","Medical_History_19"))]


par(mfrow=c(1, 2))
plot(insurance, main="With Outliers", xlab="speed", ylab="dist", pch="*", col="red", cex=2)
#install packages
##install.packages("ROSE")
library(ROSE)

#check table
table(insurance$Response)

str(insurance$Response)
#check classes distribution
prop.table(table(insurance$Response))

library(caret)
set.seed(123)
trainrows <- createDataPartition(y = insurance$Response, p = 0.80, list = F)
train <- insurance[trainrows,]
test <- insurance[-trainrows,]

##library(caret)
##std_data=preProcess(train[,!(names(train)%in% "Response")],method = c("center","scale"))
##train_data=predict(std_data,train)
##test_data=predict(std_data,test)
##library(rpart)
##treeimb <- rpart(Response ~ ., data = train)
##pred.treeimb <- predict(treeimb, newdata = test,type="prob")
##numberOfClasses <- length(unique(insurance$Response))
##library(dplyr)
##test_prediction <- matrix(pred.treeimb, nrow = numberOfClasses,
    ##                      ncol=length(pred.treeimb)/numberOfClasses) %>%
  ##t() %>%
  ##data.frame() %>%
  ##mutate(
    ##     max_prob = max.col(., "last"))

##confusionMatrix(factor(test$Response),
  ##              factor(test_prediction$max_prob),
    ##            mode = "everything")

##table(train$Response)
#over sampling
##data_balanced_over <- ovun.sample(Response ~ ., data = train, method = "over",N = 109144)$data
##table(data_balanced_over$Response)
##0    1 
##980 980

##data_balanced_under <- ovun.sample(cls ~ ., data = train, method = "under", N = 5680, seed = 1)$data
##table(data_balanced_under$cls)
##0    1 
#20  20

##data.rose <- ROSE(Response ~ ., data = train, seed = 2)$data
##table(data.rose$cls)
#check table
table(train$Response)


#check classes distribution
prop.table(table(train$Response))
train$Response<-as.factor(train$Response)
library(caret)
samp<-upSample(x = train[,-113], y = train$Response)
table(samp$Class)





library(DMwR)
data.smote<-SMOTE(Response~., data = insurance)
table(data.smote$Response)

#check classes distribution
prop.table(table(data.smote2$Response))

data.smote1<-SMOTE(Response~., data = insurance,perc.over = 1000)
table(data.smote1$Response)



data.smote2<-SMOTE(Response~., data = data.smote1,perc.under = 500,perc.over = 800)
table(data.smote2$Response)



library(randomForest)

fit<- randomForest(Class~.,data=samp)

summary(fit)
importance(fit)
pred<-predict(fit, newdata = test)
test_prediction1 <- matrix(pred)
pred1<-predict(fit, newdata = test1)

confusionMatrix(test$Response, pred) 

## create submission file
testId = test1$Id
submission = data.frame(Id = testId)
submission$Response = as.integer(pred1)
write.csv(test_prediction1, "Submission2.csv", row.names = FALSE)

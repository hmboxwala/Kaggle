otto <- read.csv("train.csv")
otto <- otto[sample.int(nrow(otto)),]
otto <- otto[,-1]
test <- read.csv("test.csv")
otto[,-94] <- 1/(1+otto[,-94])
test[,-1] <- 1/(1+test[,-1])
merged <- rbind(otto[,-94],test[,-1])

#library(fastcluster)
#mergedClust <- hclust.vector(merged,method="ward",metric="euclidean")
#mergedNormClust <- hclust.vector(mergedNorm,method="ward",metric="euclidean")
#plot(mergedClust)
#clusterGroups = cutree(mergedClust, k = 9)
#table(otto$target,clusterGroups[1:61878])
#length(unique(clusterGroups[1:61878]))
#otto$clust <- clusterGroups[1:61878]
#test$clust <- clusterGroups[61879:206246]


set.seed(20)
KMC18 = kmeans(merged, centers = 18,iter.max=1000)
KMC27 = kmeans(merged, centers = 27,iter.max=1000)
KMC54 = kmeans(merged, centers = 54,iter.max=1000)
KMC93 = kmeans(merged, centers = 93,iter.max=1000)
KMC186 = kmeans(merged, centers = 186,iter.max=1000)

table(otto$target,KMC18$cluster[1:61878])
otto$clust <- KMC18$cluster[1:61878]
otto$clust1 <- KMC27$cluster[1:61878]
otto$clust2 <- KMC54$cluster[1:61878]
otto$clust3 <- KMC93$cluster[1:61878]
otto$clust4 <- KMC186$cluster[1:61878]


test$clust <- KMC18$cluster[61879:206246]
test$clust1 <- KMC27$cluster[61879:206246]
test$clust2 <- KMC54$cluster[61879:206246]
test$clust3 <- KMC93$cluster[61879:206246]
test$clust4 <- KMC186$cluster[61879:206246]



library(caTools)
split <- sample.split(otto$target, 0.8)
train <- subset(otto,split==TRUE)
crossval <- subset(otto, split==FALSE)
train1 <- as.matrix(train[,-94])
train1 <-matrix(as.numeric(train1),nrow(train1),ncol(train1))
crossval1 <- as.matrix(crossval[,-94])
crossval1 <-matrix(as.numeric(crossval1),nrow(crossval1),ncol(crossval1))
test1 <- as.matrix(test[,-1])
test1 <-matrix(as.numeric(test1),nrow(test1),ncol(test1))


require(xgboost)
require(methods)

y = train[,94]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8)
bst.cv = xgb.cv(param=param, data = train1, label = y, 
                nfold = 10, nrounds=210)
bst = xgboost(param=param, data = train1, label = y, nrounds=214)

pred = predict(bst,crossval1)
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -sum(actual*log(predicted))/nrow(actual)
}

library(caret)
dummy.formulaTrain <- dummyVars(~ target, data=train, levelsOnly=TRUE)
actualClassesTrain <- predict(dummy.formulaTrain,train)
print(LogLoss(actualClassesTrain, nnet_predictionsTrain))

dummy.formula <- dummyVars(~ target, data=crossval, levelsOnly=TRUE)
actualClasses <- predict(dummy.formula,crossval)
print(LogLoss(actualClasses, pred))

# Using full training data
train <- otto
train1 <- as.matrix(train[,-94])
train1 <-matrix(as.numeric(train1),nrow(train1),ncol(train1))
y = train[,94]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

# Predict on test
predTest = predict(bst,test1)
predTest = matrix(predTest,9,length(predTest)/9)
predTest = t(predTest)
predTest = format(predTest, digits=2,scientific=F) # shrink the size of submission
predTest = data.frame(1:nrow(predTest),predTest)
names(predTest) = c('id', paste0('Class_',1:9))
write.csv(predTest,file='submission.csv', quote=FALSE,row.names=FALSE)

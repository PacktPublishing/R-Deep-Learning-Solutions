library("tensorflow") 
np <- import("numpy") 

xFeatures = c("Temperature", "Humidity", "Light", "CO2", "HumidityRatio")
yFeatures = "Occupancy"
occupancy_train <-read.csv("datatraining.txt",stringsAsFactors = T)
occupancy_test <- read.csv("datatest.txt",stringsAsFactors = T)

occupancy_train<-apply(occupancy_train[, c(xFeatures, yFeatures)], 2, FUN=as.numeric) 
occupancy_test<-apply(occupancy_test[, c(xFeatures, yFeatures)], 2, FUN=as.numeric)

nFeatures<-length(xFeatures)
nRow<-nrow(occupancy_train)

tf$reset_default_graph()
sess<-tf$InteractiveSession()

x <- tf$constant(unlist(occupancy_train[, xFeatures]), shape=c(nRow, nFeatures), dtype=np$float32) 
W <- tf$Variable(tf$random_uniform(shape(nFeatures, 1L)))
b <- tf$Variable(tf$zeros(shape(1L)))
y <- tf$matmul(x, W) + b

y_ <- tf$constant(unlist(occupancy_train[, yFeatures]), dtype="float32", shape=c(nRow, 1L))
cross_entropy<-tf$reduce_mean(tf$nn$sigmoid_cross_entropy_with_logits(labels=y_, logits=y, name="cross_entropy"))
optimizer <- tf$train$GradientDescentOptimizer(0.15)$minimize(cross_entropy)

init <- tf$global_variables_initializer()
sess$run(init)
 
for (step in 1:5000) {
  sess$run(optimizer)
  if (step %% 20== 0)
    cat(step, "-", sess$run(W), sess$run(b), "==>", sess$run(cross_entropy), "\n")
}

library(pROC) 
ypred <- sess$run(tf$nn$sigmoid(tf$matmul(x, W) + b))
roc_obj <- roc(occupancy_train[, yFeatures], as.numeric(ypred))

nRowt<-nrow(occupancy_test)
xt <- tf$constant(unlist(occupancy_test[, xFeatures]), shape=c(nRowt, nFeatures), dtype=np$float32) 
ypredt <- sess$run(tf$nn$sigmoid(tf$matmul(xt, W) + b))
roc_objt <- roc(occupancy_test[, yFeatures], as.numeric(ypredt))

plot.roc(roc_obj, col = "green", lty=2, lwd=2)
plot.roc(roc_objt, add=T, col="red", lty=4, lwd=2)

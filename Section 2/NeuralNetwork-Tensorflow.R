library(tensorflow)

np <- import("numpy")
tf <- import("tensorflow")

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

n_hidden_1 = 5L 
n_hidden_2 = 5L 
n_input = 5L    
n_classes = 1L  
learning_rate = 0.001
training_epochs = 10000

x = tf$constant(unlist(occupancy_train[,xFeatures]), shape=c(nRow, n_input), dtype=np$float32)
y = tf$constant(unlist(occupancy_train[,yFeatures]), dtype="float32", shape=c(nRow, 1L))

multilayer_perceptron <- function(x, weights, biases){
  layer_1 = tf$add(tf$matmul(x, weights[["h1"]]), biases[["b1"]])
  layer_1 = tf$nn$relu(layer_1)
  layer_2 = tf$add(tf$matmul(layer_1, weights[["h2"]]), biases[["b2"]])
  layer_2 = tf$nn$relu(layer_2)
  out_layer = tf$matmul(layer_2, weights[["out"]]) + biases[["out"]]
  return(out_layer)
}

weights = list(
  "h1" = tf$Variable(tf$random_normal(c(n_input, n_hidden_1))),
  "h2" = tf$Variable(tf$random_normal(c(n_hidden_1, n_hidden_2))),
  "out" = tf$Variable(tf$random_normal(c(n_hidden_2, n_classes)))
)
biases = list(
  "b1" =  tf$Variable(tf$random_normal(c(1L,n_hidden_1))),
  "b2" = tf$Variable(tf$random_normal(c(1L,n_hidden_2))),
  "out" = tf$Variable(tf$random_normal(c(1L,n_classes)))
)

pred = multilayer_perceptron(x, weights, biases)

cost = tf$reduce_mean(tf$nn$sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf$train$AdamOptimizer(learning_rate=learning_rate)$minimize(cost)

init = tf$global_variables_initializer()
sess$run(init)

for(epoch in 1:training_epochs){
  sess$run(optimizer)
  if (epoch %% 20== 0)
    cat(epoch, "-", sess$run(cost), "\n")                                   
}

library(pROC) 
ypred <- sess$run(tf$nn$sigmoid(multilayer_perceptron(x, weights, biases)))
roc_obj <- roc(occupancy_train[, yFeatures], as.numeric(ypred))

nRowt<-nrow(occupancy_test)
xt <- tf$constant(unlist(occupancy_test[, xFeatures]), shape=c(nRowt, nFeatures), dtype=np$float32)
ypredt <- sess$run(tf$nn$sigmoid(multilayer_perceptron(xt, weights, biases)))
roc_objt <- roc(occupancy_test[, yFeatures], as.numeric(ypredt))

plot.roc(roc_obj, col = "green", lty=2, lwd=2)
plot.roc(roc_objt, add=T, col="red", lty=4, lwd=2)

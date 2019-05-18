
library(tensorflow)
require(imager)
require(caret) 

datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

plot_mnist<-function(imageD, pixel.y=16){
  require(imager)
  actImage<-matrix(imageD, ncol=pixel.y, byrow=FALSE)
  img.col.mat <- imappend(list(as.cimg(actImage)), "c")
  plot(img.col.mat, axes=F)
}

reduceImage<-function(actds, n.pixel.x=16, n.pixel.y=16){
  actImage<-matrix(actds, ncol=28, byrow=FALSE)
  img.col.mat <- imappend(list(as.cimg(actImage)),"c")
  thmb <- resize(img.col.mat, n.pixel.x, n.pixel.y)
  outputImage<-matrix(thmb[,,1,1], nrow = 1, byrow = F)
  return(outputImage)
}


trainData<-t(apply(mnist$train$images, 1, FUN=reduceImage))
validData<-t(apply(mnist$test$images, 1, FUN=reduceImage))
labels <- mnist$train$labels
labels_valid <- mnist$test$labels
rm(mnist)

tf$reset_default_graph()
sess<-tf$InteractiveSession()

n_input<-16
step_size<-16
n.hidden<-64
n.class<-10
num_layers <- 3

lr<-0.01
batch<-500
iteration = 100

lstm<-function(x, weight, bias){
  x = tf$unstack(x, step_size, 1)
  lstm_cell = tf$contrib$rnn$BasicLSTMCell(n.hidden, forget_bias=1.0, state_is_tuple=TRUE)
  cell_output = tf$contrib$rnn$static_rnn(lstm_cell, x, dtype=tf$float32)
  last_vec=tail(cell_output[[1]], n=1)[[1]]
  return(tf$matmul(last_vec, weights) + bias)
} 
eval_acc<-function(yhat, y){
  correct_Count = tf$equal(tf$argmax(yhat,1L), tf$argmax(y,1L))
  mean_accuracy = tf$reduce_mean(tf$cast(correct_Count, tf$float32))
  
  return(mean_accuracy)
}

with(tf$name_scope('input'), {
  x = tf$placeholder(tf$float32, shape=shape(NULL, step_size, n_input), name='x')
  y <- tf$placeholder(tf$float32, shape(NULL, n.class), name='y')
  weights <- tf$Variable(tf$random_normal(shape(n.hidden, n.class)))
  bias <- tf$Variable(tf$random_normal(shape(n.class)))
})
yhat = lstm(x, weights, bias)
cost = tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(logits=yhat, labels=y))
optimizer = tf$train$AdamOptimizer(learning_rate=lr)$minimize(cost) 
sess$run(tf$global_variables_initializer()) 
for(i in 1:iteration){
  spls <- sample(1:dim(trainData)[1],batch)
  sample_data<-trainData[spls,]
  sample_y<-labels[spls,]  
  sample_data=tf$reshape(sample_data, shape(batch, step_size, n_input))
  out<-optimizer$run(feed_dict = dict(x=sample_data$eval(), y=sample_y))     
  if (i %% 1 == 0){
    cat("iteration - ", i, "Training Loss - ",  cost$eval(feed_dict = dict(x=sample_data$eval(), y=sample_y)), "\n")
  }
}

accuracy<-eval_acc(yhat, y)
valid_data=tf$reshape(validData, shape(-1, step_size, n_input))
yhat<-sess$run(tf$argmax(yhat, 1L), feed_dict = dict(x = valid_data$eval()))
image(t(matrix(validData[20,], ncol = 16, nrow = 16, byrow = T)), col  = gray((0:32)/32))
image(t(matrix(trainData[20,], ncol = 16, nrow = 16, byrow = T)), col  = gray((0:32)/32)) 
cost$eval(feed_dict=dict(x=valid_data$eval(), y=labels_valid))
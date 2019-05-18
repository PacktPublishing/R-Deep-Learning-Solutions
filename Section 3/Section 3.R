library(tensorflow)
library(imager)
np <- import("numpy")
tf$reset_default_graph()
sess <- tf$InteractiveSession()

labels <- read.table("batches.meta.txt")
num.images = 10000 
read.cifar.data <- function(filenames,num.images=10000){
  images.rgb <- list()
  images.lab <- list()
  for (f in 1:length(filenames)) {
    to.read <- file(paste("Cifar_10/",filenames[f], sep=""), "rb")
    for(i in 1:num.images) {
      l <- readBin(to.read, integer(), size=1, n=1, endian="big")
      r <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
      g <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
      b <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
      index <- num.images * (f-1) + i
      images.rgb[[index]] = data.frame(r, g, b)
      images.lab[[index]] = l+1
    }
    close(to.read)
    cat("completed :",  filenames[f], "\n")
    remove(l,r,g,b,f,i,index, to.read)
  }
  return(list("images.rgb"=images.rgb,"images.lab"=images.lab))
}
cifar_train <- read.cifar.data(filenames = c("data_batch_1.bin","data_batch_2.bin","data_batch_3.bin","data_batch_4.bin","data_batch_5.bin"))  
images.rgb.train <- cifar_train$images.rgb
images.lab.train <- cifar_train$images.lab
rm(cifar_train)
cifar_test <- read.cifar.data(filenames = c("test_batch.bin"))
images.rgb.test <- cifar_test$images.rgb
images.lab.test <- cifar_test$images.lab
rm(cifar_test)

flat_data <- function(x_listdata,y_listdata){
  x_listdata <- lapply(x_listdata,function(x){unlist(x)})
  x_listdata <- do.call(rbind,x_listdata)
  y_listdata <- lapply(y_listdata,function(x){a=c(rep(0,10)); a[x]=1; return(a)})
  y_listdata <- do.call(rbind,y_listdata)
  return(list("images"=x_listdata, "labels"=y_listdata))
}

train_data <- flat_data(x_listdata = images.rgb.train, y_listdata = images.lab.train)
test_data <- flat_data(x_listdata = images.rgb.test, y_listdata = images.lab.test)

drawImage <- function(index, images.rgb, images.lab=NULL) {
  require(imager)
  img <- images.rgb[[index]]
  img.r.mat <- as.cimg(matrix(img$r, ncol=32, byrow = FALSE))
  img.g.mat <- as.cimg(matrix(img$g, ncol=32, byrow = FALSE))
  img.b.mat <- as.cimg(matrix(img$b, ncol=32, byrow = FALSE))
  img.col.mat <- imappend(list(img.r.mat,img.g.mat,img.b.mat),"c") 
  if(!is.null(images.lab)){
    lab = labels[[1]][images.lab[[index]]] 
  }
  plot(img.col.mat,main=paste0(lab,":32x32 size",sep=" "),xaxt="n")
  axis(side=1, xaxp=c(10, 50, 4), las=1)
  return(list("Image label" =lab,"Image description" =img.col.mat))
}
drawImage(sample(1:(num.images), size=1), images.rgb.train, images.lab.train)

require(caret)
normalizeObj<-preProcess(train_data$images, method="range")
train_data$images<-predict(normalizeObj, train_data$images)
test_data$images <- predict(normalizeObj, test_data$images)


norm_data <- apply(train_data$images,1,function(x){
  return(data.frame(r=x[1:1024],
                    g=x[1025:2048],
                    b=x[2049:3072]))
})

drawImage(124, norm_data, images.lab.train)


filter_size1 = 5L          
num_filters1 = 64L         


filter_size2 = 5L         
num_filters2 = 64L         

fc_size = 1024L             

img_size = 32L

num_channels = 3L

img_size_flat = img_size * img_size * num_channels

img_shape = c(img_size, img_size)

num_classes = 10L

weight_variable <- function(shape) {
  initial <- tf$truncated_normal(shape, stddev=0.1)
  tf$Variable(initial)
}

bias_variable <- function(shape) {
  initial <- tf$constant(0.1, shape=shape)
  tf$Variable(initial)
}

create_conv_layer <- function(input,             
                              num_input_channels, 
                              filter_size,        
                              num_filters,        
                              use_pooling=True)   
{
  shape1 = shape(filter_size, filter_size, num_input_channels, num_filters)
  
  weights = weight_variable(shape=shape1)
  biases = bias_variable(shape=shape(num_filters))
  
  layer = tf$nn$conv2d(input=input,
                       filter=weights,
                       strides=shape(1L, 1L, 1L ,1L),
                       padding="SAME")
  
  layer = layer + biases
  
  if(use_pooling){
    layer = tf$nn$max_pool(value=layer,
                           ksize=shape(1L, 2L, 2L, 1L),
                           strides=shape(1L, 2L, 2L, 1L), 
                           padding='SAME')
  }
  
  layer = tf$nn$relu(layer)
  
  return(list("layer" = layer, "weights" = weights))
}

flatten_conv_layer <- function(layer){
  layer_shape = layer$get_shape()
  
  num_features = prod(c(layer_shape$as_list()[[2]],layer_shape$as_list()[[3]],layer_shape$as_list()[[4]]))
  
  layer_flat = tf$reshape(layer, shape(-1, num_features))
  
  return(list("layer_flat"=layer_flat, "num_features"=num_features))
}

create_fc_layer <- function(input,        
                            num_inputs,     
                            num_outputs,    
                            use_relu=True) 
{
  weights = weight_variable(shape=shape(num_inputs, num_outputs))
  biases = bias_variable(shape=shape(num_outputs))
  
  layer = tf$matmul(input, weights) + biases
  
  if(use_relu){
    layer = tf$nn$relu(layer)
  }
  
  return(layer)
}

x = tf$placeholder(tf$float32, shape=shape(NULL, img_size_flat), name='x')
x_image = tf$reshape(x, shape(-1L, img_size, img_size, num_channels))
y_true = tf$placeholder(tf$float32, shape=shape(NULL, num_classes), name='y_true')

conv1 <- create_conv_layer(input=x_image,
                           num_input_channels=num_channels,
                           filter_size=filter_size1,
                           num_filters=num_filters1,
                           use_pooling=TRUE)

layer_conv1 <- conv1$layer
weights_conv1  <- conv1$weights

conv2 <- create_conv_layer(input=layer_conv1,
                           num_input_channels=num_filters1,
                           filter_size=filter_size2,
                           num_filters=num_filters2,
                           use_pooling=TRUE)

layer_conv2 <- conv2$layer
weights_conv2 <- conv2$weights

flatten_lay <- flatten_conv_layer(layer_conv2)
layer_flat <- flatten_lay$layer_flat
num_features <- flatten_lay$num_features

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=num_features,
                            num_outputs=fc_size,
                            use_relu=TRUE)

keep_prob <- tf$placeholder(tf$float32)
layer_fc1_drop <- tf$nn$dropout(layer_fc1, keep_prob)

layer_fc2 = create_fc_layer(input=layer_fc1_drop,
                            num_inputs=fc_size,
                            num_outputs=num_classes,
                            use_relu=FALSE)

layer_fc2_drop <- tf$nn$dropout(layer_fc2, keep_prob)

y_pred = tf$nn$softmax(layer_fc2_drop)
y_pred_cls = tf$argmax(y_pred, dimension=1L)

cross_entropy = tf$nn$softmax_cross_entropy_with_logits(logits=layer_fc2_drop, labels=y_true)
cost = tf$reduce_mean(cross_entropy)

optimizer = tf$train$AdamOptimizer(learning_rate=1e-4)$minimize(cost)

y_true_cls = tf$argmax(y_true, dimension=1L)
correct_prediction = tf$equal(y_pred_cls, y_true_cls)
accuracy = tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

sess$run(tf$global_variables_initializer())
train_batch_size = 10000L
for (i in 1:100) {
  spls <- sample(1:dim(train_data$images)[1],train_batch_size)
  if (i %% 10 == 0) {
    train_accuracy <- accuracy$eval(feed_dict = dict(
      x = train_data$images[spls,], y_true = train_data$labels[spls,], keep_prob = 1.0))
    cat(sprintf("step %d, training accuracy %g\n", i, train_accuracy))
  }
  optimizer$run(feed_dict = dict(
    x = train_data$images[spls,], y_true = train_data$labels[spls,], keep_prob = 0.5))
}

test_accuracy <- accuracy$eval(feed_dict = dict(
  x = test_data$images, y_true = test_data$labels, keep_prob = 1.0))
cat(sprintf("test accuracy %g", test_accuracy))

test_pred_class <- y_pred_cls$eval(feed_dict = dict(
  x = test_data$images, y_true = test_data$labels, keep_prob = 1.0))
test_pred_class <- test_pred_class + 1
test_true_class <- c(unlist(images.lab.test))

table(actual = test_true_class, predicted = test_pred_class)

confusion <- as.data.frame(table(actual = test_true_class, predicted = test_pred_class))
plot <- ggplot(confusion)
plot + geom_tile(aes(x=actual, y=predicted, fill=Freq)) + scale_x_discrete(name="Actual Class") + scale_y_discrete(name="Predicted Class") + scale_fill_gradient(breaks=seq(from=-0, to=10, by=1)) + labs(fill="Normalized\nFrequency")
check.image <- function(images.rgb,index,true_lab, pred_lab) {
  require(imager)

  img <- images.rgb[[index]]
  img.r.mat <- as.cimg(matrix(img$r, ncol=32, byrow = FALSE))
  img.g.mat <- as.cimg(matrix(img$g, ncol=32, byrow = FALSE))
  img.b.mat <- as.cimg(matrix(img$b, ncol=32, byrow = FALSE))
  img.col.mat <- imappend(list(img.r.mat,img.g.mat,img.b.mat),"c") 

  plot(img.col.mat,main=paste0("True: ", true_lab,":: Pred: ", pred_lab),xaxt="n")
  axis(side=1, xaxp=c(10, 50, 4), las=1)
}

labels <- c("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")
plot.misclass.images <- function(images.rgb, y_actual, y_predicted,labels){
  
  indices <- which(!(y_actual == y_predicted))
  id <- sample(indices,1)
  
  true_lab <- labels[y_actual[id]]
  pred_lab <- labels[y_predicted[id]]
  check.image(images.rgb,index=id, true_lab=true_lab,pred_lab=pred_lab)
}
plot.misclass.images(images.rgb=images.rgb.test,y_actual=test_true_class,y_predicted=test_pred_class,labels=labels)



conv1_images <- conv1$layer$eval(feed_dict = dict(x = train_data$images[1:150,], 
                                                  y_true = train_data$labels[1:150,]))
weights_conv1 <- weights_conv1$eval(session=sess)

conv2_images <- conv2$layer$eval(feed_dict = dict(x = train_data$images[1:150,], y_true = train_data$labels[1:150,]))
weights_conv2 <- weights_conv2$eval(session=sess)

drawImage_conv <- function(index, images.bw, images.lab=NULL,par_imgs=8) {
  require(imager)

  img <- images.bw[index,,,]
  n_images <- dim(img)[3]
  par(mfrow=c(par_imgs,par_imgs), oma=c(0,0,0,0),
      mai=c(0.05,0.05,0.05,0.05),ann=FALSE,ask=FALSE)
  for(i in 1:n_images){
    img.bwmat <- as.cimg(img[,,i])

    if(!is.null(images.lab)){
      lab = labels[[1]][images.lab[[index]]] 
    }

    plot(img.bwmat,axes=FALSE,ann=FALSE)
  }
  par(mfrow=c(1,1))
}
drawImage_conv(5, images.bw = conv1_images, images.lab=images.lab.train)


drawImage_conv_weights <- function(weights_conv, par_imgs=8) {
  require(imager)
  n_images <- dim(weights_conv)[4]
  par(mfrow=c(par_imgs,par_imgs), oma=c(0,0,0,0),
      mai=c(0.05,0.05,0.05,0.05),ann=FALSE,ask=FALSE)
  for(i in 1:n_images){
    img.r.mat <- as.cimg(weights_conv[,,1,i])
    img.g.mat <- as.cimg(weights_conv[,,2,i])
    img.b.mat <- as.cimg(weights_conv[,,3,i])
    img.col.mat <- imappend(list(img.r.mat,img.g.mat,img.b.mat),"c") #Bind the three channels into one image
    
    plot(img.col.mat,axes=FALSE,ann=FALSE)
  }
  par(mfrow=c(1,1))
}
drawImage_conv_weights(weights_conv2)
library(tensorflow)
require(imager)
np <- import("numpy")

FUN_PATH<-"C:/Users/Test.PPMUMCPU0034/Desktop/R/Section 4/"
setwd(FUN_PATH)
source("download_cifar_data.R")
source("drawImage.R")
source("plotImage.R")
source("add_noise.R")
source("normalize.R")                                                      

DATA_PATH<-"C:/Users/Test.PPMUMCPU0034/Documents/Cifar_10/"

cifar_train <- read.cifar.data(filenames = c("data_batch_1.bin","data_batch_2.bin","data_batch_3.bin","data_batch_4.bin"))
images.rgb.train <- cifar_train$images.rgb
rm(cifar_train)

cifar_valid <- read.cifar.data(filenames = c("data_batch_5.bin"))
images.rgb.valid <- cifar_valid$images.rgb
rm(cifar_valid)

cifar_test <- read.cifar.data(filenames = c("test_batch.bin"))
images.rgb.test <- cifar_test$images.rgb
rm(cifar_test)

flat_data <- function(x_listdata,y_listdata){
  x_listdata <- lapply(x_listdata,function(x){unlist(x)})
  x_listdata <- do.call(rbind,x_listdata)
  y_listdata <- lapply(y_listdata,function(x){a=c(rep(0,10)); a[x]=1; return(a)})
  y_listdata <- do.call(rbind,y_listdata)
  return(list("images"=x_listdata, "labels"=y_listdata))
}

train_data <- flat_data(x_listdata = images.rgb.train, y_listdata = 1)
test_data <- flat_data(x_listdata = images.rgb.test, y_listdata = 1)
valid_data <- flat_data(x_listdata = images.rgb.valid, y_listdata = 1)

xcorr<-add_noise(train_data$images, frac=0.2, corr_type="masking")

plotImage(train_data$images[55,], xcorr[55, ])

tf$reset_default_graph()
sess<-tf$InteractiveSession()

img_size_flat=3072
x = tf$placeholder(tf$float32, shape=shape(NULL, img_size_flat), name='x')
x_corrput<-tf$placeholder(tf$float32, shape=shape(NULL, img_size_flat), name='x_corrput')

denoisingAutoencoder<-function(x, x_corrput, img_size_flat=3072, hidden_layer=c(1024), out_img_size=512){
  encoder = NULL
  n_input<-img_size_flat
  curentInput<-x_corrput
  layer<-c(hidden_layer, out_img_size)
  for(i in 1:length(layer)){
    n_output<-layer[i]
    W = tf$Variable(tf$random_uniform(shape(n_input, n_output), -1.0 / tf$sqrt(n_input), 1.0 / tf$sqrt(n_input)))
    b = tf$Variable(tf$zeros(shape(n_output)))
    encoder<-c(encoder, W)
    output = tf$nn$tanh(tf$matmul(curentInput, W) + b)
    curentInput = output
    n_input<-n_output
  }
  
  z = curentInput
  encoder<-rev(encoder)
  layer_rev<-c(rev(hidden_layer), img_size_flat)
  
  for(i in 1:length(layer_rev)){
    n_output<-layer_rev[i]
    W = tf$transpose(encoder[[i]])
    b = tf$Variable(tf$zeros(shape(n_output)))
    output = tf$nn$tanh(tf$matmul(curentInput, W) + b)
    curentInput = output
  }
  
  y = curentInput
  
  cost = tf$sqrt(tf$reduce_mean(tf$square(y - x)))
  return(list("x"=x, "z"=z, "y"=y, "x_corrput"=x_corrput, "cost"=cost))
}

dae_obj<-denoisingAutoencoder(x, x_corrput, img_size_flat=img_size_flat, hidden_layer=c(1024, 512), out_img_size=256)

learning_rate = 0.001
optimizer = tf$train$AdamOptimizer(learning_rate)$minimize(dae_obj$cost)


train=normalizeData(train_data$images, method="minmax")
valid=normalizeData(valid_data$images, method="minmax", obj = train)
test=normalizeData(test_data$images, method="minmax", obj = train)
x_corrput_ds<-add_noise(train$normalize_data, frac = 0.2, corr_type = "masking")
validCorrupt<-add_noise(valid$normalize_data, frac = 0.2, corr_type = "masking")
testCorrupt<-add_noise(test$normalize_data, frac = 0.2, corr_type = "masking")

iterationLog<-data.frame()
sess$run(tf$global_variables_initializer())
for(i in 1:1000){
  spls <- sample(1:dim(xcorr)[1],5000L)
  optimizer$run(feed_dict = dict(x=train$normalize_data[spls, ], x_corrput=x_corrput_ds[spls, ]))
  if (i %% 100 == 0) {
    trainingCost<-dae_obj$cost$eval((feed_dict = dict(x=train$normalize_data[spls, ], x_corrput=x_corrput_ds[spls, ])))
    validCost<-dae_obj$cost$eval((feed_dict = dict(x=valid$normalize_data, x_corrput=validCorrupt)))
    testCost<-dae_obj$cost$eval((feed_dict = dict(x=test$normalize_data, x_corrput=testCorrupt)))
    iterationLog<-rbind(iterationLog, data.frame("Iteration"=i, "Training"=trainingCost, "Validation"=validCost, "test"=testCost))
    cat("Iteration - ", i, ": Training - ", trainingCost, "  Validation - ", validCost, "\n")
  }
}

plotImage(test_data$images[55,], testCorrupt[55, ])
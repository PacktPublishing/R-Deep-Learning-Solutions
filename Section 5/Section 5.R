# COMPARE PCA AND RBM
# Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
library(tensorflow)
library(rbm)
library(ggplot2)
np <- import("numpy")

# Create TensorFlow session
# Reset the graph
tf$reset_default_graph()
# Starting session as interactive session
sess <- tf$InteractiveSession()

# Input data (MNIST)
mnist <- tf$examples$tutorials$mnist$input_data$read_data_sets("MNIST-data/",one_hot=TRUE)
trainX <- mnist$train$images[1:30000,]
trainY <- mnist$train$labels[1:30000,]
testX <- mnist$test$images
testY <- mnist$test$labels

# PCA
PCA_model <- prcomp(trainX, retx=TRUE)
RBM_model <- rbm::rbm(trainX, retx=TRUE, max_epoch=500,num_hidden =900)

# Predict on Train and Test data
PCA_pred_train <- predict(PCA_model)
RBM_pred_train <- predict(RBM_model,type='probs')

# Convert into dataframes
PCA_pred_train <- as.data.frame(PCA_pred_train)
RBM_pred_train <- as.data.frame(as.matrix(RBM_pred_train))

# Train actuals
trainY <- as.numeric(stringi::stri_sub(colnames(as.data.frame(trainY))[max.col(as.data.frame(trainY),ties.method="first")],2))

# Plot PCA and RBM
ggplot(PCA_pred_train, aes(PC1, PC2))+ 
  geom_point(aes(colour = trainY))+
  theme_bw()+labs(title="PCA - Distribution of digits")+  
  theme(plot.title = element_text(hjust = 0.5))

ggplot(RBM_pred_train, aes(Hidden_1, Hidden_2))+ 
  geom_point(aes(colour = trainY))+
  theme_bw()+labs(title="RBM - Distribution of digits")+  
  theme(plot.title = element_text(hjust = 0.5))

# Plots for variance explained in PCA
# No of Principal Components vs Cumulative Variance Explained
var_explain <- as.data.frame(PCA_model$sdev^2/sum(PCA_model$sdev^2))
var_explain <- cbind(c(1:784),var_explain,cumsum(var_explain[,1]))
colnames(var_explain) <- c("PcompNo.","Ind_Variance","Cum_Variance")
plot(var_explain$PcompNo.,var_explain$Cum_Variance, xlim = c(0,100),type='b',pch=16,xlab = "# of Principal Components",ylab = "Cumulative Variance",main = 'PCA - Explained variance')

# Plot to show reconstruction trainig error
plot(RBM_model,xlab = "# of epoch iterations",ylab = "Reconstruction error",main = 'RBM - Reconstruction Error')


## TENSOR FLOW Restricted Boltzman Machine
## Tensorflow implementation of Restricted Boltzman Machine for layerwise pretraining of deep autoencoders

# Import tenforflow libraries
# Sys.setenv(TENSORFLOW_PYTHON="C:/PROGRA~1/Python35/python.exe")
# Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
library(tensorflow)
library(ggplot2)
library(reshape)
np <- import("numpy")

# Create TensorFlow session
# Reset the graph
tf$reset_default_graph()
# Starting session as interactive session
sess <- tf$InteractiveSession()

# Input data (MNIST)
mnist <- tf$examples$tutorials$mnist$input_data$read_data_sets("MNIST-data/",one_hot=TRUE)
trainX <- mnist$train$images
trainY <- mnist$train$labels
testX <- mnist$test$images
testY <- mnist$test$labels

# Convert thr train data into gaussian distribution format i.e mean=0 and SD=1
trainX_normalised <- t(apply(trainX,1,function(x){
  return((x-mean(x))/sd(x))
}))

# Initialise parameters
num_input<-784L
num_hidden<-900L
alpha<-0.1

# Placeholder variables
vb <- tf$placeholder(tf$float32, shape = shape(num_input))
hb <- tf$placeholder(tf$float32, shape = shape(num_hidden))
W <- tf$placeholder(tf$float32, shape = shape(num_input, num_hidden))

# Phase 1 : Forward Pass
X = tf$placeholder(tf$float32, shape=shape(NULL, num_input))
prob_h0= tf$nn$sigmoid(tf$matmul(X, W) + hb)  
h0 = tf$nn$relu(tf$sign(prob_h0 - tf$random_uniform(tf$shape(prob_h0)))) 

# Look at sampling
sess$run(tf$global_variables_initializer())
s1 <- tf$constant(value = c(0.1,0.4,0.7,0.9))
cat(sess$run(s1))
s2=sess$run(tf$random_uniform(tf$shape(s1)))
cat(s2)
cat(sess$run(s1-s2))
cat(sess$run(tf$sign(s1 - s2)))
cat(sess$run(tf$nn$relu(tf$sign(s1 - s2))))

# Phase 2 : Backward Pass
prob_v1 = tf$matmul(h0, tf$transpose(W)) + vb
v1 = prob_v1 + tf$random_normal(tf$shape(prob_v1), mean=0.0, stddev=1.0, dtype=tf$float32)
h1 = tf$nn$sigmoid(tf$matmul(v1, W) + hb)    

# Calculate gradients
w_pos_grad = tf$matmul(tf$transpose(X), h0)
w_neg_grad = tf$matmul(tf$transpose(v1), h1)
CD = (w_pos_grad - w_neg_grad) / tf$to_float(tf$shape(X)[0])
update_w = W + alpha * CD
update_vb = vb + alpha * tf$reduce_mean(X - v1)
update_hb = hb + alpha * tf$reduce_mean(h0 - h1)

# Objective function
err = tf$reduce_mean(tf$square(X - v1))

# Initialise variables
cur_w = tf$Variable(tf$zeros(shape = shape(num_input, num_hidden), dtype=tf$float32))
cur_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
cur_hb = tf$Variable(tf$zeros(shape = shape(num_hidden), dtype=tf$float32))
prv_w = tf$Variable(tf$random_normal(shape=shape(num_input, num_hidden), stddev=0.01, dtype=tf$float32))
prv_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
prv_hb = tf$Variable(tf$zeros(shape = shape(num_hidden), dtype=tf$float32)) 

# Start tensorflow session
sess$run(tf$global_variables_initializer())
output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(X=trainX_normalised,
                                                                          W = prv_w$eval(),
                                                                          vb = prv_vb$eval(),
                                                                          hb = prv_hb$eval()))
prv_w <- output[[1]] 
prv_vb <- output[[2]]
prv_hb <-  output[[3]]
sess$run(err, feed_dict=dict(X= trainX_normalised, W= prv_w, vb= prv_vb, hb= prv_hb))

epochs=14
errors <- list()
weights <- list()
u=1
for(ep in 1:epochs){
  for(i in seq(0,(dim(trainX_normalised)[1]-100),100)){
    batchX <- trainX_normalised[(i+1):(i+100),]
    output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(X=batchX,
                                                                              W = prv_w,
                                                                              vb = prv_vb,
                                                                              hb = prv_hb))
    prv_w <- output[[1]] 
    prv_vb <- output[[2]]
    prv_hb <-  output[[3]]
    if(i%%10000 == 0){
      errors[[u]] <- sess$run(err, feed_dict=dict(X= trainX_normalised, W= prv_w, vb= prv_vb, hb= prv_hb))
      weights[[u]] <- output[[1]]
      u <- u+1
      cat(i , " : ")
    }
  }
  cat("epoch :", ep, " : reconstruction error : ", errors[length(errors)][[1]],"\n")
}

# Plot reconstruction error
error_vec <- unlist(errors)
plot(error_vec,xlab="# of batches",ylab="mean squared reconstruction error",main="RBM-Reconstruction MSE plot")

# Plot the last obtained weights
uw = t(weights[[length(weights)]])
numXpatches = 20
numYpatches=20
pixels <- list()
op <- par(no.readonly = TRUE)
par(mfrow = c(numXpatches,numYpatches), mar = c(0.2, 0.2, 0.2, 0.2), oma = c(3, 3, 3, 3))
for (i in 1:(numXpatches*numYpatches)) {
  denom <- sqrt(sum(uw[i, ]^2))
  pixels[[i]] <- matrix(uw[i, ]/denom, nrow = numYpatches, 
                        ncol = numXpatches)
  image(pixels[[i]], axes = F, col = gray((0:32)/32))
}
par(op)

# Sample case
sample_image <- trainX[1:4,]
mw=melt(sample_image)
mw$X3=floor((mw$X2-1)/28)+1
mw$X2=(mw$X2-1)%%28 + 1;
mw$X3=29-mw$X3
ggplot(data=mw)+geom_tile(aes(X2,X3,fill=value))+facet_wrap(~X1,nrow=2)+
  scale_fill_continuous(low='black',high='white')+coord_fixed(ratio=1)+
  labs(x=NULL,y=NULL,title="Sample digits - Actual")+
  theme(legend.position="none")+
  theme(plot.title = element_text(hjust = 0.5))


# Now pass the image for its reconstruction
hh0 = tf$nn$sigmoid(tf$matmul(X, W) + hb)
vv1 = tf$nn$sigmoid(tf$matmul(hh0, tf$transpose(W)) + vb)
feed = sess$run(hh0, feed_dict=dict( X= sample_image, W= prv_w, hb= prv_hb))
rec = sess$run(vv1, feed_dict=dict( hh0= feed, W= prv_w, vb= prv_vb))

# plot reconstructed images
mw=melt(rec)
mw$X3=floor((mw$X2-1)/28)+1
mw$X2=(mw$X2-1)%%28 + 1
mw$X3=29-mw$X3
ggplot(data=mw)+geom_tile(aes(X2,X3,fill=value))+facet_wrap(~X1,nrow=2)+
  scale_fill_continuous(low='black',high='white')+coord_fixed(ratio=1)+
  labs(x=NULL,y=NULL,title="Sample digits -Reconstructed")+
  theme(legend.position="none")+
  theme(plot.title = element_text(hjust = 0.5))



###########   COLLABORATIVE FILTERING WITH RBM
setwd("Set the working directory with movies.dat and ratings.dat files")

## Read movie lens data
txt <- readLines("movies.dat", encoding = "latin1")
txt_split <- lapply(strsplit(txt, "::"), function(x) as.data.frame(t(x), stringsAsFactors=FALSE))
movies_df <- do.call(rbind, txt_split)
names(movies_df) <- c("MovieID", "Title", "Genres")
movies_df$MovieID <- as.numeric(movies_df$MovieID)
movies_df$id_order <- 1:nrow(movies_df)

ratings_df <- read.table("ratings.dat", sep=":",header=FALSE,stringsAsFactors = F)
ratings_df <- ratings_df[,c(1,3,5,7)]
colnames(ratings_df) <- c("UserID","MovieID","Rating","Timestamp")

# Merge user ratings and movies
merged_df <- merge(movies_df, ratings_df, by="MovieID",all=FALSE)

# Remove unnecessary columns
merged_df[,c("Timestamp","Title","Genres")] <- NULL

# create % rating
merged_df$rating_per <- merged_df$Rating/5

# Generate a matrix of ratings
num_of_users <- 1000
num_of_movies <- length(unique(movies_df$MovieID))
trX <- matrix(0,nrow=num_of_users,ncol=num_of_movies)
for(i in 1:num_of_users){
  merged_df_user <- merged_df[merged_df$UserID %in% i,]
  trX[i,merged_df_user$id_order] <- merged_df_user$rating_per
}

summary(trX[1,]); summary(trX[2,]); summary(trX[3,])

# Import tenforflow libraries
# Sys.setenv(TENSORFLOW_PYTHON="C:/PROGRA~1/Python35/python.exe")
# Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
library(tensorflow)
np <- import("numpy")

# Create TensorFlow session
# Reset the graph
tf$reset_default_graph()
# Starting session as interactive session
sess <- tf$InteractiveSession()

# Model Parameters
num_hidden = 20
num_input = nrow(movies_df)
vb <- tf$placeholder(tf$float32, shape = shape(num_input))    #Number of unique movies
hb <- tf$placeholder(tf$float32, shape = shape(num_hidden))   #Number of features we're going to learn
W <- tf$placeholder(tf$float32, shape = shape(num_input, num_hidden))

#Phase 1: Input Processing
v0 = tf$placeholder(tf$float32,shape= shape(NULL, num_input))
prob_h0= tf$nn$sigmoid(tf$matmul(v0, W) + hb)
h0 = tf$nn$relu(tf$sign(prob_h0 - tf$random_uniform(tf$shape(prob_h0))))
#Phase 2: Reconstruction
prob_v1 = tf$nn$sigmoid(tf$matmul(h0, tf$transpose(W)) + vb) 
v1 = tf$nn$relu(tf$sign(prob_v1 - tf$random_uniform(tf$shape(prob_v1))))
h1 = tf$nn$sigmoid(tf$matmul(v1, W) + hb)

# RBM Parameters and functions
#Learning rate
alpha = 1.0
#Create the gradients
w_pos_grad = tf$matmul(tf$transpose(v0), h0)
w_neg_grad = tf$matmul(tf$transpose(v1), h1)
#Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf$to_float(tf$shape(v0)[1])
#Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf$reduce_mean(v0 - v1)
update_hb = hb + alpha * tf$reduce_mean(h0 - h1)

# Mean Absolute Error Function.
err = v0 - v1
err_sum = tf$reduce_mean(err * err)

# Initialise variables (current and previous)
cur_w = tf$Variable(tf$zeros(shape = shape(num_input, num_hidden), dtype=tf$float32))
cur_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
cur_hb = tf$Variable(tf$zeros(shape = shape(num_hidden), dtype=tf$float32))
prv_w = tf$Variable(tf$random_normal(shape=shape(num_input, num_hidden), stddev=0.01, dtype=tf$float32))
prv_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
prv_hb = tf$Variable(tf$zeros(shape = shape(num_hidden), dtype=tf$float32)) 

# Start tensorflow session
sess$run(tf$global_variables_initializer())
output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(v0=trX,
                                                                          W = prv_w$eval(),
                                                                          vb = prv_vb$eval(),
                                                                          hb = prv_hb$eval()))
prv_w <- output[[1]] 
prv_vb <- output[[2]]
prv_hb <-  output[[3]]
sess$run(err_sum, feed_dict=dict(v0=trX, W= prv_w, vb= prv_vb, hb= prv_hb))

# Train RBM
epochs= 500
errors <- list()
weights <- list()

for(ep in 1:epochs){
  for(i in seq(0,(dim(trX)[1]-100),100)){
    batchX <- trX[(i+1):(i+100),]
    output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(v0=batchX,
                                                                              W = prv_w,
                                                                              vb = prv_vb,
                                                                              hb = prv_hb))
    prv_w <- output[[1]] 
    prv_vb <- output[[2]]
    prv_hb <-  output[[3]]
    if(i%%1000 == 0){
      errors <- c(errors,sess$run(err_sum, feed_dict=dict(v0=batchX, W= prv_w, vb= prv_vb, hb= prv_hb)))
      weights <- c(weights,output[[1]])
      cat(i , " : ")
    }
  }
  cat("epoch :", ep, " : reconstruction error : ", errors[length(errors)][[1]],"\n")
}


# Plot reconstruction error
error_vec <- unlist(errors)
plot(error_vec,xlab="# of batches",ylab="mean squared reconstruction error",main="RBM-Reconstruction MSE plot")

# Recommendation
#Selecting the input user
inputUser = as.matrix(t(trX[75,]))
names(inputUser) <- movies_df$id_order

# Remove the movies not watched yet
inputUser1 <- inputUser[inputUser>0]

# Plot the top genre movies
top_rated_movies <- movies_df[as.numeric(names(inputUser1)[order(inputUser1,decreasing = TRUE)]),]$Title
top_rated_genres <- movies_df[as.numeric(names(inputUser1)[order(inputUser1,decreasing = TRUE)]),]$Genres
top_rated_genres <- as.data.frame(top_rated_genres,stringsAsFactors=F)
top_rated_genres$count <- 1
top_rated_genres <- aggregate(count~top_rated_genres,FUN=sum,data=top_rated_genres)
top_rated_genres <- top_rated_genres[with(top_rated_genres, order(-count)), ]
top_rated_genres$top_rated_genres <- factor(top_rated_genres$top_rated_genres, levels = top_rated_genres$top_rated_genres)
ggplot(top_rated_genres[top_rated_genres$count>1,],aes(x=top_rated_genres,y=count))+
  geom_bar(stat="identity")+ 
  theme_bw()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  labs(x="Genres",y="count",title="Top Rated Genres")+
  theme(plot.title = element_text(hjust = 0.5))


#Feeding in the user and reconstructing the input
hh0 = tf$nn$sigmoid(tf$matmul(v0, W) + hb)
vv1 = tf$nn$sigmoid(tf$matmul(hh0, tf$transpose(W)) + vb)
feed = sess$run(hh0, feed_dict=dict( v0= inputUser, W= prv_w, hb= prv_hb))
rec = sess$run(vv1, feed_dict=dict( hh0= feed, W= prv_w, vb= prv_vb))
names(rec) <- movies_df$id_order

# Select all recommended movies
top_recom_movies <- movies_df[as.numeric(names(rec)[order(rec,decreasing = TRUE)]),]$Title[1:10]
top_recom_genres <- movies_df[as.numeric(names(rec)[order(rec,decreasing = TRUE)]),]$Genres
top_recom_genres <- as.data.frame(top_recom_genres,stringsAsFactors=F)
top_recom_genres$count <- 1
top_recom_genres <- aggregate(count~top_recom_genres,FUN=sum,data=top_recom_genres)
top_recom_genres <- top_recom_genres[with(top_recom_genres, order(-count)), ]
top_recom_genres$top_recom_genres <- factor(top_recom_genres$top_recom_genres, levels = top_recom_genres$top_recom_genres)
ggplot(top_recom_genres[top_recom_genres$count>20,],aes(x=top_recom_genres,y=count))+
  geom_bar(stat="identity")+ 
  theme_bw()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  labs(x="Genres",y="count",title="Top Recommended Genres")+
  theme(plot.title = element_text(hjust = 0.5))

top_recom_movies <- movies_df[as.numeric(names(rec)[order(rec,decreasing = TRUE)]),]$Title[1:10]



######  DEEP BELIEF NETWORKS
# Import tenforflow libraries
# Sys.setenv(TENSORFLOW_PYTHON="C:/PROGRA~1/Python35/python.exe")
# Sys.setenv(TENSORFLOW_PYTHON_VERSION = 3)
library(tensorflow)
np <- import("numpy")

# Create TensorFlow session
# Reset the graph
tf$reset_default_graph()
# Starting session as interactive session
sess <- tf$InteractiveSession()

# Input data (MNIST)
mnist <- tf$examples$tutorials$mnist$input_data$read_data_sets("MNIST-data/",one_hot=TRUE)
trainX <- mnist$train$images
trainY <- mnist$train$labels
testX <- mnist$test$images
testY <- mnist$test$labels

# Creating DBN
RBM_hidden_sizes = c(900, 500 , 300 ) 


# Function to initialize RBM
RBM <- function(input_data, 
                num_input, 
                num_output,
                epochs = 5,
                alpha = 0.1,
                batchsize=100){
  
  # Placeholder variables
  vb <- tf$placeholder(tf$float32, shape = shape(num_input))
  hb <- tf$placeholder(tf$float32, shape = shape(num_output))
  W <- tf$placeholder(tf$float32, shape = shape(num_input, num_output))
  
  # Phase 1 : Forward Pass
  X = tf$placeholder(tf$float32, shape=shape(NULL, num_input))
  prob_h0= tf$nn$sigmoid(tf$matmul(X, W) + hb)  #probabilities of the hidden units
  h0 = tf$nn$relu(tf$sign(prob_h0 - tf$random_uniform(tf$shape(prob_h0)))) #sample_h_given_X
  
  # Phase 2 : Backward Pass
  prob_v1 = tf$nn$sigmoid(tf$matmul(h0, tf$transpose(W)) + vb) 
  v1 = tf$nn$relu(tf$sign(prob_v1 - tf$random_uniform(tf$shape(prob_v1)))) 
  h1 = tf$nn$sigmoid(tf$matmul(v1, W) + hb)   
  
  # Calculate gradients
  w_pos_grad = tf$matmul(tf$transpose(X), h0)
  w_neg_grad = tf$matmul(tf$transpose(v1), h1)
  CD = (w_pos_grad - w_neg_grad) / tf$to_float(tf$shape(X)[0])
  update_w = W + alpha * CD
  update_vb = vb + alpha * tf$reduce_mean(X - v1)
  update_hb = hb + alpha * tf$reduce_mean(h0 - h1)
  
  # Objective function
  err = tf$reduce_mean(tf$square(X - v1))
  
  # Initialise variables
  cur_w = tf$Variable(tf$zeros(shape = shape(num_input, num_output), dtype=tf$float32))
  cur_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
  cur_hb = tf$Variable(tf$zeros(shape = shape(num_output), dtype=tf$float32))
  prv_w = tf$Variable(tf$random_normal(shape=shape(num_input, num_output), stddev=0.01, dtype=tf$float32))
  prv_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32))
  prv_hb = tf$Variable(tf$zeros(shape = shape(num_output), dtype=tf$float32)) 
  
  # Start tensorflow session
  sess$run(tf$global_variables_initializer())
  output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(X=input_data,
                                                                            W = prv_w$eval(),
                                                                            vb = prv_vb$eval(),
                                                                            hb = prv_hb$eval()))
  prv_w <- output[[1]] 
  prv_vb <- output[[2]]
  prv_hb <-  output[[3]]
  sess$run(err, feed_dict=dict(X= input_data, W= prv_w, vb= prv_vb, hb= prv_hb))
  
  errors <- list()
  weights <- list()
  u=1
  for(ep in 1:epochs){
    for(i in seq(0,(dim(input_data)[1]-batchsize),batchsize)){
      batchX <- input_data[(i+1):(i+batchsize),]
      output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(X=batchX,
                                                                                W = prv_w,
                                                                                vb = prv_vb,
                                                                                hb = prv_hb))
      prv_w <- output[[1]] 
      prv_vb <- output[[2]]
      prv_hb <-  output[[3]]
      if(i%%10000 == 0){
        errors[[u]] <- sess$run(err, feed_dict=dict(X= batchX, W= prv_w, vb= prv_vb, hb= prv_hb))
        weights[[u]] <- output[[1]]
        u=u+1
        cat(i , " : ")
      }
    }
    cat("epoch :", ep, " : reconstruction error : ", errors[length(errors)][[1]],"\n")
  }
  
  w <- prv_w
  vb <- prv_vb
  hb <- prv_hb
  
  # Get the output
  input_X = tf$constant(input_data)
  ph_w = tf$constant(w)
  ph_hb = tf$constant(hb)
  
  out = tf$nn$sigmoid(tf$matmul(input_X, ph_w) + ph_hb)
  
  sess$run(tf$global_variables_initializer())
  return(list(output_data = sess$run(out),
              error_list=errors,
              weight_list=weights,
              weight_final=w,
              bias_final=hb))
}

#Since we are training, set input as training data
inpX = trainX

#Size of inputs is the number of inputs in the training set
num_input = ncol(inpX)

#Train RBM
RBM_output <- list()
for(i in 1:length(RBM_hidden_sizes)){
  size <- RBM_hidden_sizes[i]
  
  # Train the RBM
  RBM_output[[i]] <- RBM(input_data=inpX, 
                         num_input=num_input, 
                         num_output=size,
                         epochs = 5,
                         alpha = 0.1,
                         batchsize=100)
  
  # Update the input data
  inpX <- RBM_output[[i]]$output_data
  
  
  # Update the input_size
  num_input = size
  
  cat("completed size :", size,"\n")
}

# Plot reconstruction error
error_df <- data.frame("error"=c(unlist(RBM_output[[1]]$error_list),unlist(RBM_output[[2]]$error_list),unlist(RBM_output[[3]]$error_list)),
                       "batches"=c(rep(seq(1:length(unlist(RBM_output[[1]]$error_list))),times=3)),
                       "hidden_layer"=c(rep(c(1,2,3),each=length(unlist(RBM_output[[1]]$error_list)))),
                       stringsAsFactors = FALSE)

plot(error ~ batches,
     xlab = "# of batches",
     ylab = "Reconstruction Error",
     pch = c(1, 7, 16)[hidden_layer], 
     main = "Stacked RBM-Reconstruction MSE plot",
     data = error_df)

legend('topright',
       c("H1_900","H2_500","H3_300"), 
       pch = c(1, 7, 16))



###############  DEEP Restricted Boltzmann Machine
# Input data (MNIST)
mnist <- tf$examples$tutorials$mnist$input_data$read_data_sets("MNIST-data/",one_hot=TRUE)
trainX <- mnist$train$images
trainY <- mnist$train$labels
testX <- mnist$test$images
testY <- mnist$test$labels

# Global Parameters
learning_rate      = 0.005     
momentum      = 0.005     
minbatch_size      = 25        
hidden_layers = c(400,100) 
biases  = list(-1,-1)   

# Helper functions
arcsigm <- function(x){
  return(atanh((2*x)-1)*2)
}

sigm <- function(x){
  return(tanh((x/2)+1)/2)
}

binarize <- function(x){
  # truncated rnorm
  trnrom <- function(n, mean, sd, minval = -Inf, maxval = Inf){
    qnorm(runif(n, pnorm(minval, mean, sd), pnorm(maxval, mean, sd)), mean, sd)
  }
  return((x > matrix( trnrom(n=nrow(x)*ncol(x),mean=0,sd=1,minval=0,maxval=1), nrow(x), ncol(x)))*1)
}

re_construct <- function(x){
  x = x - min(x) + 1e-9
  x = x / (max(x) + 1e-9)
  return(x*255)
}

gibbs <- function(X,l,initials){
  if(l>1){
    bu <- (X[l-1][[1]] - matrix(rep(initials$param_O[[l-1]],minbatch_size),minbatch_size,byrow=TRUE)) %*%
      initials$param_W[l-1][[1]]
  } else {
    bu <- 0
  }
  if((l+1) < length(X)){
    td <- (X[l+1][[1]] - matrix(rep(initials$param_O[[l+1]],minbatch_size),minbatch_size,byrow=TRUE))%*%
      t(initials$param_W[l][[1]])
  } else {
    td <- 0
  }
  X[[l]] <- binarize(sigm(bu+td+matrix(rep(initials$param_B[[l]],minbatch_size),minbatch_size,byrow=TRUE)))
  return(X[[l]])
}

# Reparameterization
reparamBias <- function(X,l,initials){
  if(l>1){
    bu <- colMeans((X[[l-1]] - matrix(rep(initials$param_O[[l-1]],minbatch_size),minbatch_size,byrow=TRUE))%*%
                     initials$param_W[[l-1]])
  } else {
    bu <- 0
  }
  if((l+1) < length(X)){
    td <- colMeans((X[[l+1]] - matrix(rep(initials$param_O[[l+1]],minbatch_size),minbatch_size,byrow=TRUE))%*%
                     t(initials$param_W[[l]]))
  } else {
    td <- 0
  }
  initials$param_B[[l]] <- (1-momentum)*initials$param_B[[l]] + momentum*(initials$param_B[[l]] + bu + td)
  return(initials$param_B[[l]])
}

reparamO <- function(X,l,initials){
  initials$param_O[[l]] <- colMeans((1-momentum)*matrix(rep(initials$param_O[[l]],minbatch_size),minbatch_size,byrow=TRUE) + momentum*(X[[l]]))
  return(initials$param_O[[l]])
}

DRBM_initialize <- function(layers,bias_list){
  # Initialize model parameters and particles
  param_W <- list()
  for(i in 1:(length(layers)-1)){
    param_W[[i]] <- matrix(0L, nrow=layers[i], ncol=layers[i+1])
  }
  param_B <- list()
  for(i in 1:length(layers)){
    param_B[[i]] <- matrix(0L, nrow=layers[i], ncol=1) + bias_list[[i]]
  }
  param_O <- list()
  for(i in 1:length(param_B)){
    param_O[[i]] <- sigm(param_B[[i]])
  }
  param_X <- list()
  for(i in 1:length(layers)){
    param_X[[i]] <- matrix(0L, nrow=minbatch_size, ncol=layers[i]) + matrix(rep(param_O[[i]],minbatch_size),minbatch_size,byrow=TRUE)
  }  
  return(list(param_W=param_W,param_B=param_B,param_O=param_O,param_X=param_X))
}

# Run Initialize
X <- trainX/255
layers <- c(784,hidden_layers)
bias_list <- list(arcsigm(pmax(colMeans(X),0.001)),biases[[1]],biases[[2]])
initials <-DRBM_initialize(layers,bias_list)

# START TRAINING
batchX <- X[sample(nrow(X))[1:minbatch_size],]
for(iter in 1:1000){
  # Perform some learnings
  for(j in 1:100){
    # Initialize a data particle
    dat <- list()
    dat[[1]] <- binarize(batchX)
    for(l in 2:length(initials$param_X)){
      dat[[l]] <- initials$param_X[l][[1]]*0 + matrix(rep(initials$param_O[l][[1]],minbatch_size),minbatch_size,byrow=TRUE)
    }
    # Alternate gibbs sampler on data and free particles
    for(l in rep(c(seq(2,length(initials$param_X),2), seq(3,length(initials$param_X),2)),5)){
      dat[[l]] <- gibbs(dat,l,initials)
    }
    
    for(l in rep(c(seq(2,length(initials$param_X),2), seq(1,length(initials$param_X),2)),1)){
      initials$param_X[[l]] <- gibbs(initials$param_X,l,initials)
    }
    
    # Parameter update
    for(i in 1:length(initials$param_W)){
      initials$param_W[[i]] <- initials$param_W[[i]] + (learning_rate*((t(dat[[i]] - matrix(rep(initials$param_O[i][[1]],minbatch_size),minbatch_size,byrow=TRUE)) %*%
                                                                          (dat[[i+1]] - matrix(rep(initials$param_O[i+1][[1]],minbatch_size),minbatch_size,byrow=TRUE))) - 
                                                                         (t(initials$param_X[[i]] - matrix(rep(initials$param_O[i][[1]],minbatch_size),minbatch_size,byrow=TRUE)) %*%
                                                                            (initials$param_X[[i+1]] - matrix(rep(initials$param_O[i+1][[1]],minbatch_size),minbatch_size,byrow=TRUE))))/nrow(batchX))
    }
    
    for(i in 1:length(initials$param_B)){
      initials$param_B[[i]] <- colMeans(matrix(rep(initials$param_B[[i]],minbatch_size),minbatch_size,byrow=TRUE) + (learning_rate*(dat[[i]] - initials$param_X[[i]])))
    }
    
    # Reparameterization
    for(l in 1:length(initials$param_B)){
      initials$param_B[[l]] <- reparamBias(dat,l,initials)
    }
    for(l in 1:length(initials$param_O)){
      initials$param_O[[l]] <- reparamO(dat,l,initials)
    }
  }
  
  # Generate necessary outputs
  cat("Iteration:",iter," ","Mean of W of VL-HL1:",mean(initials$param_W[[1]])," ","Mean of W of HL1-HL2:",mean(initials$param_W[[2]]) ,"\n")
  cat("Iteration:",iter," ","SDev of W of VL-HL1:",sd(initials$param_W[[1]])," ","SDev of W of HL1-HL2:",sd(initials$param_W[[2]]) ,"\n")
  
  # Plot weight matrices
  W=diag(nrow(initials$param_W[[1]]))
  for(l in 1:length(initials$param_W)){
    W = W %*% initials$param_W[[l]]
    m = dim(W)[2] * 0.05
    w1_arr <- matrix(0,28*m,28*m)
    i=1
    for(k in 1:m){
      for(j in 1:28){
        vec <- c(W[(28*j-28+1):(28*j),(k*m-m+1):(k*m)])
        w1_arr[i,] <- vec
        i=i+1
      }
    }
    w1_arr = re_construct(w1_arr)
    w1_arr <- floor(w1_arr)
    image(w1_arr,axes = TRUE, col = grey(seq(0, 1, length = 256)))
  }
  
}





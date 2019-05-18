library(text2vec)
library(dplyr)
library(tensorflow)
library(RCurl)
library(tidytext)
data("movie_review")

labels <- as.matrix(data.frame("Positive_flag" = movie_review$sentiment,      
                               "negative_flag" = (1-movie_review$sentiment)))  

reviews <- data.frame("Sno" = 1:nrow(movie_review),
                      "text"=movie_review$review,
                      stringsAsFactors=F)

reviews_sortedWords <- reviews %>% unnest_tokens(word,text) %>% dplyr::count(word, sort = TRUE)
reviews_sortedWords$orderNo <- 1:nrow(reviews_sortedWords)
reviews_sortedWords <- as.data.frame(reviews_sortedWords)

reviews_words <- reviews %>% unnest_tokens(word,text)
reviews_words <- plyr::join(reviews_words,reviews_sortedWords,by="word")

reviews_words_sno <- list()
for(i in 1:length(reviews$text)){
  reviews_words_sno[[i]] <- c(subset(reviews_words,Sno==i,orderNo))
  if(i %% 500 == 0) cat("completed : ", i,"\n")
}

wordLen_counter <- data.frame("counts"=unlist(lapply(reviews_words_sno,function(x) length(x[[1]])))) %>% dplyr::count(counts)
hist(x=wordLen_counter$counts,
     freq=wordLen_counter$n)

reviews_words_sno <- lapply(reviews_words_sno,function(x) {
  x <- x$orderNo
  if(length(x)>150){
    return (x[1:150])
  } else {
    return(c(rep(0,150-length(x)),x))
  }
})

train_samples <- caret::createDataPartition(c(1:length(labels[,1])),p = 0.7)$Resample1

train_reviews <- reviews_words_sno[train_samples]
test_reviews <- reviews_words_sno[-train_samples]

train_reviews <- do.call(rbind,train_reviews)
test_reviews <- do.call(rbind,test_reviews)

train_labels <- as.matrix(labels[train_samples,])
test_labels <- as.matrix(labels[-train_samples,])

tf$reset_default_graph()
sess<-tf$InteractiveSession()

n_input<-15
step_size<-10
n.hidden<-2
n.class<-2

lr<-0.01
batch<-200
iteration = 500

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

lstm<-function(x, weight, bias){
  x = tf$unstack(x, step_size, 1)
  lstm_cell = tf$contrib$rnn$BasicLSTMCell(n.hidden, forget_bias=1.0, state_is_tuple=TRUE)
  cell_output = tf$contrib$rnn$static_rnn(lstm_cell, x, dtype=tf$float32)
  last_vec=tail(cell_output[[1]], n=1)[[1]]
  return(tf$matmul(last_vec, weights) + bias)
}

yhat = lstm(x, weights, bias)

cost = tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(logits=yhat, labels=y))
optimizer = tf$train$AdamOptimizer(learning_rate=lr)$minimize(cost)

sess$run(tf$global_variables_initializer())
train_error <- c()
for(i in 1:iteration){
  spls <- sample(1:dim(train_reviews)[1],batch)
  sample_data<-train_reviews[spls,]
  sample_y<-train_labels[spls,]
  sample_data=tf$reshape(sample_data, shape(batch, step_size, n_input))
  out<-optimizer$run(feed_dict = dict(x=sample_data$eval(session = sess), y=sample_y))
  
  if (i %% 1 == 0){
    cat("iteration - ", i, "Training Loss - ",  cost$eval(feed_dict = dict(x=sample_data$eval(), y=sample_y)), "\n")
  }
  train_error <-  c(train_error,cost$eval(feed_dict = dict(x=sample_data$eval(), y=sample_y)))
}


plot(train_error, main="Training sentiment prediction error", xlab="Iterations", ylab = "Train Error")

test_data=tf$reshape(test_reviews, shape(-1, step_size, n_input))
cost$eval(feed_dict=dict(x=test_data$eval(), y=test_labels))

library(h2o)
library(caret)
localH2O = h2o.init(ip = "localhost", port = 54321, nthreads = 8)

occupancy_train <- read.csv("datatraining.txt",stringsAsFactors = T)
occupancy_test <- read.csv("datatest.txt",stringsAsFactors = T)

x = c("Temperature", "Humidity", "Light", "CO2", "HumidityRatio")
y = "Occupancy"

occupancy_train$Occupancy <- as.factor(occupancy_train$Occupancy)
occupancy_test$Occupancy <- as.factor(occupancy_test$Occupancy)

occupancy_train.hex <- as.h2o(x = occupancy_train, destination_frame = "occupancy_train.hex")
occupancy_test.hex <- as.h2o(x = occupancy_test, destination_frame = "occupancy_test.hex")

occupancy.deepmodel <- h2o.deeplearning(x = x,	 	 
                                        y = y, 	 
                                        training_frame = occupancy_train.hex,	 	 
                                        validation_frame = occupancy_test.hex,	 	 
                                        standardize = F,	 	 
                                        activation = "Rectifier",	 	 
                                        epochs = 50,	 	 
                                        seed = 1234567,	 	 
                                        hidden = 5,	 	 
                                        variable_importances = T,
                                        nfolds = 5)	

xval_performance <- h2o.performance(occupancy.deepmodel,xval = T)
xval_performance@metrics$AUC

train_performance <- h2o.performance(occupancy.deepmodel,train = T)
train_performance@metrics$AUC

test_performance <- h2o.performance(occupancy.deepmodel,valid = T)
test_performance@metrics$AUC
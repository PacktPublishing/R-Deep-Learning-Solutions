library(h2o)
library(caret)
library(pROC)

localH2O = h2o.init(ip = "localhost", port = 54321, nthreads = 8)

occupancy_train <- read.csv("datatraining.txt",stringsAsFactors = T)
occupancy_test <- read.csv("datatest.txt",stringsAsFactors = T)

x = c("Temperature", "Humidity", "Light", "CO2", "HumidityRatio")
y = "Occupancy"

occupancy_train$Occupancy <- as.factor(occupancy_train$Occupancy)
occupancy_test$Occupancy <- as.factor(occupancy_test$Occupancy)

occupancy_train.hex <- as.h2o(x = occupancy_train, destination_frame = "occupancy_train.hex")
occupancy_test.hex <- as.h2o(x = occupancy_test, destination_frame = "occupancy_test.hex")

occupancy_train.glm <- h2o.glm(x = x,  
                               y = y,    
                               training_frame = occupancy_train.hex, 
                               seed = 1234567,     
                               family = "binomial",  
                               lambda_search = TRUE,
                               alpha = 0.5,
                               nfolds = 5    
)

occupancy_train.glm@model$training_metrics@metrics$AUC

occupancy_train.glm@model$cross_validation_metrics@metrics$AUC

yhat <- h2o.predict(occupancy_train.glm, occupancy_test.hex)

yhat$pmax <- pmax(yhat$p0, yhat$p1, na.rm = TRUE) 
roc_obj <- pROC::roc(c(as.matrix(occupancy_test.hex$Occupancy)), c(as.matrix(yhat$pmax)))
auc(roc_obj)

h2o.varimp_plot(occupancy_train.glm,num_of_features = 5)

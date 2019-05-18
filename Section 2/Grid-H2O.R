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

activation_opt <- c("Rectifier","RectifierWithDropout", "Maxout","MaxoutWithDropout")
hidden_opt <- list(5, c(5,5))
epoch_opt <- c(10,50,100)
l1_opt <- c(0,1e-3,1e-4)
l2_opt <- c(0,1e-3,1e-4)

hyper_params <- list( activation = activation_opt,
                      hidden = hidden_opt,
                      epochs = epoch_opt,
                      l1 = l1_opt,
                      l2 = l2_opt
)

search_criteria <- list(strategy = "RandomDiscrete", max_models=300)

dl_grid <- h2o.grid(x = x,
                    y = y,
                    algorithm = "deeplearning",
                    grid_id = "deep_learn",
                    hyper_params = hyper_params,
                    search_criteria = search_criteria,
                    training_frame = occupancy_train.hex,
                    nfolds = 5)

d_grid <- h2o.getGrid("deep_learn",sort_by = "auc", decreasing = T)
best_dl_model <- h2o.getModel(d_grid@model_ids[[1]])

xval_performance.grid <- h2o.performance(best_dl_model,xval = T)
xval_performance.grid@metrics$AUC

train_performance.grid <- h2o.performance(best_dl_model,train = T)
train_performance.grid@metrics$AUC

yhat <- h2o.predict(best_dl_model, occupancy_test.hex)

yhat$pmax <- pmax(yhat$p0, yhat$p1, na.rm = TRUE) 
roc_obj <- pROC::roc(c(as.matrix(occupancy_test.hex$Occupancy)), c(as.matrix(yhat$pmax)))
pROC::auc(roc_obj)


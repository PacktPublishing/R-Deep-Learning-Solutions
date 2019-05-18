require(SAENET)

load_occupancy_data<-function(train){
  xFeatures = c("Temperature", "Humidity", "Light", "CO2", "HumidityRatio")
  yFeatures = "Occupancy"
  if(train){
    occupancy_ds <- as.matrix(read.csv("datatraining.txt",stringsAsFactors = T))
  } else
  {
    occupancy_ds <- as.matrix(read.csv("datatest.txt",stringsAsFactors = T))
  }
  occupancy_ds<-apply(occupancy_ds[, c(xFeatures, yFeatures)], 2, FUN=as.numeric) 
  return(list("data"=occupancy_ds, "xFeatures"=xFeatures, "yFeatures"=yFeatures))
}

minmax.normalize<-function(ds, scaler=NULL){
  if(is.null(scaler)){
    for(f in ds$xFeatures){
      scaler[[f]]$minval<-min(ds$data[,f])
      scaler[[f]]$maxval<-max(ds$data[,f])
      ds$data[,f]<-(ds$data[,f]-scaler[[f]]$minval)/(scaler[[f]]$maxval-scaler[[f]]$minval)
    }
    ds$scaler<-scaler
  } else
  {
    for(f in ds$xFeatures){
      ds$data[,f]<-(ds$data[,f]-scaler[[f]]$minval)/(scaler[[f]]$maxval-scaler[[f]]$minval)
    }
  }
  return(ds)
}

occupancy_train <-load_occupancy_data(train=T)
occupancy_test <- load_occupancy_data(train = F)

occupancy_train<-minmax.normalize(occupancy_train, scaler = NULL)
occupancy_test<-minmax.normalize(occupancy_test, scaler =
                                   occupancy_train$scaler)

SAE_obj<-SAENET.train(X.train= subset(occupancy_train$data,
                                      select=-c(Occupancy)), n.nodes=c(4, 3, 2), unit.type ="tanh",
                      lambda = 1e-5, beta = 1e-5, rho = 0.01, epsilon = 0.01,
                      max.iterations=1000)

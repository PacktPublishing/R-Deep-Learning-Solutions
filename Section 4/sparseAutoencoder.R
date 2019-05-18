require(autoencoder)

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
occupancy_test<-minmax.normalize(occupancy_test, scaler = occupancy_train$scaler)

nl<-3 
N.hidden<-7 
unit.type<-"tanh"
lambda<-0.001 
rho<-0.01 
beta<-6 
max.iterations<-2000 
epsilon<-0.001 

spe_ae_obj <- autoencode(X.train=occupancy_train$data,  X.test = occupancy_test$data, nl=nl,
                         N.hidden=N.hidden, unit.type=unit.type,lambda=lambda,beta=beta,
                         epsilon=epsilon,rho=rho,max.iterations=max.iterations)

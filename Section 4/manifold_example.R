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
occupancy_test<-minmax.normalize(occupancy_test, scaler = occupancy_train$scaler)

pca_obj <- prcomp(subset(occupancy_train$data, select=-c(Occupancy)),
                 center = TRUE,
                 scale. = TRUE)

plot(pca_obj, type = "l")

xlab("Principal Componets")
ds <- data.frame(occupancy_train$data, pca_obj$x[,1:2])
component_plot <- qplot(x=PC1, y=PC2, data=ds, colour=factor(Occupancy)) + theme(legend.position="none") 

require(ggplot2)
SAE_obj<-SAENET.train(X.train= subset(occupancy_train$data, select=-c(Occupancy)), n.nodes=c(4, 3, 1), unit.type ="tanh", lambda = 1e-5, beta = 1e-5, rho = 0.01, epsilon = 0.01, max.iterations=1000)  

plot(SAE_obj[[3]]$X.output[,1], col="blue", xlab = "Node 1 of layer 3", ylab = "Values")
ix<-occupancy_train$data[,6]==1  
points(seq(1, nrow(SAE_obj[[3]]$X.output),by=1)[ix], SAE_obj[[3]]$X.output[ix,1], col="red")
legend(7000,0.45, c("0","1"), lty=c(0,0), pch=1, col=c("blue","red")) # gives the legend lines the correct color and width

SAE_obj<-SAENET.train(X.train= subset(occupancy_train$data, select=-c(Occupancy)), n.nodes=c(4, 3, 2), unit.type ="tanh", lambda = 1e-5, beta = 1e-5, rho = 0.01, epsilon = 0.01, max.iterations=1000)  

plot(SAE_obj[[3]]$X.output[,1], SAE_obj[[3]]$X.output[,2], col="blue", xlab = "Node 1 of layer 3", ylab = "Node 2 of layer 3")
ix<-occupancy_train$data[,6]==1  
points(SAE_obj[[3]]$X.output[ix,1], SAE_obj[[3]]$X.output[ix,2], col="red")
legend(0,0.6, c("0","1"), lty=c(0,0), pch=1, col=c("blue","red")) # gives the legend lines the correct color and width

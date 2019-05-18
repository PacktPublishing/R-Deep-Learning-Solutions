add_noise<-function(data, frac=0.10, corr_type=c("masking", "saltPepper", "none")){
  if(length(corr_type)>1) corr_type<-corr_type[1] 
  data_noise = data
  nROW<-nrow(data)
  nCOL<-ncol(data)
  nMask<-floor(frac*nCOL)
  if(corr_type=="masking"){
    for( i in 1:nROW){
      maskCol<-sample(nCOL, nMask)
      data_noise[i,maskCol]<-0
    }
  } else if(corr_type=="saltPepper"){
    minval<-min(data)
    maxval<-max(data)
    for( i in 1:nROW){
      maskCol<-sample(nCOL, nMask)
      randval<-runif(length(maskCol))
      ixmin<-randval<0.5
      ixmax<-randval>=0.5
      if(sum(ixmin)>0) data_noise[i,maskCol[ixmin]]<-minval
      if(sum(ixmax)>0) data_noise[i,maskCol[ixmax]]<-maxval
    }
  } else
  {
    data_noise<-data
  }
  return(data_noise)
}
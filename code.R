rm(list = ls())
setwd("D:/github_oicjacky/ML-Stock-price-forecast/ALLDATA")
tw1101<-read.csv( file = "tw1101.csv")
tw2002<-read.csv( file = "tw2002.csv")
tw2327<-read.csv( file = "tw2327.csv")
tw2330<-read.csv( file = "tw2330.csv")
tw2412<-read.csv( file = "tw2412.csv")
tw2610<-read.csv( file = "tw2610.csv")
tw3008<-read.csv( file = "tw3008.csv")
allstock<-list(tw1101=tw1101,tw2002=tw2002,tw2327=tw2327,tw2330=tw2330,tw2412=tw2412,tw2610=tw2610,tw3008=tw3008)
stock_names<-c("tw1101","tw2002","tw2327","tw2330","tw2412","tw2610","tw3008")

#檢查資料是否有遺漏值
summary(tw2330)
prime <- function(x){
  if(x==1) return ("NULL")
  y <- c()
  a <- c(2:x)
  b <- min(a[which(x%%a==0)])
  x<-x/b
  if(x==1) return ("NULL")
  y <-c(y,b)
  while(!x==1){
    
    a <- c(2:x)
    b <- min(a[which(x%%a==0)])
    x<-x/b
    y <- c(y,b)
  }
  return(y)
}#找質因數
for (i in 1:7) {
  print(
    prime(dim(allstock[[i]])[1]-23)
  )
}
prime(2044)
##############################################
#載入package
#install.packages(c("timetk","tidyquant","tibbletime","cowplot","recipes","rsample","yardstick","keras","tensorflow"))
library(tidyverse)
library(glue)
library(forcats)

# Time Series
library(timetk)
library(tidyquant)
library(tibbletime)

# Visualization
library(cowplot)

# Preprocessing
library(recipes)

# Sampling / Accuracy
library(rsample)
library(yardstick) 

# Modeling
library(keras)
library(tensorflow)

#計算lstm_tw2330
set.seed(10)
tw2330<-read.csv( file = "tw2330.csv")
LSTM<-function(dataset){
  diffed<-diff(dataset,differences = 1)#前後兩天的股價相差
  lag_transform <- function(x, k= 1){
    lagged =  c(rep(NA, k), x[1:(length(x)-k)])
    DF = as.data.frame(cbind(lagged, x))
    colnames(DF) <- c( paste0('x-', k), 'x')
    DF[is.na(DF)] <- 0
    return(DF)
  }
  supervised = lag_transform(diffed,1)
  supervised[(dim(supervised)[1]+1),]<-c(supervised[dim(supervised)[1],2],0)
  supervised<-supervised[2:dim(supervised)[1],]
  row.names(supervised)<-1:dim(supervised)[1]
  N = nrow(supervised)#總共data
  n = round(N *0.8, digits = 0)
  train = supervised[1:n, ]#traindata
  test  = supervised[(n+1):N,]#testdata
  scale_data=function(train, test, feature_range = c(0, 1)) {
    x = train
    fr_min = feature_range[1]
    fr_max = feature_range[2]
    std_train = ((x - min(x) ) / (max(x) - min(x)  ))
    std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
    scaled_train = std_train *(fr_max -fr_min) + fr_min
    scaled_test = std_test *(fr_max -fr_min) + fr_min
    return(list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
  }
  Scaled = scale_data(train, test, c(-1, 1))
  y_train = Scaled$scaled_train[, 2]#train 目標差x
  x_train = Scaled$scaled_train[, 1]#train 慢一天的差x-1
  y_test = Scaled$scaled_test[, 2]#test 慢一天的差x
  x_test = Scaled$scaled_test[, 1]#test 慢一天的差x-1
  invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){
    min = Scaled$scaler[1]
    max = Scaled$scaler[2]
    t = length(scaled)
    mins = feature_range[1]
    maxs = feature_range[2]
    inverted_dfs = numeric(t)
    for(i in 1:t){
      X = (scaled[i]- mins)/(maxs - mins)
      rawValues = X *(max - min)+min
      inverted_dfs[i] <- rawValues
    }
    return(inverted_dfs)
  }
  dim(x_train) <- c(length(x_train), 1, 1)
  # specify required arguments
  X_shape2 = dim(x_train)[2]
  X_shape3 = dim(x_train)[3]
  batch_size = 7                # must be a common factor of both the train and test samples
  units = 1                     # can adjust this, in model tuninig phase
  model <- keras_model_sequential() 
  model%>%
    layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
    layer_dense(units = 1)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
    metrics = c('accuracy')
  )
  summary(model)
  Epochs = 100   
  for(i in 1:Epochs ){
    model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
    model %>% reset_states()
  }
  L = length(x_test)
  scaler = Scaled$scaler
  predictions = numeric(L)
  for(i in 1:L){
    X = x_test[i:(i+6)]
    dim(X) = c(7,1,1)
    yhat = model %>% predict(X, batch_size=batch_size)
    #invert scaling
    yhat= invert_scaling(yhat[1], scaler,c(-1, 1))
    # invert differencing
    yhat  = yhat + dataset[(n+i+1)]
    # store
    predictions[i] <- yhat[1]
  }
  return(predictions)
}
tw2330<-read.csv( file = "tw2330.csv")
tw2330_sma10<-LSTM(tw2330$sma10)#tw2330$sma10[2047:2556] vs tw2330_sma10[1:510]
tw2330_wma10<-LSTM(tw2330$wma10)#同上
tw2330_k_value<-LSTM(tw2330$k_value)
tw2330_d_value<-LSTM(tw2330$d_value)
tw2330_RSI9<-LSTM(tw2330$RSI.9)
tw2330_CCI14<-LSTM(tw2330$CCI14)
tw2330_R<-LSTM(tw2330$R)
tw2330_it<-LSTM(tw2330$it)
tw2330_de<-LSTM(tw2330$de)
tw2330_fi<-LSTM(tw2330$fi)
Mom<-momentum(tw2330$close, n = 9)#Momentum
Mom[1:9]<-c(-0.6,1.3,-1,-0.7,-1.85,-2.1,-2.4,-1.7,-1.4)
tw2330_MOm<-LSTM(Mom)

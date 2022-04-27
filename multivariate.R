#TIME SERIES 
library(ggplot2)
library(readr)
require(tidyverse)
require(corrplot)
library(tseries)
library(MASS)
library(dplyr)
library(forecast)
library(tseries)
library(zoo)
library(dplyr)


#LOAD THE DATASET
dati<- read.csv("data.csv", sep=";",stringsAsFactors=FALSE, dec=',', header=TRUE)

#convert to date
dati$day<- as.Date(dati$day)

#data ranges from 1st november 2018 to 31st january 2020
#1371 observations
data<-dati %>% slice(52:1422) 


#create a new dataset with weekly data of sales, convrate and visits
#new dataset "weeklydata1", contains these three variables for each city in the range of the above dates
library(lubridate)
library(zoo)


dayweek <- function(x)format(x, '%Y.%W')

weekly_sales<- data%>% group_by(city, dayweek(day)) %>%
  summarise(totsales= sum(sales), day= first(day))

weekly_rate<- data%>% group_by(city, dayweek(day)) %>%
  summarise(totrate= sum(convrate), day= first(day))

weekly_visits<- data%>% group_by(city, dayweek(day)) %>%
  summarise(totvis= sum(visits), day= first(day))

#merge the datasets into 1
weekly_data<- merge(weekly_sales, weekly_rate,all.x=TRUE, all.y =TRUE)
weekly_data1<- merge(weekly_data, weekly_visits,all.x=TRUE, all.y = TRUE)

#split the dataset 
#test set with january 2020 
#train set with all the other data

mil<- weekly_data1[64:68, 1:6, drop = FALSE]
nap<-weekly_data1[132:136, 1:6, drop = FALSE]
rom<-weekly_data1[200:204, 1:6, drop = FALSE]
test_set<- rbind(mil, rom, nap)
train_set<- weekly_data1[-c(64:68, 132:136, 200:204), ]


###################################################################
#################### MULTIVARIATE MILAN ###########################
##################################################################
#time series
milan<- weekly_data1[1:63, 1:6, drop = FALSE] 

#ggplot sales
ggplot(milan, aes(day, totsales, color= city))+ #color city può essere
  geom_line()+
  ylab('tot sales x week')+xlab('date')+
  scale_x_date(limits = c(min(milan$day), max(milan$day)))

#ggplot convrate
ggplot(milan, aes(day, totrate, color=city))+ #color city può essere
  geom_line()+
  ylab('tot cvrate x week')+xlab('date')+
  scale_x_date(limits = c(min(milan$day), max(milan$day)))

#ggplot visits
ggplot(milan, aes(day, totvis, color=city))+ #color city può essere
  geom_line()+
  ylab('tot cvrate x week')+xlab('date')+
  scale_x_date(limits = c(min(milan$day), max(milan$day)))

#transform to ts function
milan<- weekly_data1[1:63, 4:6, drop = FALSE] 

mts_milan = ts(milan,
           frequency = 61,#61 is the weeks 
           start = c(2018),
           end=c(2020))
mts_milan
#plot of the multivariate time series
plot(mts_milan, main="time series milan")

#plot of the three variables
theme_set(theme_bw())
autoplot(mts_milan) +
  ggtitle("Time Series Plot of the `multivariate Time-Series") +
  theme(plot.title = element_text(hjust = 0.5))

class(mts_milan)



#adf test to check for stationarity
#non stationary so proceed with differicing
apply(mts_milan, 2, adf.test)

# Differencing the multivariate time-series
require(tseries)
require(lmtest)
require(lattice)
library(zoo)
library(MTS)
#difference operation on a vector of time series. 
#Default order of differencing is 1.
stnry = diffM(mts_milan) 

# Retest
#passed adf test with one differencing
apply(stnry, 2, adf.test) 
#plot the stationary time series
plot.ts(stnry)

autoplot(ts(stnry,
            start = c(2018),
            end= c(2020),
            frequency = 61)) +
  ggtitle("Plot of the stationary Time Series Milan")

#LAG ORDER IDENTIFICATION
#according to the var select function, putting max lag as default as 10, the lag order that minimizes the AIC is the 10th

library(vars)
VARselect(stnry, 
          type = "none", #type of deterministic regressors to include. We use none because the time series was made stationary using differencing above. 
          lag.max = 10)

#Now that we have the lag order we can fit the model with Var function
fit_milan<- vars::VAR(stnry,
                   lag.max = 10, #highest lag order for lag length selection according to the choosen ic
                   ic = "AIC", #information criterion
                   type = "none") #type of deterministic regressors to include
summary(fit_milan)


#Residual diagnostics
#Check for causality between function with the Granger test

serial.test(fit_milan)

causality(fit_milan, #VAR model
          cause = c("totrate"))

causality(fit_milan, #VAR model
          cause = c("totvis"))


#IRF FUNCTION
#one variable, in this case visits is the shock and the function checks the response of the other variable (sales) to the shock

#for sales
#significant
irf1<- irf( fit_milan, impulse = "totvis", response= "totsales", n.ahead= 10, boot= TRUE, run= 200, ci= 0.95)
plot(irf1, ylab= "totsales", main="response of sales to visits Milan")

#for convrate
#significant
irf3<- irf( fit_milan, impulse = "totrate", response= "totsales", n.ahead= 10, boot= TRUE, run= 200, ci= 0.95)
plot(irf3, ylab= "totsales", main="response of sales to convrate Milan")

#insignificant if we put sales as shock
irf2<- irf( fit_milan, impulse = "totsales", response= "totvis", n.ahead= 10, boot= TRUE, run= 200, ci= 0.95)
plot(irf2, ylab= "totvis", main="response of vists to sales Milan")

#variance decomposition
#if there is a shock to a variable, how much of the response is due to the main variable and how much is due to the other variabels in the system
library(vars)
library(tseries)
vd<- fevd(fit_milan, n.ahead=5)
vd

plot(vd)

## Forecasting Milan fitted model

fcast = predict(fit_milan, n.ahead = 5) 
par(mar = c(2.5,2.5,2.5,2.5))
plot(fcast)
fcast
fcast$fcst
#extract forecast
fcast$fcst[1]

totsales= fcast$fcst[1]; totsales # type list

#extracting the forecast column
x = totsales$totsales[,1];x

x = cumsum(x) + 5473.72
par(mar = c(2.5,2.5,1,2.5)) #bottom, left, top, and right
plot.ts(x)





################################################################
################### MULTIVARIATE NAPLES ########################
################################################################

naples<- train_set[64:126, 1:6, drop = FALSE]

#ggplot sales
ggplot(naples, aes(day, totsales, color= city))+ #color city può essere
  geom_line()+
  ylab('tot sales x week')+xlab('date')+
  scale_x_date(limits = c(min(naples$day), max(naples$day)))

#ggplot convrate
ggplot(naples, aes(day, totrate, color=city))+ #color city può essere
  geom_line()+
  ylab('tot cvrate x week')+xlab('date')+
  scale_x_date(limits = c(min(naples$day), max(naples$day)))

#ggplot visits
ggplot(naples, aes(day, totvis, color=city))+ #color city può essere
  geom_line()+
  ylab('tot cvrate x week')+xlab('date')+
  scale_x_date(limits = c(min(naples$day), max(naples$day)))

#transform to ts function
naples<- weekly_data1[64:126, 4:6, drop = FALSE] 

mts_naples = ts(naples,
               frequency = 61,#61 is the weeks 
               start = c(2018),
               end=c(2020))
mts_naples
#plot of the multivariate time series
plot(mts_naples, main=' time series naples')

#plot of the three variables
theme_set(theme_bw())
autoplot(mts_naples) +
  ggtitle("Time Series Plot of the `multivariate Time-Series") +
  theme(plot.title = element_text(hjust = 0.5))

class(mts_naples)

#check for seasonality
acf(ts_milan$totsales)
acf(ts_milan$totrate)
acf(ts_milan$totvis)

#adf test to check for stationarity
#non stationary so proceed with differicing
apply(mts_naples, 2, adf.test)

# Differencing the multivariate time-series
#difference operation on a vector of time series. 
#Default order of differencing is 1.
stnry1 = diffM(mts_naples) 

# Retest
#passed adf test with one differencing
apply(stnry1, 2, adf.test) 
#plot the stationary time series
plot.ts(stnry1)

autoplot(ts(stnry1,
            start = c(2018),
            end= c(2020),
            frequency = 61)) +
  ggtitle("Plot of the stationary Time Series Naples")

#LAG ORDER IDENTIFICATION
#according to the var select function, putting max lag as default as 10, the lag order that minimizes the AIC is the 10th
VARselect(stnry1, 
          type = "none", #type of deterministic regressors to include. We use none because the time series was made stationary using differencing above. 
          lag.max = 10)

#Now that we have the lag order we can fit the model with Var function
fit_naples<- vars::VAR(stnry1,
                      lag.max = 10, #highest lag order for lag length selection according to the choosen ic
                      ic = "AIC", #information criterion
                      type = "none") #type of deterministic regressors to include
summary(fit_naples)


#Residual diagnostics
#Check for causality between function with the Granger test

serial.test(fit_naples)

causality(fit_naples, #VAR model
          cause = c("totrate"))

causality(fit_naples, #VAR model
          cause = c("totvis"))

#IRF FUNCTION
#one variable, in this case visits is the shock and the function checks the response of the other variable (sales) to the shock

#for sales
#significant
irf_naples<- irf( fit_naples, impulse = "totvis", response= "totsales", n.ahead= 10, boot= TRUE, run= 200, ci= 0.95)
plot(irf2, ylab= "totsales", main="response of sales to visits  Naples")

#for convrate
#significant
irf1_naples<- irf( fit_naples, impulse = "totrate", response= "totsales", n.ahead= 10, boot= TRUE, run= 200, ci= 0.95)
plot(irf1_naples, ylab= "totsales", main="response of sales to convrate Naples")

#insignificant if we put sales as shock
irf2_naples<- irf( fit_naples, impulse = "totsales", response= "totvis", n.ahead= 10, boot= TRUE, run= 200, ci= 0.95)
plot(irf2_naples, ylab= "totvis", main="response of vists to sales Naples")

#variance decomposition
#if there is a shock to a variable, how much of the response is due to the main variable and how much is due to the other variabels in the system
vd_napels<- fevd(fit_naples, n.ahead=5)
plot(vd_napels)

## Forecasting Naples fitted model
#we forecast 10 months
fcast_naples = predict(fit_naples, n.ahead = 5) 
par(mar = c(2.5,2.5,2.5,2.5))
plot(fcast_naples)
fcast$fcst
#extract forecast
fcast_naples$fcst[1]

totsales_naples= fcast$fcst[1]; totsales # type list

#extracting the forecast column
x = totsales_naples$totsales[,1];x

x = cumsum(x) + 5473.72
par(mar = c(2.5,2.5,1,2.5)) #bottom, left, top, and right
plot.ts(x)


################################################################
#################### MULTIVARIATE ROMA #########################
################################################################
rome<- train_set[127:189, 1:6, drop = FALSE]

#ggplot sales
ggplot(rome, aes(day, totsales, color= city))+ 
  geom_line()+
  ylab('tot sales x week')+xlab('date')+
  scale_x_date(limits = c(min(rome$day), max(rome$day)))

#ggplot convrate
ggplot(rome, aes(day, totrate, color=city))+ #color city può essere
  geom_line()+
  ylab('tot cvrate x week')+xlab('date')+
  scale_x_date(limits = c(min(rome$day), max(rome$day)))

#ggplot visits
ggplot(rome, aes(day, totvis, color=city))+ #color city può essere
  geom_line()+
  ylab('tot cvrate x week')+xlab('date')+
  scale_x_date(limits = c(min(rome$day), max(rome$day)))

#transform to ts function
rome<- weekly_data1[127:189, 4:6, drop = FALSE] 

mts_rome = ts(rome,
                frequency = 61,#61 is the weeks 
                start = c(2018),
                end=c(2020))
mts_rome
#plot of the multivariate time series
plot(mts_rome, main="time series rome")

#plot of the three variables
theme_set(theme_bw())
autoplot(mts_rome) +
  ggtitle("Time Series Plot of the `multivariate Time-Series") +
  theme(plot.title = element_text(hjust = 0.5))

class(mts_rome)

#adf test to check for stationarity
#non stationary so proceed with differicing
apply(mts_rome, 2, adf.test)

# Differencing the multivariate time-series
#difference operation on a vector of time series. 
#Default order of differencing is 1.
stnry2 = diffM(mts_rome) 

# Retest
#passed adf test with one differencing
apply(stnry2, 2, adf.test) 
#plot the stationary time series
plot.ts(stnry2)

autoplot(ts(stnry2,
            start = c(2018),
            end= c(2020),
            frequency = 61)) +
  ggtitle("Plot of the stationary Time Series Rome")

#LAG ORDER IDENTIFICATION
#according to the var select function, putting max lag as default as 10, the lag order that minimizes the AIC is the 10th
VARselect(stnry2, 
          type = "none", #type of deterministic regressors to include. We use none because the time series was made stationary using differencing above. 
          lag.max = 10)

#Now that we have the lag order we can fit the model with Var function
fit_rome<- vars::VAR(stnry2,
                       lag.max = 10, #highest lag order for lag length selection according to the choosen ic
                       ic = "AIC", #information criterion
                       type = "none") #type of deterministic regressors to include
summary(fit_rome)
accuracy(fit_rome$varresult[[1]])
accuracy(fit_naples$varresult[[1]])
accuracy(fit_milan$varresult[[1]])



#Residual diagnostics
#Check for causality between function with the Granger test

serial.test(fit_rome)

causality(fit_rome, #VAR model
          cause = c("totrate"))

causality(fit_rome, #VAR model
          cause = c("totvis"))

#IRF FUNCTION
#one variable, in this case visits is the shock and the function checks the response of the other variable (sales) to the shock

#for sales
#significant
irf_rome<- irf( fit_rome, impulse = "totvis", response= "totsales", n.ahead= 10, boot= TRUE, run= 200, ci= 0.95)
plot(irf_rome, ylab= "totsales", main="response of sales to visits Rome")

#for convrate
#significant
irf1_rome<- irf( fit_rome, impulse = "totrate", response= "totsales", n.ahead= 10, boot= TRUE, run= 200, ci= 0.95)
plot(irf1_rome, ylab= "totsales", main="response of sales to convrate Rome")

#insignificant if we put sales as shock
irf2_rome<- irf( fit_rome, impulse = "totsales", response= "totvis", n.ahead= 10, boot= TRUE, run= 200, ci= 0.95)
plot(irf2_rome, ylab= "totvis", main="response of vists to sales Rome")

#variance decomposition
#if there is a shock to a variable, how much of the response is due to the main variable and how much is due to the other variabels in the system
vd_rome<- fevd(fit_rome, n.ahead=5)
vd_rome
plot(vd_rome)

## Forecasting Naples fitted model
#we forecast 10 months
fcast_rome = predict(fit_rome, n.ahead = 5) 
par(mar = c(2.5,2.5,2.5,2.5))
plot(fcast_rome)
fcast_rome$fcst

#extract forecast
fcast_rome$fcst[1]

totsales_rome= fcast_rome$fcst[1]; totsales # type list

#extracting the forecast column
x = totsales_rome$totsales[,1];x

x = cumsum(x) + 5473.72
par(mar = c(2.5,2.5,1,2.5)) #bottom, left, top, and right
plot.ts(x)











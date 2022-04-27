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

#prendiamo dal 1 dicembre 2018 al 31 gennaio 2020
data<-dati %>% slice(52:1422) 
#togliamo il mese di aprile 2019
#new_data<- data[-c(364:453),]

#creiamo weekly sales in cui mettiamo il totale delle sales
library(lubridate)
library(zoo)


dayweek <- function(x)format(x, '%Y.%W')
weekly_sales<- data%>% group_by(city, dayweek(day)) %>%
  summarise(totsales= sum(sales), day= first(day))

#data pre-processing fino a qui. creazione di weekly sales

#MODELING
#plot the sales from november 2018 to january 2020 before splitting
#plot the sales 
ggplot(weekly_sales, aes(day, totsales, color=city))+ #color city può essere
  geom_line()+
  ylab('tot sales x week')+xlab('date')+
  scale_x_date(limits = c(min(weekly_sales$day), max(weekly_sales$day)))


#split the dataset
#test set= january
#train set= il resto
mil<- weekly_sales[64:68, 1:4, drop = FALSE]
nap<-weekly_sales[132:136, 1:4, drop = FALSE]
rom<-weekly_sales[200:204, 1:4, drop = FALSE]
test_set<- rbind(mil, rom, nap)

train_set<- weekly_sales[-c(64:68, 132:136, 200:204), ]


#grafico train set delle sales
train_set %>% 
  ggplot(aes(x=day, y=totsales, color= city))+ geom_line()+ 
  scale_x_date(limits = c(min(train_set$day), max(train_set$day)))

#clean the sales
#time series cleaned
#sales_ts<- ts(train_set[,c("totsales")])
#train_set$clean_sales<-tsclean(sales_ts)

#ggplot(train_set, aes(day, clean_sales, color= city))+
#geom_line()+ scale_x_date('Time')+
#ylab('tot sales x week')+xlab('')

#moving average important to understand trend and mean value estimation in data
train_set$sales_ma<- ma(train_set$totsales, order=7)
train_set$sales_ma_30<- ma(train_set$totsales, order=30)

#how much of the sales is MA and how much Count
ggplot()+
  geom_line(data= train_set, aes(x= day, y=totsales, colour='counts'))+
  geom_line(data= train_set, aes(x= day, y=sales_ma, colour='moving average (w)'))+
  geom_line(data= train_set, aes(x= day, y=sales_ma_30, colour='moving average(m)'))+
  ylab("sales")

##################################MILAN################################
##################################MILAN################################
#################################MILAN#################################
#define the cities for the analysis
ts_milan<- train_set[1:63, 1:6, drop = FALSE]

#decomposition of MILAN
#ts has seasonality trend and the rest is the error
#we will use stl

sales_milan= ts(na.omit(ts_milan[]$sales_ma), frequency=7)
decomp=stl(sales_milan, s.window='periodic')
library(forecast)
deseasonal_sales<- seasadj(decomp)  
plot(decomp) 

#now we can fit arima to stationary model
#stationary means mean and variance with no trend or seasonality

adf.test(sales_milan, alternative='stationary') #p-value=0.9
#p-value >0.05, we are acceptuing the null....it is not stationary
#remedies= simple differencing/log/log difference
#the number of difference is I in Arima

#Autocorrelation and choosing order

#ACF AND PACF
acf(sales_milan, main='ACF MILAN')
pacf(sales_milan, main='PACF MILAN')

sales_deseasonal_milan= diff(deseasonal_sales, diff=2) #1 differencing to pass the adf test
plot(sales_deseasonal_milan)
acf(sales_deseasonal_milan)
pacf(sales_deseasonal_milan)
#mean is now stationary but not the variance
adf.test(sales_deseasonal_milan, alternative='stationary')

#passed the adf test by differencing 1 time , now it is 0.01

#acf(sales_d1, main='acf for difference')
#pacf(sales_d1, main='pacf for difference')


###################################ORA ROMA############################
###################################ORA ROMA#########################
#define the city
ts_rome<- train_set[127:189, 1:6, drop = FALSE]

#decomposition of MILAN
#ts has seasonality trend and the rest is the error
#we will use stl

sales_rome= ts(na.omit(ts_rome[]$sales_ma), frequency=7)
decomp=stl(sales_rome, s.window='periodic')
library(forecast)
deseasonal_rome<- seasadj(decomp)  
plot(decomp) 

#now we can fit arima to stationary model
#stationary means mean and variance with no trend or seasonality

adf.test(sales_rome, alternative='stationary') #p-value=0.6276
#p-value >0.05, we are acceptuing the null....it is staionary
#remedies= simple differencing/log/log difference
#the number of difference is I in Arima

#Autocorrelation and choosing order

#ACF AND PACF
acf(sales_rome, main='ACF ROME')
pacf(sales_rome, main='PACF ROME')

sales_differenced= diff(deseasonal_rome, diff=1) #1 differencing to pass the adf test
plot(sales_differenced)
acf(sales_differenced)
#mean is now stationary but not the variance
adf.test(sales_differenced, alternative='stationary') #now it is stationary

#passed the adf test by differencing 1 time , now it is 0.01

#acf(sales_d1, main='acf for difference')
#pacf(sales_d1, main='pacf for difference')


##################################################
######################NAPLES########################
#define the city
ts_naples<- train_set[64:126, 1:6, drop = FALSE]

#decomposition of MILAN
#ts has seasonality trend and the rest is the error
#we will use stl

sales_naples= ts(na.omit(ts_naples[]$sales_ma), frequency=7)
decomp=stl(sales_naples, s.window='periodic')
library(forecast)
deseasonal_naples<- seasadj(decomp)  
plot(decomp) 

#now we can fit arima to stationary model
#stationary means mean and variance with no trend or seasonality

adf.test(sales_naples, alternative='stationary') #p-value=0.01
#p-value >0.05, we are acceptuing the null....it is staionary
#remedies= simple differencing/log/log difference
#the number of difference is I in Arima

#Autocorrelation and choosing order

#ACF AND PACF
acf(sales_naples, main='ACF NAPLES')
pacf(sales_naples, main='PACF NAPLES')

sales_differenced_naples= diff(deseasonal_naples, diff=2) #2 differencing to pass the adf test
plot(sales_differenced_naples)
acf(sales_differenced_naples)
#mean is now stationary but not the variance
adf.test(sales_differenced_naples, alternative='stationary') #now it is stationary

#passed the adf test by differencing 1 time , now it is 0.01

#acf(sales_d1, main='acf for difference')
#pacf(sales_d1, main='pacf for difference')






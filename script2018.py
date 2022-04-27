# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 07:54:34 2022

@author: Utente
"""

#IMPORT LIBRARIES 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

#UPLOAD DATASET
dataset1= pd.read_csv("Sales.csv")
dataset2= pd.read_csv("WebTraffic.csv")
dataset1.info()

#MERGE ONE COLUMN
joined_dataset= dataset1.join(dataset2, lsuffix='', rsuffix='_dataset2')
joined_dataset.drop('city_dataset2', axis=1, inplace=True)
joined_dataset.drop('day_dataset2', axis=1, inplace=True) 

year_2018=joined_dataset[0:234]

#Transform the date column 
joined_dataset['day']=pd.to_datetime(joined_dataset.day)
year_2018['day']=pd.to_datetime(year_2018.day)

#OPERAZIONI: aggiungi le variabili 
joined_dataset["sales"]= joined_dataset['visits']* joined_dataset['convrate']
joined_dataset["total_spending"]=joined_dataset['avspend']* joined_dataset['sales']


year_2018["sales"]= year_2018['visits']* year_2018['convrate']
year_2018["total_spending"]=year_2018['avspend']* year_2018['sales']


#ISOLATE DECEMBER 2018
#3-24 dicembre
dec_2018= year_2018.iloc[147:212, :]

#DATA VISUALISATION
#plot for day visits city 
#COME SISTEMARE LE DATE
abc=sns.lineplot(x='day', y= 'visits',data=year_2018 , hue='city') 
abc.tick_params(axis='x', rotation=45)
#abc.set_xticklabels([str(i) for i in abc.get_xticks()], fontsize = 7)


#AVERAGE SPEND 2018
sns.displot(x="avspend", data=dec_2018, kde=True, color="green")

#CONVERSION RATE 2018
#plot 3: convrate "Ecommerce conversion rate, computed as the fraction of website visitors that finalized an online purchase of the products"
sns.displot(x="convrate", data= dec_2018, hue="city", multiple="stack")

##plot 4: 
#COME SISTEMARE LE DATE
e=sns.lineplot(x='day', y='sales', data=dec_2018) 
e.tick_params(axis='x', rotation=45)
#VEDI SE AGGIUNGERE SALES


#CORRELATION MATRIX
sns.heatmap(dec_2018.corr())

#divide the dataset in different areas
Milan= dec_2018[dec_2018["city"]=="Milan"]
Naples = dec_2018[dec_2018["city"]=="Naples"]
Rome= dec_2018[dec_2018["city"]=="Rome"]

#plot per area
#set the parameters 
#DA AGGIUNGER LE ALTRE CITTA'
#COME FAR VEDERE MEGLIO I NOMI
#riduci font label


#MILAN AND ITS VARIABLES: visits, sales, convrate, avgspend, total spending
f = plt.figure(figsize=(20, 8))
gs = f.add_gridspec(2, 3) #add grid specifications, 2 rows and 3 col
f.suptitle('Milan Variables')
#VEDI LINEPLOT SALES DI TUTTO 2018
#GIRA TUTTE LE DATE
# darkgrid
with sns.axes_style("dark"):
    ax = f.add_subplot(gs[0, 0])
    b=sns.lineplot(x='day', y='visits', data=Milan)
    b.tick_params(axis='x', which='major', labelsize=8)

#ticklabels scritte piccole
#solo labels x
#b.set_yticklabels(b.get_yticks(), size = 15)
with sns.axes_style("dark"):
    ax = f.add_subplot(gs[0, 1])
    c= sns.lineplot(x='day', y='sales', data=Milan)
    c.tick_params(axis='x', which='major', labelsize=8)

with sns.axes_style("dark"):
    ax = f.add_subplot(gs[0, 2])
    d= sns.lineplot(x='day', y='convrate', data=Milan)
    d.tick_params(axis='x', which='major', labelsize=8)
    
with sns.axes_style("dark"):
    ax = f.add_subplot(gs[1,0])
    e=sns.lineplot(x='day', y='avspend', data=Milan)
    e.tick_params(axis='x', which='major', labelsize=8)

with sns.axes_style("dark"):
    ax = f.add_subplot(gs[1,1])
    f=sns.lineplot(x='day', y='total_spending', data=Milan)
    f.tick_params(axis='x', which='major', labelsize=8)

sns.despine()



#ROME AND ITS VARIABLES: visits, sales, convrate, avgspend, total spending
f = plt.figure(figsize=(20, 8))
gs = f.add_gridspec(2, 3) #add grid specifications, 2 rows and 3 col
f.suptitle('Rome Variables')
#VEDI LINEPLOT SALES DI TUTTO 2018
#GIRA TUTTE LE DATE
# darkgrid
with sns.axes_style("dark"):
    ax = f.add_subplot(gs[0, 0])
    b=sns.lineplot(x='day', y='visits', data=Rome, color='orange')
    b.tick_params(axis='x', which='major', labelsize=8)

#ticklabels scritte piccole
#solo labels x
#b.set_yticklabels(b.get_yticks(), size = 15)
with sns.axes_style("dark"):
    ax = f.add_subplot(gs[0, 1])
    c= sns.lineplot(x='day', y='sales', data=Rome, color='orange')
    c.tick_params(axis='x', which='major', labelsize=8)

with sns.axes_style("dark"):
    ax = f.add_subplot(gs[0, 2])
    d= sns.lineplot(x='day', y='convrate', data=Rome, color='orange')
    d.tick_params(axis='x', which='major', labelsize=8)
    
with sns.axes_style("dark"):
    ax = f.add_subplot(gs[1,0])
    e=sns.lineplot(x='day', y='avspend', data=Rome, color='orange')
    e.tick_params(axis='x', which='major', labelsize=8)

with sns.axes_style("dark"):
    ax = f.add_subplot(gs[1,1])
    f=sns.lineplot(x='day', y='total_spending', data=Rome, color='orange')
    f.tick_params(axis='x', which='major', labelsize=8)

sns.despine()


#NAPLES AND ITS VARIABLES: visits, sales, convrate, avgspend, total spending
f = plt.figure(figsize=(20, 8))
gs = f.add_gridspec(2, 3) #add grid specifications, 2 rows and 3 col
f.suptitle('Naples Variables')
#VEDI LINEPLOT SALES DI TUTTO 2018
#GIRA TUTTE LE DATE
# darkgrid
with sns.axes_style("dark"):
    ax = f.add_subplot(gs[0, 0])
    b=sns.lineplot(x='day', y='visits', data=Naples, color='green')
    b.tick_params(axis='x', which='major', labelsize=8)

#ticklabels scritte piccole
#solo labels x
#b.set_yticklabels(b.get_yticks(), size = 15)
with sns.axes_style("dark"):
    ax = f.add_subplot(gs[0, 1])
    c= sns.lineplot(x='day', y='sales', data=Naples, color='green')
    c.tick_params(axis='x', which='major', labelsize=8)

with sns.axes_style("dark"):
    ax = f.add_subplot(gs[0, 2])
    d= sns.lineplot(x='day', y='convrate', data=Naples, color='green')
    d.tick_params(axis='x', which='major', labelsize=8)
    
with sns.axes_style("dark"):
    ax = f.add_subplot(gs[1,0])
    e=sns.lineplot(x='day', y='avspend', data=Naples, color='green')
    e.tick_params(axis='x', which='major', labelsize=8)

with sns.axes_style("dark"):
    ax = f.add_subplot(gs[1,1])
    f=sns.lineplot(x='day', y='total_spending', data=Naples, color='green')
    f.tick_params(axis='x', which='major', labelsize=8)

sns.despine()



#plot the sales in Milan Naples and Rome a confronto
#RIMPICCIOLISCI LA Y
#è tutto da ruotare
f = plt.figure(figsize=(15, 5))
gs = f.add_gridspec(1, 3) #add grid specifications, 2 rows and 3 col
f.suptitle('Sales in Milan, Rome and Naples')

with sns.axes_style("dark"):
    ax = f.add_subplot(gs[0, 0])
    g= sns.lineplot(data = Milan.groupby('sales').max(), x="day", y="sales")
    g.tick_params(axis='x', which='major', labelsize=10, rotation=45)

with sns.axes_style("dark"):
    ax = f.add_subplot(gs[0, 1])
    h= sns.lineplot(data = Rome.groupby('sales').max(), x="day", y="sales", color='orange')
    h.tick_params(axis='x', which='major', labelsize=10, rotation=45)

with sns.axes_style("dark"):
    ax = f.add_subplot(gs[0, 2])
    i= sns.lineplot(data = Naples.groupby('sales').max(), x="day", y="sales", color='green')
    i.tick_params(axis='x', which='major', labelsize=10, rotation=45)



    
#altro
sns.histplot(data = Milan, y = "visits")   
sns.histplot(data=Rome, y='visits')
#Plots solo per dicembre 2018: dal 3 dicembre al 24 dicembre 

#plots 
#1. visits per day 
#CAPIRE SE POSSIAMO METTERE QUALCOSA DENTRO ALLE DATE ORDINATE
#INUTILE USALO SOLO PER TICKLABELS
df_plt1 = dec_2018.groupby('day').size().to_frame('Count')
g = sns.barplot(x = df_plt1.index, y = 'Count', data = df_plt1)
g.set(title='Sales in december 2018')
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
g.set_ylabel('Size')

#K-MEANS
#DATA SPLITTING
#NBUYERS-VISITS
X = year_2018.iloc[:, [4,3 ]].values
y = year_2018.iloc[:, 5].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#KEMEANS
from sklearn.cluster import KMeans   

#COMPUTE INTER CLASS VARIANCE
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#FIT K-MEANS TO THE DATASET
#YKEMEANS NOFCLUSTERS
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
#plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('visits')
plt.ylabel('avgspend')
plt.legend()
plt.show()


#FOR DECEMBER
X = dec_2018.iloc[:, [4,3 ]].values
y = dec_2018.iloc[:, 5].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#KEMEANS
from sklearn.cluster import KMeans   

#COMPUTE INTER CLASS VARIANCE
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#FIT K-MEANS TO THE DATASET
#YKEMEANS NOFCLUSTERS
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
#plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('average spending')
plt.ylabel('visits')
plt.legend()
plt.show()
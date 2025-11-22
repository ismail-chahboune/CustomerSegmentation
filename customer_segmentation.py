# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 14:00:27 2025

@author: Ismail
"""

import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA





df = pd.read_csv("customer_segmentation.csv")
print(df)
print(df.head())
print(df.columns)
print(df.info())

print(df.isna().sum())
print(df.shape)
df.dropna(inplace=True)

print(df.isna().sum())

print(df.describe())


print(df["Education"].value_counts())
print(df["Marital_Status"].value_counts())


df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"],dayfirst = True)
print(df.info())

df["Age"] = datetime.now().year - df["Year_Birth"] 
print(df["Age"])

df["Total Children"] = df["Kidhome"] + df["Teenhome"] 

print(df["Total Children"]) 



total_spend = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]

df["Total_spending"] = df[total_spend].sum(axis=1)
print(df["Total_spending"])


 

sns.histplot(df["Age"], bins=30,kde = True )
plt.title("Age distribution")
plt.show()



sns.histplot(df["Income"], bins = 30 , kde = True)
plt.title(" Income distribution")
plt.show()

 

sns.histplot(df["Total_spending"],bins=30,kde=True)

plt.title("Spending distribution")
plt.show()




sns.boxplot(x="Education",y="Income",data=df)
plt.xticks(rotation=45)
plt.title("Income by Education level")
plt.show()



sns.boxplot(x="Marital_Status",y="Total_spending",data=df)
plt.xticks(rotation=45)
plt.title("Spending by Martial Status")
plt.show()


corr = df[["Income","Age","Recency","Total_spending","NumWebPurchases","NumStorePurchases"]].corr()

sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.title("Correlation matrix")
plt.show()


 

pivot_income = df.pivot_table(values="Income",index = "Education",columns="Marital_Status",aggfunc="mean")
print(pivot_income)


sns.heatmap(pivot_income,annot=True,fmt=".0f",cmap="YlGnBu")
plt.title("avg income by education and marital status")
plt.show()



group1 = df.groupby("Education")["Total_spending"].mean().sort_values(ascending=False)
print(group1)



group1.plot(kind="bar",color = "green")
plt.title("Avg spending by education")
plt.ylabel("Average Total Spending") 
plt.xticks(rotation=45)
plt.show() 



df["AcceptedAny"] = df[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response"]].sum(axis=1)


print(df["AcceptedAny"])
print(df["AcceptedAny"].unique()) 

df["AcceptedAny"] = df["AcceptedAny"].apply(lambda x:1 if x>0 else 0)

groupe2 = df.groupby("Marital_Status")["AcceptedAny"].mean().sort_values(ascending=False)

groupe2.plot(kind="bar",color="red")
plt.title("Campain acceptance by marital status")
plt.xticks(rotation=45)
plt.ylabel("Acceptance Rate")
plt.show()

 
bins = [18,30,40,50,60,70,90]  
labels = ["18-29","30--39","40-49","50--59","60-69","70+"] 

df["Agegroup"] = pd.cut(df["Age"] , bins = bins , labels = labels)
print(df["Agegroup"])

group3 = df.groupby("Agegroup")["Income"].mean()


group3.plot(kind="bar",color="yellow")
plt.title("Avg Income by age group")
plt.xlabel("Average Income")
plt.show()



features = ["Age","Income","Total_spending","NumWebPurchases","NumStorePurchases","NumWebVisitsMonth","Recency"]

X=df[features].copy()

print(X)


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


wcss = [] 
for i in range(2,10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
print(wcss)



plt.plot(range(2,10),wcss,marker='o')
plt.title("Elbow method for optimal K")
plt.xlabel("Nbr of Clusters")
plt.ylabel("WCSS")
plt.show()


kmeans=KMeans(n_clusters=6)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(df.head())




cluster_summary = df.groupby("Cluster")[features].mean() 

print(cluster_summary)

df["Cluster"].value_counts()



pca = PCA(n_components = 2)
pca_data = pca.fit_transform(X_scaled)
df["PCA1"],df["PCA2"] = pca_data[:,0] , pca_data[:,1]


sns.scatterplot(x="PCA1",y="PCA2",hue = "Cluster",data=df,palette="Set1")
plt.title("Cluster segmentation (PCA)")
plt.show()



import joblib
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
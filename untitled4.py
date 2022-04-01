import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

data=pd.read_csv("tested.csv")

print("No of passengers:" +str(len(data.index)))

#Analyzing data
sns.countplot(x="Survived", data = data)

sns.countplot(x="Survived",hue="Sex",data=data)

sns.countplot(x="Survived",hue="Pclass",data=data)

data["Age"].plot.hist()
data["Fare"].plot.hist(bins=20, figsize=(10,5))


sns.countplot(x="SibSp",data=data)


#Data Wrangling
data.isnull().sum()
sns.heatmap(data.isnull(),yticklabels=False )

sns.boxplot(x="Pclass",y="Age",data=data)

#Dropping the cabin column
data.drop("Cabin",axis=1,inplace=True)
#Dropping the null values
data.dropna(inplace=True)

#Converting the string values to categorical values
# as in logistic regression it only takes two values
sex=pd.get_dummies(data["Sex"],drop_first=True)
embark=pd.get_dummies(data["Embarked"],drop_first=True)
pc=pd.get_dummies(data["Pclass"],drop_first=True)

data=pd.concat([data,sex,embark,pc],axis=1)

data.drop(['Sex','Pclass','Embarked','Ticket','PassengerId','Name'],axis=1,inplace=True)
print(data.head())


#Training and Testing the data

X=data.drop("Survived",axis=1)
Y=data["Survived"]

#splitting data into train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)

#Scaling the values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#importing logistic regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train , y_train)

predictions = logmodel.predict(x_test)

from sklearn.metrics import classification_report
classification_report(y_test,predictions)

#calculating accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions)*100)




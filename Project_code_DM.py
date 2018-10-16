# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 23:56:46 2018

@author: dominika.leszko
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r'C:\Users\dominika.leszko\Desktop\NOVA IMS\Data Mining\DM Project\A2Z Insurance.csv')

#from sklearn.preprocessing import Imputer
df = pd.DataFrame(df)
pd.set_option('display.max_columns', None)

#DOMINIKA: Why replacing empty columns with NaN?
#from sklearn.preprocessing import Imputer
#df = df.where(df!='')  # Replacing empty values with NaN
#df=df.values
#imputer = Imputer(missing_values=np.nan, strategy = 'median', axis = 0)
#imputer = imputer.fit(df[:,-2:-1])
#
#df[:,-2:-1] = imputer.transform(df[:,-2:-1])

df.info()
df.describe()

#Renaming columns for easier analysis
df.columns.values

coldict={'Customer Identity':'CustId', 'First Policy´s Year':'1stPolYear', 'Brithday Year':'BirthYear',
       'Educational Degree':'EduDegree', 'Gross Monthly Salary':'GrossMthSalary',
       'Geographic Living Area':'GeoLivArea', 'Has Children (Y=1)':'HasChild',
       'Customer Monetary Value':'CustMonetVal', 'Claims Rate':'ClaimRate', 'Premiums in LOB: Motor':'PremLOBMotor',
       'Premiums in LOB: Household':'PremLOBHousehold', 'Premiums in LOB: Health':'PremLOBHealth',
       'Premiums in LOB:  Life':'PremLOBLife', 'Premiums in LOB: Work Compensations':'PremLOBWorkCompensation'}

df.rename(columns=coldict, inplace=True)

df.columns.values

##############################Handling Error Data##############################################################
#
#   1st Pol Year drop >2016 DONE
#   Bday Year drop >2016 and <1900 DONE
#   no error in education DONE
#   gross month salary-drop outliers ????????????????????
#   geography no errors DONE
#   has children no errors DONE
#   customer monetary, claims rate lets keep for now
#   LOB all good  cause from system

df.shape#10296 rows, 14 columns


df['1stPolYear'].describe()
#Drop values >2016, as the database comes from 2016
df = df.drop(df[df['1stPolYear']>2016].index)


df['BirthYear'].describe()
#Drop values >2016 and <1900
df=df.drop(df[df['BirthYear']>2016].index)
df=df.drop(df[df['BirthYear']<1900].index)

df['GrossMthSalary'].describe()
sns.boxplot(x=df['GrossMthSalary'])
plt.xlim(30000,)
df=df.drop(df[df['GrossMthSalary']>30000].index)

#We dropped only 4 rows...

#Can we drop odd LOB Premiums? See boxplots

sns.boxplot(x=df['PremLOBMotor'])
df=df.drop(df[df['PremLOBMotor']>2000].index)


sns.boxplot(x=df['PremLOBHousehold'])#sigma rule? here is more outliers

sns.boxplot(x=df['PremLOBHealth'])
df=df.drop(df[df['PremLOBHealth']>5000].index)


sns.boxplot(x=df['PremLOBLife'])


sns.boxplot(x=df['PremLOBWorkCompensation'])#sigma rule? more outliers


#############################Handling null values################################################################

sns.heatmap(df.isnull())

df.isna().any()
df.isnull().sum(axis=0)


#Replace Birthday with Regression on Salary

#Replace Salary with Regression on Bday

#1st Year-NN (30 nulls - mean/median?)
plt.figure(figsize=(8,6))
df['1stPolYear'].hist()

#Education-NN (17 nulls -mean/median?)
plt.figure(figsize=(8,6))
df['EduDegree'].hist()

#Geogrpahy-NN (1 null- missing a lot of other columns, DROP!)
df=df.dropna(subset=['GeoLivArea'])

#Has children-NN (21 nulls, KNN/replace with 1?) after doing knn if 80% has 1 then its good!
df['HasChild']=df['HasChild'].fillna(1)

#Customer Monet-NO NULLS
#Claims Ratio - NO NULLS

df['PremLOBWorkCompensation']=df['PremLOBWorkCompensation'].fillna(0)
df['PremLOBLife']=df['PremLOBLife'].fillna(0)
df['PremLOBHealth']=df['PremLOBHealth'].fillna(0)
df['PremLOBMotor']=df['PremLOBMotor'].fillna(0)



#df.dropna(subset=['1stPolYear'], inplace=True)
##PremLOBWorkCompensation nulls replace with median
#df['PremLOBWorkCompensation'].hist(bins=40)
#df['PremLOBWorkCompensation']=df['PremLOBWorkCompensation'].fillna(df['PremLOBWorkCompensation'].median())
#df['PremLOBWorkCompensation'].isnull().any()
#
##PremLOBLife nulls replace with median
#df['PremLOBLife'].hist(bins=40)
#df['PremLOBLife']=df['PremLOBLife'].fillna(df['PremLOBLife'].median())
#df['PremLOBLife'].isnull().any()
#
##PremLOBHealth nulls replace with median
#df['PremLOBHealth'].hist(bins=130)
#plt.xlim(0,1000)
#df['PremLOBHealth']=df['PremLOBHealth'].fillna(df['PremLOBHealth'].median())
#df['PremLOBHealth'].isnull().any()
#
##PremLOB nulls replace with mean
#df['PremLOB'].hist(bins=130)
#plt.xlim(0,1000)#normal distribution, replace with mean
#df['PremLOB']=df['PremLOBHealth'].fillna(df['PremLOB'].mean())
#df['PremLOB'].isnull().any()
#
##HasChild nulls replace with median
#df['HasChild'].hist(bins=10)
#df['HasChild']=df['HasChild'].fillna(df['HasChild'].median())
#df['HasChild'].isnull().any()
#
#
##GrossMthSalary nulls replace with
#df['GrossMthSalary'].hist(bins=100)
#plt.xlim(0,7000)#normal distribution, replace with mean
#df['GrossMthSalary']=df['GrossMthSalary'].fillna(df['GrossMthSalary'].mean())
#df['GrossMthSalary'].isnull().any()
#
##BirthYear, 1stPolYear, GeoLivArea, EduDegree
#df.dropna(subset=['BirthYear'], inplace=True)
#
#df.dropna(subset=['EduDegree'], inplace=True)

################################TRANSFORMING AND CLEANING DATA############################################################

#Drop CustId
df.drop(['CustId'], axis=1, inplace=True)

#EduDegree is an object. Convert to ordinal.

df.describe()
df.info()

#degree_dummies=pd.get_dummies(df['EduDegree'], drop_first=False)
#df.drop(['EduDegree'], axis=1, inplace=True)
#df=pd.concat([df, degree_dummies], axis=1)

ord_edu=df['EduDegree'].str.split(' - ', 1, expand=True)
df['EduDegree']=ord_edu[0]

######################DETECTING OUTLIERS########################

####heat map 
sns.set(rc={'figure.figsize':(20,20)})
sns.heatmap(df.corr(), annot=True)


##############Z Score

from scipy import stats
z = np.abs(stats.zscore(df))
print(z)



threshold = 3
print(np.where(z > 3))#returns 1 array of rows and 2 array of columns of outliers

##########Interquartile range IQR

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

print(df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))
#returns trues and falses

#Filtering out outliers detected by Z Score

df_oZ = df[(z < 3).all(axis=1)]
df.shape
df_oZ.shape#Z score filters out 1101 outliers.its 10% of dataset

#Filtering out outliers detected by IQR

df_oIQR = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df_oIQR.shape#IQR filters out 2816 outliers. its 27%

df.head()

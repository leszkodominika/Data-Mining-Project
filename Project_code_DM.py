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
#from sklearn.preprocessing import Imputer
#df = df.where(df!='')  # Replacing empty values with NaN
#df=df.values
#imputer = Imputer(missing_values=np.nan, strategy = 'median', axis = 0)
#imputer = imputer.fit(df[:,-2:-1])
#
#df[:,-2:-1] = imputer.transform(df[:,-2:-1])

df.info()

#Renaming columns
df.columns.values

coldict={'Customer Identity':'CustId', 'First Policy´s Year':'1stPolYear', 'Brithday Year':'BirthYear',
       'Educational Degree':'EduDegree', 'Gross Monthly Salary':'GrossMthSalary',
       'Geographic Living Area':'GeoLivArea', 'Has Children (Y=1)':'HasChild',
       'Customer Monetary Value':'CustMonetVal', 'Claims Rate':'ClaimRate', 'Premiums in LOB: Motor':'PremLOB',
       'Premiums in LOB: Household':'PremLOBHousehold', 'Premiums in LOB: Health':'PremLOBHealth',
       'Premiums in LOB:  Life':'PremLOBLife', 'Premiums in LOB: Work Compensations':'PremLOBWorkCompensation'}

df.rename(columns=coldict, inplace=True)

df.columns.values
#############################Handling null values################################################################

sns.heatmap(df.isnull())

df.isna().any()
df.isnull().sum(axis=0)

#PremLOBWorkCompensation nulls replace with median
df['PremLOBWorkCompensation'].hist(bins=40)
df['PremLOBWorkCompensation']=df['PremLOBWorkCompensation'].fillna(df['PremLOBWorkCompensation'].median())
df['PremLOBWorkCompensation'].isnull().any()

#PremLOBLife nulls replace with median
df['PremLOBLife'].hist(bins=40)
df['PremLOBLife']=df['PremLOBLife'].fillna(df['PremLOBLife'].median())
df['PremLOBLife'].isnull().any()

#PremLOBHealth nulls replace with median
df['PremLOBHealth'].hist(bins=130)
plt.xlim(0,1000)
df['PremLOBHealth']=df['PremLOBHealth'].fillna(df['PremLOBHealth'].median())
df['PremLOBHealth'].isnull().any()

#PremLOB nulls replace with mean
df['PremLOB'].hist(bins=130)
plt.xlim(0,1000)#normal distribution, replace with mean
df['PremLOB']=df['PremLOBHealth'].fillna(df['PremLOB'].mean())
df['PremLOB'].isnull().any()

#HasChild nulls replace with median
df['HasChild'].hist(bins=10)
df['HasChild']=df['HasChild'].fillna(df['HasChild'].median())
df['HasChild'].isnull().any()


#GrossMthSalary nulls replace with
df['GrossMthSalary'].hist(bins=100)
plt.xlim(0,7000)#normal distribution, replace with mean
df['GrossMthSalary']=df['GrossMthSalary'].fillna(df['GrossMthSalary'].mean())
df['GrossMthSalary'].isnull().any()

#BirthYear, 1stPolYear, GeoLivArea, EduDegree
df.dropna(subset=['BirthYear'], inplace=True)
df.dropna(subset=['1stPolYear'], inplace=True)
df.dropna(subset=['GeoLivArea'], inplace=True)
df.dropna(subset=['EduDegree'], inplace=True)

################################TRANSFORMING AND CLEANING DATA############################################################
##EduDegree is an object. Needs encoding

df.describe()
df.info()

degree_dummies=pd.get_dummies(df['EduDegree'], drop_first=False)
df.drop(['EduDegree'], axis=1, inplace=True)
df=pd.concat([df, degree_dummies], axis=1)

#Drop CustId
df.drop(['CustId'], axis=1, inplace=True)

df.info()#lost only 40 entries after data cleaning.

######################DETECTING OUTLIERS########################

df.columns.values

sns.boxplot(x=df['1stPolYear'])#outlier>2016


sns.boxplot(x=df['BirthYear'])#outlier<1926 means older than 90



sns.boxplot(x=df['GrossMthSalary'])#Very rich?



sns.boxplot(x=df['GeoLivArea'])#No outliers



sns.boxplot(x=df['HasChild'])#No outliers



sns.boxplot(x=df['CustMonetVal'])#negative customer monetary value?not sure#lf_in case you dont know what this means: contract signed 
#generates a value for the company (in the normalcase the insurance comapny makes money selling polices). Nervertheless each customer 
#behavies on an idividual basis - therfore the value of the policy(customer here) can be evaluated on its on bases on what has happend in the past
#and predictions of the future. You can see that customers who have a high claimsrate/claimsratio are usually valued negativ as the money earned by 
#the insurance company is less that what the insuracne comapny has to pay to the customer for his valid claims, hence i am pretty sure that CustMonetVal and 
#claimsratio  are negativly correlated# hahah -0.99 if thats no suprise ;)


sns.boxplot(x=df['ClaimRate'])#not sure #lf claims rate is defined by Premium/Cost of claims


sns.boxplot(x=df['PremLOB'])

sns.boxplot(x=df['PremLOBHousehold'])

sns.boxplot(x=df['PremLOBHealth'])


sns.boxplot(x=df['PremLOBLife'])


sns.boxplot(x=df['PremLOBWorkCompensation'])


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

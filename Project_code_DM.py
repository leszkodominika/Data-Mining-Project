# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 19:24:28 2018
@author: dominika.leszko
"""
### DATA MINING PROJECT #######################################################
###############################################################################

### 1. IMPORTING THE LIBRARIES#################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### 2. IMPORTING THE DATASET###################################################
df=pd.read_csv(r'C:\Users\dominika.leszko\Desktop\NOVA IMS\Data Mining\DM Project\A2Z Insurance.csv')
df = pd.DataFrame(df)

#Display all columns
pd.set_option('display.max_columns', None)

#Rename columns
coldict={'Customer Identity':'CustId', 'First Policy´s Year':'1stPolYear', 'Brithday Year':'BirthYear',
       'Educational Degree':'EduDegree', 'Gross Monthly Salary':'GrossMthSalary',
       'Geographic Living Area':'GeoLivArea', 'Has Children (Y=1)':'HasChild',
       'Customer Monetary Value':'CustMonetVal', 'Claims Rate':'ClaimRate', 'Premiums in LOB: Motor':'PremLOBMotor',
       'Premiums in LOB: Household':'PremLOBHousehold', 'Premiums in LOB: Health':'PremLOBHealth',
       'Premiums in LOB:  Life':'PremLOBLife', 'Premiums in LOB: Work Compensations':'PremLOBWorkCompensation'}
df.rename(columns=coldict, inplace=True)

#Drop CustId
df.drop(['CustId'], axis=1, inplace=True)


### 3. HANDLING OUTLIERS, EXTREME VALUES & SKEWNESS############################

df.shape#rows: 10296 

df['1stPolYear'].describe()
#Drop values >2016, as the database comes from 2016
df = df.drop(df[df['1stPolYear']>2016].index)
sns.kdeplot(df['1stPolYear']).set_title('1st Policy Year')

df['BirthYear'].describe()
#Drop values <1900
df=df.drop(df[df['BirthYear']<1900].index)
df['BirthYear'].hist(bins=50).set_title('Birth Year')

df['GrossMthSalary'].describe()
sns.boxplot(x=df['GrossMthSalary'])
#Drop Salary>30000
df=df.drop(df[df['GrossMthSalary']>30000].index)
df['GrossMthSalary'].hist(bins=50).set_title('Gross  Monthly Salary')

df['PremLOBMotor'].describe()
sns.boxplot(x=df['PremLOBMotor'])
#Drop PremLOBMotor>2000
df=df.drop(df[df['PremLOBMotor']>2000].index)
sns.kdeplot(df['PremLOBMotor']).set_title('Premiums in LOB: Motor')

sns.boxplot(x=df['PremLOBHealth'])
#Drop PremLOBHealth>5000
df=df.drop(df[df['PremLOBHealth']>5000].index)
sns.kdeplot(df['PremLOBHealth']).set_title('Premiums in LOB: Health')

# SKEWED!!!!!!!!!Dropped 90
df['PremLOBHousehold'].hist(bins=100).set_title('Premiums in LOB: Household')
plt.xlim(0,4000)
#Skewed distribution -> Perform log transformation
df['PremLOBHousehold']=np.log(df['PremLOBHousehold'] + 1 - min(df['PremLOBHousehold']))
#Applying 3 sigma rule for outliers
df=df[np.abs(df['PremLOBHousehold'] - df['PremLOBHousehold'].mean())<=3*df['PremLOBHousehold'].std()]
sns.boxplot(x=df['PremLOBHousehold'])


# SKEWED!!!!!!!!!Dropped 165
df['PremLOBLife'].hist().set_title('Premiums in LOB: Life')
#Skewed distribution -> Perform log transformation
df['PremLOBLife']=np.log(df['PremLOBLife'] + 1 - min(df['PremLOBLife']))
#Applying 3 sigma rule for outliers#
df=df[np.abs(df['PremLOBLife'] - df['PremLOBLife'].mean())<=3*df['PremLOBLife'].std()]
sns.boxplot(x=df['PremLOBLife'])

# SKEWED!!!!!!!!!Dropped 167
df['PremLOBWorkCompensation'].hist(bins=100).set_title('PremLOBWorkCompensation in LOB: Life')
#Skewed distribution -> Perform log transformation
df['PremLOBWorkCompensation']=np.log(df['PremLOBWorkCompensation'] + 1 - min(df['PremLOBWorkCompensation']))
#Applying 3 sigma rule for outliers
df=df[np.abs(df['PremLOBWorkCompensation'] - df['PremLOBWorkCompensation'].mean())<=3*df['PremLOBWorkCompensation'].std()]
sns.boxplot(x=df['PremLOBWorkCompensation'])

#rows: 9862 --> meaning 434 rows dropped (4.2%)

### 4. HANDLING MISSING VALUES ################################################

df = df.replace({'': np.nan})

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(2)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
missing_values_table(df)

#Replace Premium LOB null values with 0 --> null means not insured, didn't pay
df['PremLOBHealth']=df['PremLOBHealth'].fillna(0)
df['PremLOBMotor']=df['PremLOBMotor'].fillna(0)

#Drop null value in Geography because of poor quality of values in remaining columns
df=df.dropna(subset=['GeoLivArea'])

#1st Year-KNN (30 nulls - mean/median?)
plt.figure(figsize=(8,6))
df['1stPolYear'].hist()


#### Replacing missing data with Regression ###################################

# We need to do same the other way round: for Salary based on regression on Bday, is it correct? 
y_train = df['BirthYear']
y_test = y_train.loc[y_train.index.isin(list(y_train.index[(y_train >= 0)== False]))]
X_train = pd.DataFrame(df['GrossMthSalary'].loc[y_train.index.isin(list(y_train.index[(y_train >=0)== True ]))])
X_test  = pd.DataFrame(df['GrossMthSalary'].loc[y_train.index.isin(list(y_train.index[(y_train >= 0)== False]))])
y_train = y_train.dropna()


from sklearn.linear_model import LinearRegression#ERROR we r training on grossmthsalary with 33 null values
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred= regressor.predict(X_test)

X_train.isna().sum()

i=0
for index in y_test.index:
    df['BirthYear'][index] = y_pred[i]
    i+=1

#####Replacing missing data with K-Nearest Neighbors ###############################

X = df.drop(columns=['HasChild', 'EduDegree', 'GeoLivArea'])
y = df['HasChild']

y_train = y
y_test = y_train.loc[y_train.index.isin(list(y_train.index[(y_train >= -1)== False]))]
X_train = pd.DataFrame(X.loc[y_train.index.isin(list(y_train.index[(y_train >= -1)== True]))])
X_test = pd.DataFrame(X.loc[y_train.index.isin(list(y_train.index[(y_train >= -1)== False]))])
y_train = y_train.dropna()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

X.isna().sum()

y_pred = classifier.predict(X_test)

i=0
for index in y_test.index:
    df['HasChild'][index] = y_pred[i]
    i+=1
       
####Replacing missing data with Decision Tree   #####################################
    
X = df.drop(columns=['EduDegree'])
y = df['EduDegree']

y_train = y
y_test = y.loc[y.isin(list(y[y.isna()== True]))]
X_train = pd.DataFrame(X.loc[y.isin(list(y[y.isna()== False]))])
X_test  = pd.DataFrame(X.loc[y.isin(list(y[y.isna()== True ]))])
y_train = y.dropna()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder1 = LabelEncoder()
y_train = labelEncoder1.fit_transform(y_train)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

i=0
for index in y_test.index:
    df['EduDegree'][index] = y_pred[i]
    i+=1       


### 5. EXPLORATORY DATA ANALYSIS###############################################

sns.set(rc={'figure.figsize':(20,20)})
sns.heatmap(df.corr(), annot=True)

sns.pairplot(df)



### 6. FEATURE ENGINEERING AND SELECTION#######################################

#Convert EduDegree to dummies and drop one column to avoid dummy trap
edu_dummies=pd.get_dummies(df['EduDegree'], drop_first=True)
df.drop(['EduDegree'], axis=1, inplace=True)
df=pd.concat([df, edu_dummies], axis=1)


#DO DATA SCALING AND MAKE MEAN=1,STD=1 (Z SCORE). We need it scale cause its clustering

#
###############Z Score#same as min max and standardscaler?
#
#from scipy import stats
#z = np.abs(stats.zscore(df))
#print(z)
#
#threshold = 3
#print(np.where(z > 3))#returns 1 array of rows and 2 array of columns of outliers
#
###########Interquartile range IQR
#a = np.array(df['PremLOBHousehold'])
#a = np.sort(a)
#Q1 = df['PremLOBHousehold'].quantile(0.002)
#Q3 = df.quantile(0.75)
#IQR = Q3 - Q1
#print(IQR)
#
#print(df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))
##returns trues and falses
#
##Filtering out outliers detected by Z Score
#
#df_oZ = df[(z < 3).all(axis=1)]
#df.shape
#df_oZ.shape#Z score filters out 1101 outliers.its 10% of dataset
#
##Filtering out outliers detected by IQR
#
#df_oIQR = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
#df_oIQR.shape#IQR filters out 2816 outliers. its 27%
#
#df.head()
#
###
#
##from sklearn.preprocessing import Imputer#DONE AT THE TOP
##df = df.where(df!='')  # Replacing empty values with NaN
#df=df.values
#imputer = Imputer(missing_values=np.nan, strategy = 'median', axis = 0)
#imputer = imputer.fit(df[:,-2:-1])
#
#df[:,-2:-1] = imputer.transform(df[:,-2:-1])


### 7. MODELLING ##############################################################

from sklearn.cluster import KMeans

df=pd.read_csv(r'C:\Users\dominika.leszko\Desktop\NOVA IMS\Data Mining\DM Project\A2Z Insurance.csv')

#from sklearn.preprocessing import Imputer
df = pd.DataFrame(df)

df = df.replace({'': np.nan})

#Renaming columns for easier analysis
df.columns.values

coldict={'Customer Identity':'CustId', 'First Policy´s Year':'1stPolYear', 'Brithday Year':'BirthYear',
       'Educational Degree':'EduDegree', 'Gross Monthly Salary':'GrossMthSalary',
       'Geographic Living Area':'GeoLivArea', 'Has Children (Y=1)':'HasChild',
       'Customer Monetary Value':'CustMonetVal', 'Claims Rate':'ClaimRate', 'Premiums in LOB: Motor':'PremLOBMotor',
       'Premiums in LOB: Household':'PremLOBHousehold', 'Premiums in LOB: Health':'PremLOBHealth',
       'Premiums in LOB:  Life':'PremLOBLife', 'Premiums in LOB: Work Compensations':'PremLOBWorkCompensation'}

df.rename(columns=coldict, inplace=True)
df.dropna(inplace=True)
df.columns


['CustId', '1stPolYear', 'BirthYear', 'EduDegree', 'GrossMthSalary',
       'GeoLivArea', 'HasChild', 'CustMonetVal', 'ClaimRate', 'PremLOBMotor',
       'PremLOBHousehold', 'PremLOBHealth', 'PremLOBLife',
       'PremLOBWorkCompensation']

#Create data frame with only numerical values. Dropped CustId, Edu Degree, GeoLivArea, HasChild
df_numerical=df[['1stPolYear', 'BirthYear', 'GrossMthSalary', 'CustMonetVal', 'ClaimRate', 'PremLOBMotor','PremLOBHousehold', 'PremLOBHealth', 'PremLOBLife','PremLOBWorkCompensation']]

#Scale data

from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
df_numerical = pca.fit_transform(df_numerical)
#ex_var = pca.explained_variance_ratio_

pca.fit(df_numerical)

red_pca = np.dot(pca.transform(df_numerical)[:,:5],pca.components_[:5,:])
red_pca += np.mean(df_numerical, axis=0)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_numerical = scaler.fit_transform(df_numerical)


#Plotting elbow graph to determine K
wcss=[]
for i in range(1,16):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_numerical)
    wcss.append(kmeans.inertia_)
    #inertia = Sum of squared distances of samples to their closest cluster center.

plt.plot(range(1,16), wcss, color='green')
plt.title('Elbow Graph')
plt.xlabel('Number of clusters K')
plt.ylabel('WCSS')

#Training the model
kmeans=KMeans(n_clusters=9)
df_numerical['kmean']=kmeans.fit_predict(df_numerical)

df_numerical['kmean'].hist()
l1 = [df['EduDegree'],df['HasChild'],df['GeoLivArea']]
#df['hc_split'] = pd.concat(, axis=1 )


df['hc_split'] = df['EduDegree'].map(str) + df['HasChild'].map(str) + df['GeoLivArea'].map(str)
sns.countplot('hc_split', data=df)

df.drop(['EduDegree', 'HasChild','GeoLivArea'],inplace=True, axis=1)
df.drop(['CustId'],inplace=True, axis=1)

df_hc = df.drop(['hc_split'], axis=1)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df.columns.values
df[['1stPolYear', 'BirthYear', 'GrossMthSalary', 'CustMonetVal',
       'ClaimRate', 'PremLOBMotor', 'PremLOBHousehold', 'PremLOBHealth',
       'PremLOBLife', 'PremLOBWorkCompensation']] = scaler.fit_transform(df['1stPolYear', 'BirthYear', 'GrossMthSalary', 'CustMonetVal',
       'ClaimRate', 'PremLOBMotor', 'PremLOBHousehold', 'PremLOBHealth',
       'PremLOBLife', 'PremLOBWorkCompensation'])

X=dataset.iloc[:,[3,4]].values
labels = df['hc_split'].values
labels =list(labels)

df_hc = df.iloc[:,:].values
df_hc.set_index('hc_split')
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(df_hc, method='ward'), labels = labels)#ward minimizuje variance within cluster scatter
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

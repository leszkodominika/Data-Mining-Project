### DATA MINING PROJECT #######################################################
###############################################################################

### 1. IMPORTING THE LIBRARIES#################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### 2. IMPORTING THE DATASET###################################################
df=pd.read_csv('A2Z Insurance.csv')
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

#Drop CustMonetVal< -2000
df['CustMonetVal'].describe()
sns.boxplot(x=df['CustMonetVal'])
df=df.drop(df[df['CustMonetVal']<-2000].index) 

#Drop ClaimRate > 3
df['ClaimRate'].describe()
sns.boxplot(x=df['ClaimRate'])
df=df.drop(df[df['ClaimRate']>3].index) 

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
df=df[np.abs(df['PremLOBHousehold'] - df['PremLOBHousehold'].mean())<4*df['PremLOBHousehold'].std()]
sns.boxplot(x=df['PremLOBHousehold'])


# SKEWED!!!!!!!!!Dropped 165
df['PremLOBLife'].hist().set_title('Premiums in LOB: Life')
#Skewed distribution -> Perform log transformation
df['PremLOBLife']=np.log(df['PremLOBLife'] + 1 - min(df['PremLOBLife']))
#Applying 3 sigma rule for outliers#
df=df[np.abs(df['PremLOBLife'] - df['PremLOBLife'].mean())<=4*df['PremLOBLife'].std()]
sns.boxplot(x=df['PremLOBLife'])

# SKEWED!!!!!!!!!Dropped 167
df['PremLOBWorkCompensation'].hist(bins=100).set_title('PremLOBWorkCompensation in LOB: Life')
#Skewed distribution -> Perform log transformation
df['PremLOBWorkCompensation']=np.log(df['PremLOBWorkCompensation'] + 1 - min(df['PremLOBWorkCompensation']))
#Applying 3 sigma rule for outliers
df=df[np.abs(df['PremLOBWorkCompensation'] - df['PremLOBWorkCompensation'].mean())<=4*df['PremLOBWorkCompensation'].std()]
sns.boxplot(x=df['PremLOBWorkCompensation'])

#rows: 9862 --> meaning 434 rows dropped (4.2%)
#3.5 sigma: 9927 rows --> 369 rows dropped (3.6%)
#4 sigma:10022 rows --> 274 rows dropped (2.6%)

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

#1st Year-KNN (28 nulls)
plt.figure(figsize=(8,6))
df['1stPolYear'].hist()
#1963 values with 1stPolYear < BirthYear!!!!!!!!! 20% Let's drop BirthYear (after regression on BirthYear for GrossMthSalary)
df[df['1stPolYear']<df['BirthYear']][['1stPolYear','BirthYear']]

"""
DOMINIKA: Drop BirthYear column as 1stYearPolicy is company data so more accurate. Additionally BirthYear is highly correlated 
with GrossMthSalary
"""


#### Replacing missing data with Regression ###################################

##DOMINIKA
y_train = df[df['BirthYear'].isna()==False]['GrossMthSalary']
y_test = y_train.loc[y_train.index.isin(list(y_train.index[(y_train >= 0)== False]))]
X_train = pd.DataFrame(df[df['BirthYear'].isna()==False]['BirthYear'].loc[y_train.index.isin(list(y_train.index[(y_train >=0)== True]))])
X_test  = pd.DataFrame(df['BirthYear'].loc[y_train.index.isin(list(y_train.index[(y_train >= 0)== False]))])
y_train = pd.DataFrame(y_train.dropna())

from sklearn.linear_model import LinearRegression#ERROR we r training on grossmthsalary with 33 null values
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred= regressor.predict(X_test)

i=0
for index in y_test.index:
    df['GrossMthSalary'][index] = y_pred[i]
    i+=1

#from scipy.stats import linregress
#X_train = np.array(X_train)
#y_train = np.array(y_train)
#slope, intercept, r_value, p_value, std_err = linregress(X_train[:,0], y_train[:,0])

    

#####Replacing missing data  on [HasChild] with K-Nearest Neighbors ###############################

#DOMINIKA: Once we handle previously 1stPolYear then this will work properly
# WE dont have to drop column with geoLivArea for decission tree but we dont want to multiply sets
X = df.drop(columns=['HasChild', 'EduDegree', 'GeoLivArea',  '1stPolYear', 'BirthYear'])
y = df['HasChild']

y_train = y
y_test = y_train.loc[y_train.index.isin(list(y_train.index[(y_train >= -1)== False]))]
X_train = pd.DataFrame(X.loc[y_train.index.isin(list(y_train.index[(y_train >= -1)== True]))])
X_test = pd.DataFrame(X.loc[y_train.index.isin(list(y_train.index[(y_train >= -1)== False]))])
y_train = y_train.dropna()


from sklearn.cross_validation import train_test_split
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree.fit(X_train_train, y_train_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train_train, y_train_train)

y_pred_1 = tree.predict(X_train_test)
y_pred_2 = knn.predict(X_train_test)

#dif_tree, dif_knn = 0, 0
#for i in range(len(y_train_test)):
dif_tree=np.abs(y_pred_1-y_train_test)
dif_knn=np.abs(y_pred_2-y_train_test)
print(np.sum(dif_tree)/len(y_train_test))
print(np.sum(dif_knn)/len(y_train_test))

i=0
for index in y_test.index:
    df['HasChild'][index] = y_pred_2[i]
    i+=1
# We choose knn becuse avg mistake is smaller

####Replacing missing data [EduDegree] with Decision Tree   #####################################
#DOMINIKA: Same here, Once we handle previously 1stPolYear then this will work properly
X = df.drop(columns=['EduDegree', '1stPolYear', 'GeoLivArea','BirthYear'])
y = df['EduDegree']

#for index in y.index:
#    if type(y[index]) == float:
#        continue
#    a = y[index].split(' ')[0]
#    y[index] = int(a)

y_train = y
y_test = y.loc[y.isin(list(y[y.isna()== True]))]
X_train = pd.DataFrame(X.loc[y.isin(list(y[y.isna()== False]))])
X_test  = pd.DataFrame(X.loc[y.isin(list(y[y.isna()== True ]))])
y_train = y.dropna()

from sklearn.cross_validation import train_test_split
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree.fit(X_train_train, y_train_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train_train, y_train_train)

y_pred_1 = tree.predict(X_train_test)
y_pred_2 = knn.predict(X_train_test)

from sklearn.metrics import confusion_matrix
cm_tree = confusion_matrix(y_train_test, y_pred_1)
cm_knn = confusion_matrix(y_train_test, y_pred_2)

print(np.sum(cm_tree[i][i] for i in range(len(cm_tree))))
print(np.sum(cm_knn[i][i] for i in range(len(cm_knn))))

# We choose knn becuse avg mistake is smaller

i=0
for index in y_test.index:
    df['EduDegree'][index] = y_pred_2[i]
    i+=1   
    

##### Replacing data in 1stYearPolicy

    
X = df.drop(columns=['1stPolYear', 'EduDegree', 'GeoLivArea','BirthYear'])
y = df['1stPolYear']
   
y_train = y
y_test = y.loc[y.isin(list(y[y.isna()== True]))]
X_train = pd.DataFrame(X.loc[y.isin(list(y[y.isna()== False]))])
X_test  = pd.DataFrame(X.loc[y.isin(list(y[y.isna()== True ]))])
y_train = y.dropna()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.cross_validation import train_test_split
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)


from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(criterion="mse", random_state = 0)
tree.fit(X_train_train, y_train_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train_train, y_train_train)

y_pred_1 = tree.predict(X_train_test)
y_pred_2 = knn.predict(X_train_test)

#dif_tree, dif_knn = 0, 0
#for i in range(len(y_train_test)):
dif_tree=np.abs(y_pred_1-y_train_test)
dif_knn=np.abs(y_pred_2-y_train_test)
print(np.sum(dif_tree)/len(y_train_test))
print(np.sum(dif_knn)/len(y_train_test))
# We choose tree becuse avg mistake is smaller

i=0
for index in y_test.index:
    df['1stPolYear'][index] = y_pred_1[i]
    i+=1

# we can try to select better column because avg error is 7-8 year which is a lot
    
### 5. EXPLORATORY DATA ANALYSIS###############################################

sns.set(rc={'figure.figsize':(20,20)})
sns.heatmap(df.corr(), annot=True)

###sns.pairplot(df) this gives an error 

## 6. FEATURE ENGINEERING AND SELECTION#######################################
# compbine 3+4 for keepting cat 
#Convert EduDegree to dummies and drop one column to avoid dummy trap
edu_dummies=pd.get_dummies(df['EduDegree'], drop_first=False)
df.drop(['EduDegree'], axis=1, inplace=True)
df=pd.concat([df, edu_dummies], axis=1)

# Merging two categories (PhD only 1% of dataset)
df['LowerEdu'] = df['1 - Basic'] + df['2 - High School']
df['HigherEdu'] = df['3 - BSc/MSc'] + df['4 - PhD']
df.drop(['3 - BSc/MSc','4 - PhD'], axis=1, inplace=True)

#Convert GeoLivArea to dummies and drop one column to avoid dummy trap no significant for clustering neigther on encoded nor catigorical level 
#edu_dummies=pd.get_dummies(df['GeoLivArea'], drop_first=False)
#df.drop(['GeoLivArea'], axis=1, inplace=True)
#df=pd.concat([df, edu_dummies], axis=1)
#df.rename(columns={1.0:'Geo_1', 2.0:'Geo_2', 3.0:'Geo_3', 4.0:'Geo_4'}, inplace=True)
#

df['HigherEdu'].value_counts()
df.reset_index(inplace=True, drop = True)
df.columns.values

# Feature engineering

#df['BirthYear_pred'] 

#y = [1,2]
#
#
#xx = pd.DataFrame()
#xx['BirthYear_pred'] =y
#
#pd.Series(y, dtype = int)
#yy=y.T
#
#y = ((df['GrossMthSalary'].tolist() - regressor.intercept_)/regressor.coef_)
#y = df['GrossMthSalary'].tolist()
#'''THIS IS STILL NOT WORKING FOR ME'''
#
#
#
#
#
#df['BirthYear_pred'] = 0
#for index in df.index:
#    df.loc[:,'BirthYear_pred'][index] = (df.loc[:,'GrossMthSalary'][index] - regressor.intercept_)/regressor.coef_
#df.loc['BirthYear_pred'] = df.loc['BirthYear_pred'].astype(int)
#
##Drop BirthYear because of high correlation with 1stPolYear 
#df.drop(['BirthYear'], axis=1, inplace=True)


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
########################








###### 7 Modelling ######


###7.1 potential data #####

##### getting the potential as the value not spent on insurance ####
#df['potential%']=(df['GrossMthSalary']-df['PremLOBMotor']-df['PremLOBHousehold']-df[ 'PremLOBHealth']-df['PremLOBLife']-df[ 'PremLOBWorkCompensation'])/df['GrossMthSalary']
#df['potential_abs']=(df['GrossMthSalary']-df['PremLOBMotor']-df['PremLOBHousehold']-df[ 'PremLOBHealth']-df['PremLOBLife']-df[ 'PremLOBWorkCompensation'])
##what should we do if someone spends more than they actually earn?
#
##### performing pca on all the LOB Premiums as they are correlated####
#
#new_df=df_comp[['PremLOBMotor','PremLOBHousehold', 'PremLOBHealth','PremLOBLife', 'PremLOBWorkCompensation']]
##
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 2)
#new_df_pca = pca.fit_transform(new_df)
#explained_variance = pca.explained_variance_ratio_
#
#df_pca=pd.DataFrame(new_df_pca)
#df_pca.reset_index(drop=True, inplace=True)
#df_pca['pca1'], df_pca['pca2']= new_df_pca.iloc[:,0],new_df_pca.iloc[:,1]


#'Premiums in LOB: Motor':'PremLOBMotor',
#       'Premiums in LOB: Household':'PremLOBHousehold', 'Premiums in LOB: Health':'PremLOBHealth',
#       'Premiums in LOB:  Life':'PremLOBLife', 'Premiums in LOB: Work Compensations':'PremLOBWorkCompensation'}
#    


############plot the distribution in 3d
#
#from mpl_toolkits.mplot3d import Axes3D  
#import matplotlib.pyplot as plt
#import numpy as np
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#xs = df_comp.iloc[:,0]
#ys = df_comp.iloc[:,1]
#zs = df_comp.iloc[:,2]
#ax.scatter(xs, ys, zs, alpha = .07, color ='r')
#
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#
#plt.show()
#
###########end plot 
#
###### K means not used cause correlTED
#from sklearn.cluster import KMeans
#
##use elbow to estimate clusters
#wcss=[]
#for i in range(1,16):
#    kmeans=KMeans(n_clusters=i)
#    kmeans.fit(df_comp)
#    wcss.append(kmeans.inertia_)
#    #inertia = Sum of squared distances of samples to their closest cluster center.
#
#plt.plot(range(1,16), wcss, color='green')
#plt.title('Elbow Graph')
#plt.xlabel('Number of clusters K')
#plt.ylabel('WCSS')
#
#
#######no clear elbow -> propably the clusters are not very good

#####DBSCAN clustering
#
#from sklearn.cluster import DBSCAN, estimate_bandwidth
#from sklearn import metrics
#
## The following bandwidth can be automatically detected using
#my_bandwidth = estimate_bandwidth(df_comp, quantile=0.008)
#
#db = DBSCAN(eps=my_bandwidth, min_samples=100).fit(df_comp)
#
#labels = db.labels_
#
#df_comp['dbscan_label']= db.labels_
#
## Number of clusters in labels, ignoring noise if present.
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#
#unique_clusters, counts_clusters = np.unique(db.labels_, return_counts = True)
#print(np.asarray((unique_clusters, counts_clusters)))
#####- > one strong cluster with 0.008
#
#####plot dbscan results
###############something is wrong here
#my_color=[]
#my_marker=[]
#for i in range(df_comp.shape[0]):
#    if df_comp['dbscan_label'][i] == -1:
#        my_color.append('r')
#        my_marker.append('+')
#    elif df_comp['dbscan_label'][i] == 0:
#        my_color.append('b')
#        my_marker.append('o')
#    elif df_comp['dbscan_label'][i] == 1:
#        my_color.append('g')
#        my_marker.append('*')
#        
#        
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#for i in range(df_comp.shape[0]):
#    ax.scatter(df_comp.iloc[i,0], df_comp.iloc[i,1], df_comp.iloc[i,2], c=my_color[i], marker=my_marker[i])
#    
#ax.set_xlabel('pca^ 1')
#ax.set_ylabel('potential')
#ax.set_zlabel('CustMonet')
####end dbscan 
#
df_comp = df [['PremLOBMotor',
               'PremLOBHousehold', 
               'PremLOBHealth', 
               'PremLOBLife', 
               'PremLOBWorkCompensation']]
#sns.heatmap(df_comp.corr(), annot=True)

# using PCA1 potenntial% and CustMonetVal  f


###scaling the potentail data

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_comp = scaler.fit_transform(df_comp)

df_comp = pd.DataFrame(df_comp)
df_comp.columns = ['PremLOBMotor', 
                   'PremLOBHousehold', 
                   'PremLOBHealth', 
                   'PremLOBLife', 
                   'PremLOBWorkCompensation']

#
######MEAN SHIFT
#import numpy as np
#from sklearn.cluster import MeanShift, estimate_bandwidth
#my_bandwidth = estimate_bandwidth(df_pca,
#                               quantile=0.03, n_jobs = -1)
#
#ms = MeanShift(bandwidth=my_bandwidth, bin_seeding=True)
#
#ms.fit(df_pca)
#labels = ms.labels_
#cluster_centers = ms.cluster_centers_
#
#labels_unique = np.unique(labels)
#n_clusters_ = len(labels_unique)
#
#len(labels_unique)
#
#df_pca['mshift_label']=ms.labels_
#df_pca['mshift_label'].value_counts()


# Expectation-Maximization

from sklearn import mixture
gmm = mixture.GaussianMixture(n_components=2,
                              covariance_type='full',
                              init_params='kmeans',
                              max_iter=1000,
                              n_init=10)

em_set=df_comp[['PremLOBMotor', 
                'PremLOBHousehold', 
                'PremLOBHealth', 
                'PremLOBLife', 
                'PremLOBWorkCompensation']]
gmm.fit(em_set)
EM_labels_ = gmm.predict(em_set)
#Elbow
em_set_score = gmm.score(em_set)
#Individual
em_set_samp = gmm.score_samples(em_set)
#Individual
em_set_prob = gmm.predict_proba(em_set)


y=1
for col in range(em_set_prob.shape[1]):
    for row in range(em_set_prob.shape[0]):
        if em_set_prob[row][col] > 0.8:
            em_set_prob[row][col]=y
        else:
            em_set_prob[row][col]=0
    y+=1
    

df_comp['EM_label']=em_set_prob[:,0]+em_set_prob[:,1]#+em_set_prob[:,2]+em_set_prob[:,3]#+em_set_prob[:,4]#+em_set_prob[:,5]
df_comp['EM_label'].value_counts()
#
#my_color=[]
#my_marker=[]
#for i in range(df_comp.shape[0]):
#    if df_comp['EM_label'][i] == 0:
#        my_color.append('r')
#        my_marker.append('+')
#    elif df_comp['EM_label'][i] == 1:
#        my_color.append('b')
#        my_marker.append('o')
#    elif df_comp['EM_label'][i] == 2:
#        my_color.append('g')
#        my_marker.append('*')
#    elif df_comp['EM_label'][i] == 3:
#        my_color.append('c')
#        my_marker.append('x')
#        
#        
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#for i in range(df_comp.shape[0]):
#    ax.scatter(df_comp.iloc[i,0], df_comp.iloc[i,1], df_comp.iloc[i,2], c=my_color[i], marker=my_marker[i])
#    
#ax.set_xlabel('pca^1')
#ax.set_ylabel('potential')
#ax.set_zlabel('CustMonet')

df_comp=df_comp.drop(['EM_label'], axis = 1)

df_comp = pd.DataFrame(scaler.inverse_transform(X=df_comp), columns=['PremLOBMotor',
                       'PremLOBHousehold',
                       'PremLOBHealth', 
                       'PremLOBLife',
                       'PremLOBWorkCompensation'])


#back to original data
df_comp['PremLOBHousehold']=np.exp(df_comp['PremLOBHousehold'])-1-75### this is the min in the original data 
df_comp['PremLOBLife']=np.exp(df_comp['PremLOBLife']) -1-7#### this is the min in the original data 
df_comp['PremLOBWorkCompensation']=np.exp(df_comp['PremLOBWorkCompensation'])- 1 -12 #### this is the min in the original data 


df_comp['EM_label']=em_set_prob[:,0]+em_set_prob[:,1]#+em_set_prob[:,2]+em_set_prob[:,3]#+em_set_prob[:,4]#+em_set_prob[:,5]

###plotting the three variabl distributions in EM clusters ####['PremLOBMotor', 'PremLOBHousehold', 'PremLOBHealth', 'PremLOBLife', 'PremLOBWorkCompensation']
# Sort the dataframe by target (potnetial)
target_0 = df_comp.loc[df_comp['EM_label'] == 0]
target_1 = df_comp.loc[df_comp['EM_label'] == 1]
target_2 = df_comp.loc[df_comp['EM_label'] == 2]
target_3 = df_comp.loc[df_comp['EM_label'] == 3]
target_4 = df_comp.loc[df_comp['EM_label'] == 4]
#potentail

#2 clusters
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(flatui)
df=df.drop(df[df['GrossMthSalary']<30000].index)
sns.set_style("white")
sns.distplot(target_1['PremLOBMotor'], hist=False, kde=True, rug=False, kde_kws={"lw": 5})#some what more
sns.distplot(target_2['PremLOBMotor'], hist=False, kde=True, rug=False, kde_kws={"lw": 5}).set_title('Distribution of LOB Motor for 2 Clusters', fontsize=20, weight='bold')
sns.despine()
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('')
plt.ylabel('')
plt.legend(size=18)
plt.show()

#3 clusters
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(flatui)
df=df.drop(df[df['GrossMthSalary']<30000].index)
sns.set_style("white")
sns.distplot(target_1['PremLOBMotor'], hist=False, kde=True, rug=False, kde_kws={"lw": 5})#some what more
sns.distplot(target_2['PremLOBMotor'], hist=False, kde=True, rug=False, kde_kws={"lw": 5})
sns.distplot(target_3['PremLOBMotor'], hist=False, kde=True, rug=False, kde_kws={"lw": 5})
sns.distplot(target_4['PremLOBMotor'], hist=False, kde=True, rug=False, kde_kws={"lw": 5}).set_title('Distribution of LOB Motor for 4 Clusters', fontsize=20, weight='bold')
sns.despine()
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('')
plt.ylabel('')
plt.legend(size=18)
plt.show()


#3 clusters
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(flatui)
df=df.drop(df[df['GrossMthSalary']<30000].index)
sns.set_style("white")
sns.distplot(target_1['PremLOBMotor'], hist=False, kde=True, rug=False, kde_kws={"lw": 5})#some what more
sns.distplot(target_2['PremLOBMotor'], hist=False, kde=True, rug=False, kde_kws={"lw": 5})
sns.distplot(target_3['PremLOBMotor'], hist=False, kde=True, rug=False, kde_kws={"lw": 5}).set_title('Distribution of LOB Motor for 3 Clusters', fontsize=20, weight='bold')
sns.despine()
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('')
plt.ylabel('')
plt.legend(size=18)
plt.show()


#sns.distplot(target_0['PremLOBMotor'], hist=False, kde=True, rug=False)
sns.distplot(target_1['PremLOBMotor'], hist=False, kde=True, rug=False)#some what more
sns.distplot(target_2['PremLOBMotor'], hist=False, kde=True, rug=False)#get rebates
sns.distplot(target_3['PremLOBMotor'], hist=False, kde=True, rug=False)#spend very much
sns.distplot(target_4['PremLOBMotor'], hist=False, kde=True, rug=False, color='yellow')#same as blue almost but no rebase





#pca1
#sns.distplot(target_0['PremLOBHousehold'], hist=False, kde=True, rug=False)
sns.distplot(target_1['PremLOBHousehold'], hist=False, kde=True, rug=False, color='green')#not much spent
sns.distplot(target_2['PremLOBHousehold'], hist=False, kde=True, rug=False, color='blue')#get rebates to very high
sns.distplot(target_3['PremLOBHousehold'], hist=False, kde=True,  rug=False, color='red')#close to nothing spent
sns.distplot(target_4['PremLOBHousehold'], hist=False, kde=True,  rug=False, color='yellow')#highest median



#2 clusters
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(flatui)
df=df.drop(df[df['GrossMthSalary']<30000].index)
sns.set_style("white")
sns.distplot(target_1['PremLOBHousehold'], hist=False, kde=True, rug=False, kde_kws={"lw": 5})#some what more
sns.distplot(target_2['PremLOBHousehold'], hist=False, kde=True, rug=False, kde_kws={"lw": 5}).set_title('Distribution of LOB Household for 2 Clusters', fontsize=20, weight='bold')
sns.despine()
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('')
plt.ylabel('')
plt.legend(size=18)
plt.show()


###maybe health is not important for clusterong?? no it is very important for clustering !!!!
#CustMonet
#sns.distplot(target_0['PremLOBHealth'], hist=False, kde=True, rug=False)
sns.distplot(target_1['PremLOBHealth'], hist=False, kde=True, rug=False, color='green')#2
sns.distplot(target_2['PremLOBHealth'], hist=False, kde=True, rug=False, color='blue')#1
sns.distplot(target_3['PremLOBHealth'], hist=False, kde=True, rug=False, color='red')#3
sns.distplot(target_4['PremLOBHealth'], hist=False, kde=True, rug=False, color='yellow')#3


#2 clusters
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(flatui)
df=df.drop(df[df['GrossMthSalary']<30000].index)
sns.set_style("white")
sns.distplot(target_1['PremLOBHealth'], hist=False, kde=True, rug=False, kde_kws={"lw": 5})#some what more
sns.distplot(target_2['PremLOBHealth'], hist=False, kde=True, rug=False, kde_kws={"lw": 5}).set_title('Distribution of LOB Health for 2 Clusters', fontsize=20, weight='bold')
sns.despine()
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('')
plt.ylabel('')
plt.legend(size=18)
plt.show()



#CustMonet
#sns.distplot(target_0['PremLOBLife'], hist=False, kde=True, rug=False)
sns.distplot(target_1['PremLOBLife'], hist=False, kde=True, rug=False, color='green')#3
sns.distplot(target_2['PremLOBLife'], hist=False, kde=True, rug=False, color='blue')#1
sns.distplot(target_3['PremLOBLife'], hist=False, kde=True, rug=False, color='red')#2
sns.distplot(target_4['PremLOBLife'], hist=False, kde=True, rug=False, color='yellow')#2


#2 clusters
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(flatui)
df=df.drop(df[df['GrossMthSalary']<30000].index)
sns.set_style("white")
sns.distplot(target_1['PremLOBLife'], hist=False, kde=True, rug=False, kde_kws={"lw": 5})#some what more
sns.distplot(target_2['PremLOBLife'], hist=False, kde=True, rug=False, kde_kws={"lw": 5}).set_title('Distribution of LOB Life for 2 Clusters', fontsize=20, weight='bold')
sns.despine()
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('')
plt.ylabel('')
plt.legend(size=18)
plt.show()

#CustMonet
#sns.distplot(target_0['PremLOBWorkCompensation'], hist=False, kde=True, rug=False)
sns.distplot(target_1['PremLOBWorkCompensation'], hist=False, kde=True, rug=False, color='green')#3
sns.distplot(target_2['PremLOBWorkCompensation'], hist=False, kde=True, rug=False, color='blue')#1
sns.distplot(target_3['PremLOBWorkCompensation'], hist=False, kde=True, rug=False, color='red')#2
sns.distplot(target_4['PremLOBWorkCompensation'], hist=False, kde=True, rug=False, color='yellow')#2

#2 clusters
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(flatui)
df=df.drop(df[df['GrossMthSalary']<30000].index)
sns.set_style("white")
sns.distplot(target_1['PremLOBWorkCompensation'], hist=False, kde=True, rug=False, kde_kws={"lw": 5})#some what more
sns.distplot(target_2['PremLOBWorkCompensation'], hist=False, kde=True, rug=False, kde_kws={"lw": 5}).set_title('Distribution of LOB Workcompensation for 2 Clusters', fontsize=20, weight='bold')
sns.despine()
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('')
plt.ylabel('')
plt.legend(size=18)
plt.show()


mean1_motor = target_1['PremLOBMotor'].mean()
mean2_motor = target_2['PremLOBMotor'].mean()
mean1_house = target_1['PremLOBHousehold'].mean()
mean2_house = target_2['PremLOBHousehold'].mean()
mean1_health = target_1['PremLOBHealth'].mean()
mean2_health = target_2['PremLOBHealth'].mean()
mean1_life = target_1['PremLOBLife'].mean()
mean2_life = target_2['PremLOBLife'].mean()
mean1_work = target_1['PremLOBWorkCompensation'].mean()
mean2_work = target_2['PremLOBWorkCompensation'].mean()

potential_1 = (mean2_motor-mean1_motor) 
potential_2 = (mean1_house-mean2_house)+(mean1_health-mean2_health)+(mean1_life-mean2_life)+(mean1_work-mean2_work)

### ms on cust data 

#cust_cont = df[[]]

from kmodes.kprototypes import KPrototypes

#df.columns

#sns.distplot(df['ClaimRate'], hist=False, kde=True, rug=False, color='red')#2


cust_cont = df[[ 'HasChild', 'HigherEdu']]
to_scale= df[['GrossMthSalary', 'ClaimRate']]

to_scale.corr()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

xxx = pd.DataFrame(scaler.fit_transform(to_scale), columns=['GrossMthSalary', 'ClaimRate'])
cust_cont['GrossMthSalary'],cust_cont['ClaimRate']=xxx['GrossMthSalary'] ,xxx['ClaimRate']


missing_values_table(cust_cont)

cust_cont = cust_cont.values


kproto = KPrototypes(n_clusters=3, init='Huang', verbose=2)
clusters = kproto.fit_predict(cust_cont, categorical=[0,1])
clusters = pd.DataFrame(clusters, columns=['cluster'])

cust_cont=pd.DataFrame(cust_cont, columns=['HasChild', 'HigherEdu','GrossMthSalary', 'ClaimRate'])
### do binary for education 
scaleed_back = pd.DataFrame(scaler.inverse_transform(X=cust_cont[['GrossMthSalary', 'ClaimRate']]), columns=['GrossMthSalary', 'ClaimRate'])
## to do get original data back!! 
cust_cont['GrossMthSalary'],cust_cont['ClaimRate'] = scaleed_back['GrossMthSalary'],scaleed_back['ClaimRate']

cust_data = pd.concat([cust_cont,clusters], axis =1)

clust_centersnom = kproto.cluster_centroids_[0]

clust_centersnom = scaler.inverse_transform(clust_centersnom)


### k modes on cust data 



#import numpy as np
#from kmodes.kmodes import KModes
#
#my_modes = df[['GeoLivArea','EduDegree', 'HasChild']].astype(str)
#
#km2 = KModes(n_clusters=2, 
#            init='Huang', 
#            n_init=5, 
#            verbose=1)
#
#clusters = km2.fit_predict(my_modes)
#
#km3 = KModes(n_clusters=3, 
#            init='Huang', 
#            n_init=5, 
#            verbose=1)
#
#clusters = km3.fit_predict(my_modes)
#
#km4 = KModes(n_clusters=4, 
#            init='Huang', 
#            n_init=5, 
#            verbose=1)
#
#clusters = km4.fit_predict(my_modes)
#
#km5 = KModes(n_clusters=5, 
#            init='Huang', 
#            n_init=5, 
#            verbose=1)
#
#clusters = km5.fit_predict(my_modes)
#
## Print the cluster centroids
#print(km2.cluster_centroids_)
#print(km3.cluster_centroids_)
#print(km4.cluster_centroids_)
#print(km5.cluster_centroids_)
#
#clusters = pd.Series(clusters)
#my_modes= pd.concat([my_modes,clusters], axis=1)
#my_modes.iloc[:,-1].value_counts()
#





#Join the two solutions
import numpy as np
join_clust = pd.DataFrame(np.column_stack((df_comp['EM_label'],clusters)))
join_clust.columns =  ['Product','Customer']

crosstab=pd.crosstab(join_clust.Customer,join_clust.Product)

crosstab.index
#cluster 2 viel für motor, und in allen anderen wenig
#cluster 1 wenig für motor, viel für alles und mittel für health
#cluster 3 überall normal etwas mehr in health 

coldict={0.0:'Unsure', 1.0:'Low on Motor, but everything else', 2.0:'High on motor, but everything else'}
idxdict={0.0:'Medium Salary | Low claims | Children | High Education', 1.0:'Low Salary | High claims | Children | Low Education', 2.0:'High Salary | Medium high claims | No children | High Education'}
crosstab.rename(columns=coldict,index=idxdict , inplace=True)



treeplot1 = pd.DataFrame(crosstab.iloc[:,1]*potential_1)
treeplot2 =pd.DataFrame(crosstab.iloc[:,2]*potential_2)

import matplotlib.pyplot as plt
import squarify # pip install squarify (algorithm for treemap)
 
# Change color
squarify.plot(sizes=[265336.7513616276,400743.1235386683,303668.70230880455,931734.8204554497,712473.3110765314,605627.6332960423],
              label=["Low Motor and  Medium Salary | Low claims | Children | High Education", "Low Motor and Low Salary | High claims | Children | Low Education", "Low Motor and High Salary | Medium high claims | No children | High Education", "High Motor and  Medium Salary | Low claims | Children | High Education", "High Motor and Low Salary | High claims | Children | Low Education", "High Motor and High Salary | Medium high claims | No children | High Education"],
              color = ['#713b87', '#9b59b6', '#bb8dcd', '#1d6fa5', '#217ebb','#8bc5ea'])
plt.axis('off')
plt.show()






#Evaluate the Customer

#from sklearn.decomposition import PCA
#pca = PCA(n_components=2).fit(df_comp)
#pca_2d = pca.transform(df_comp)
#for i in range(0, pca_2d.shape[0]):
#    if db.labels_[i] == 0:
#        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
#    elif db.labels_[i] == 1:
#        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
#    elif db.labels_[i] == 2:
#        c4 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='k',marker='v')
#    elif db.labels_[i] == 3:
#        c5 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='y',marker='s')
#    elif db.labels_[i] == 4:
#        c6 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='m',marker='p')
#    elif db.labels_[i] == 5:
#        c7 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='c',marker='H')
#    #elif db.labels_[i] == -1:
#    #    c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
#
#plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
#plt.title('DBSCAN finds 2 clusters and noise')
#plt.show()

''' old clustering from here on '''


### 7. MODELLING ##############################################################

from sklearn.cluster import KMeans

df.drop(['BirthYear_pred'], axis=1, inplace=True)

# Drop ClaimRate because it is in formula for Customer Monetary Value and corr is -0.99
# Customer Monetary Value contains info about how long the customer is a client
df.corr()
df.drop(['ClaimRate'], axis=1, inplace=True)


#Create data frame with only numerical values. Dropped CustId, Edu Degree, GeoLivArea, HasChild
#df_numerical=df[['1stPolYear', 'GrossMthSalary', 'CustMonetVal', 'PremLOBMotor','PremLOBHousehold', 'PremLOBHealth', 'PremLOBLife','PremLOBWorkCompensation']]
#df_customer=df[['1stPolYear', 'GrossMthSalary']]
df_company=df[['CustMonetVal', 'PremLOBMotor','PremLOBHousehold', 'PremLOBHealth', 'PremLOBLife','PremLOBWorkCompensation']]

#Scale data

#from sklearn.decomposition import PCA
#pca = PCA(n_components = None)
#df_numerical = pca.fit_transform(df_numerical)
#ex_var = pca.explained_variance_ratio_

#pca.fit(df_numerical)

#red_pca = np.dot(pca.transform(df_numerical)[:,:5],pca.components_[:5,:])
#red_pca += np.mean(df_numerical, axis=0)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_company = scaler.fit_transform(df_company)

df_company = pd.DataFrame(df_company)
#Plotting elbow graph to determine K
wcss=[]
for i in range(1,16):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_company)
    wcss.append(kmeans.inertia_)
    #inertia = Sum of squared distances of samples to their closest cluster center.

plt.plot(range(1,16), wcss, color='green')
plt.title('Elbow Graph')
plt.xlabel('Number of clusters K')
plt.ylabel('WCSS')

#Training the model
kmeans=KMeans(n_clusters=6)
df_company['kmean']=kmeans.fit_predict(df_company)

df_company['kmean'].hist()
#plt.scatter(df_company[0], df_company[1])
plt.scatter(df_company[df_company['kmean']==0][2], df_company[df_company['kmean']==0][4], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(df_company[df_company['kmean']==1][2], df_company[df_company['kmean']==1][4], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(df_company[df_company['kmean']==2][2], df_company[df_company['kmean']==2][4], s = 100, c = 'green', label = 'Cluster 1')


sns.pairplot(df_company, hue='kmean')

# ############################ Hierarchical Clustering

#l1 = [df['EduDegree'],df['HasChild'],df['GeoLivArea']]
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



import numpy as np
import pandas as pd

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage#, set_link_color_pallete
#from scipy.cluster.hierarchy import fcluster
#from scipy.cluster.hierarchy import cophenet
#from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy

from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

np.set_printoptions(precision=4,
                    threshold = 200,
                    suppress = True)

plt.figure(figsize=(10,5))
plt.style.use('seaborn-whitegrid')

#Scipy generate dendrograms



test = df[['1stPolYear','BirthYear','GrossMthSalary', 'GeoLivArea', 'HasChild']]
test = test.dropna()

my_scaler = StandardScaler()

test = my_scaler.fit_transform(test)

Z = linkage(test,
            method = 'ward')#method='single, complete

dendrogram(Z,
           #truncate_mode='none',
           truncate_mode='lastp',
           p=40,
           orientation = 'top',
           leaf_rotation=45.,
           leaf_font_size=10.,
           show_contracted=True,
           show_leaf_counts=True)

plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')

#plt.axhline(y=50)
plt.axhline(y=55)
plt.show()


#Scikit

k=4
from sklearn.cluster import AgglomerativeClustering
Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')

#Replace the test with proper data
my_HC = Hclustering.fit(test)

my_labels = pd.DataFrame(my_HC.labels_)
my_labels.columns =  ['Labels']

test = pd.concat([pd.DataFrame(test), my_labels], axis=1)
test.columns =  ['1stPolYear','BirthYear','GrossMthSalary', 'GeoLivArea', 'HasChild', 'Labels']

to_revert = test.groupby(['Labels'])['1stPolYear','BirthYear','GrossMthSalary', 'GeoLivArea', 'HasChild'].mean()
#to_revert = to_revert.loc[:,-'Index']

my_scaler.inverse_transform(X=to_revert )
test['Labels'].value_counts()




#### DBSCAN


from sklearn.cluster import DBSCAN
from sklearn import metrics

db = DBSCAN(eps=0.2,
            min_samples=5).fit(test)

labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_clusters, counts_clusters = np.unique(db.labels_, return_counts = True)
print(np.asarray((unique_clusters, counts_clusters)))

from sklearn.decomposition import PCA
pca = PCA(n_components = None).fit(test)
pca_2d = pca.transform(test)
explained_variance = pca.explained_variance_ratio_


from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(test)
pca_2d = pca.transform(test)
for i in range(0, pca_2d.shape[0]):
    if db.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif db.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif db.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')

plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
plt.title('DBSCAN finds 2 clusters and noise')
plt.show()

#Mean Shift

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# Compute clustering with MeanShift

to_MS = df_company
to_MS = to_MS.dropna()
#to_MS.income = to_MS.income.astype(int)

#test = StandardScaler().fit_transform(test)
#To reverse
from sklearn.preprocessing import MinMaxScaler
my_scaler = MinMaxScaler()

to_MS = my_scaler.fit_transform(to_MS)


# The following bandwidth can be automatically detected using
my_bandwidth = estimate_bandwidth(to_MS,
                               quantile=0.2,
                               n_samples=1000)

ms = MeanShift(bandwidth=my_bandwidth,
               #bandwidth=0.15,
               cluster_all=False,
               bin_seeding=True)

ms.fit(to_MS)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


#Values
my_scaler.inverse_transform(X=cluster_centers)

#Count
unique, counts = np.unique(labels, return_counts=True)

print(np.asarray((unique, counts)).T)

# lets check our are they distributed

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(to_MS)
pca_2d = pca.transform(to_MS)
for i in range(0, pca_2d.shape[0]):
    if labels[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif labels[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif labels[i] == 2:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
    elif labels[i] == -1:
        c4 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='c',marker='H')
    

plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Cluster 3', ','])
plt.title('Mean Shift found 3 clusters')
plt.show()

# ################################# K - Modes!!


# Lets do k MODES Wiii !!!
df.reset_index(drop=True, inplace=True)
df['1stPolYear_bin'] = ''
for i in range(len(df)):
    if df['1stPolYear'][i] < df['1stPolYear'].quantile(q=0.2):
        df['1stPolYear_bin'][i] = '1stQ'
    elif df['1stPolYear'][i] < df['1stPolYear'].quantile(q=0.4):
        df['1stPolYear_bin'][i] = '2ndQ'
    elif df['1stPolYear'][i] < df['1stPolYear'].quantile(q=0.6):
        df['1stPolYear_bin'][i] = '3rdQ'
    elif df['1stPolYear'][i] < df['1stPolYear'].quantile(q=0.8):
        df['1stPolYear_bin'][i] = '4thQ'
    else: df['1stPolYear_bin'][i] = '5thQ'
        
    96, 84, 76 

#df_cust = df['1stPolYear', 'HasChild', '2 - HighSchool', 'HigherEdu', '2','3','4']


df['1stPolYear'].quantile(q=0.2)
df['1stPolYear'].quantile(q=0.4)
df['1stPolYear'].quantile(q=0.6)
df['1stPolYear'].quantile(q=0.8)





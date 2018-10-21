Schedule:
Monday: Ask Joel about DM questions, Homwework pdf shared by Lukas
Tuesday: Questions about homework to the professor, finish homework report, fit DM data code

Questions:

Questions Data Mining:
1)	3 sigma didn’t drop outliers in one case
2)	Is log of 0 ok? 
#LOG OF VALUE 0? 
df['PremLOBHousehold'].min()#min is -75
df['PremLOBHousehold']=np.log(df['PremLOBHousehold'] + 1 - min(df['PremLOBHousehold']))
#ACTUALLY I THINK ITS OK, WE DID+1 SO THERES NO LOG OF 0….
df['PremLOBHousehold'].min()#min is 0#THEN ITS JUST THE RESULT OF LOF WHICH IS 0
3)	Should we do log of all numeric variables? or is Standard Scaler going to do the job (normalize)
4)	Should we scale binary?  Ordinal?
5)	Can we make a cardinal variable of education? But then, how is distance going to be measured in models such as clustering where distance is calculated? Do we include categorical variables in such models? 
6)	How does label encoder assign labels to category, alphabetical or..?
7)	Is it ok to use kde plot to check the distribution?

*** Which models require data scaling?
*** Which models require normal distribution?


                                                      Tasks:
                                                      
1)) Error data
-Specify all error data (what kind of errors in data, e.g. really old people, 1 year old beneficiary etc.)
 First: Column by Column
 Then: Combinatory analysis of columns
   
   ERRORS:
   1st Pol Year drop >2016
   Bday Year drop >2016 and <1900
   no error in education
   gross month salary-drop outliers
   geography no errors
   has children no errors
   customer monetary, claims rate lets keep for now
   LOB all good  cause from system

2)Null values replacement 
ID-Drop
Bday-Regression with Salary First
Salary-Regression on Bday Second
1st Year-NN (30 nulls - mean/median?)
Education-NN (17 nulls -mean/median?)
Geogrpahy-NN (1 null- missing a lot of other columns, DROP!)
Has children-NN (21 nulls, KNN/replace with 1?) after doing knn if 80% has 1 then its good!
Customer Monet-NO NULLS
Claims Ratio - NO NULLS
Premiums-replace with 0 or N/A
 
   
3)Outliers detection

For now we will model without caring about outliers.


 -Drop 0.377% of data by applying 3 sigma rule.
 -Z-Score and IQR-recalculate
  What do we do with wrong data? 
    -Replace with mean? (if we replace 20% of column with mean then this column might mistakenly turn out to be insignificant)
    -Replace with nearest neighbor? 
    -Create temporary variable to compare salary vs sum of Premiums and replace where salary<Sum of Premiums with Sum of Premiums.
    -Categorize age (Under age, middle, elderly etc.)


4)Feature selection(PCA calculation)
Correlation
drop Id column
PCA/Backward vs Forward elimination

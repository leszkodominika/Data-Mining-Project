# Data-Mining-Project
Questions:
1) Database info (Description and formulas for all the columns,  why all LOB so highly correlated, what are these LOB columns?)

When presenting model we need to give red flags: what we assume, what we found etc. write about all the issues
Dont remove more than 10% of data!
for outliers max 3%

-LOB column: someone actually paid the premium? and if not then what is this value? (interpretation) Yes, paid. 
-What if someone under 18 has value in LOB Premium Motor Insurance?
-What does Premium LOB negative mean? someone paid last year, then dropped insurance and is getting back the money or is getting a refund from last year
-Premium Amount higher than salary?
-All LOB highly correlated?
-What does it mean that 1st policy year BEFORE birth year? In modify phase. Drop column or Drop rows with nonsense bday (but we will lose info if its a lot).or replace with mean or median 
-Formulas for LOB columns? (coolinearility)


replacing "wrong" data with some mean or something like that instead of droping that observation


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

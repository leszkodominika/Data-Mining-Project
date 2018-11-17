1. Prem LOB columns: SPLITTING (2 models?). Will we use all of them? Maybe: Try to sum these columns and present as 1.
2. In skewed distribution (mainly PremLobHousehold, Life, Compensation) 3 sigma rule drops 4.2% of data. Apply 3.5/4 sigma
********************************************NULL HANDLING************************************************
                    *******MAKE SURE NO ERRORS IN NULL VALUES REPLACEMENT***********
3.Determine how to remove nulls in 1st Year Policy.
4.Regression GrossMthSalary-BirthYear. Drop Birth Year
5. KNN for has a child (X train should have no nulls if points 1 and 2 performed)
6. Decision Tree for EduDegree 
****************************************************************MODELLING**************************************
7. K Means: numeric columns only! Make sure you managed LOB Splitting.
8. K Modes: categorical columns only!!!
9. Hierarchical numeric columns only! Make sure you managed LOB Splitting.
10. DBSCAN

11.DO SCATTERPLOT FOR EACH VARIABLE COLORED BY CLASS LABEL TO SEE DEPENDENCIES.
12.DO PCA OUT OF THE COLUMNS WHICH DONT HAVE PARTICULAR DEPENDENCIES (OVERLAPPING)
**************************************************************************************************************
NOTES:
For remaining algorithms make sure if we use weights
-We do regression only on data without errors/outliers
-Implement one scaler (minmax/standard) and put some reasoning why we chose that one. 
-don’t scale binary
=Remember to have arguments on why u use method X to replace the values. 
-Do Exploratory Data Analysis like in IP.


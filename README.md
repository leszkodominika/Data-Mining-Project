Our comments and tasks:

-Change Education to dummy, keep -1 column -DOMINIKA
Create 2 datastes:
1) For K-Means, hierarchical don’t use categorical columns
2) For remaining algorithms make sure if we use weights
-We do regression only on data without errors/outliers
-Implement one scaler (minmax/standard) and put some reasoning why we chose that one. 
-don’t scale binary

1. Prem LOB columns: SPLITTING (2 models?). Will we use all of them? Maybe: Try to sum these columns and present as 1.
2. In skewed distribution (mainly PremLobHousehold, Life, Compensation) 3 sigma rule drops 4.2% of data. Apply 3.5/4 sigma
********************************************NULL HANDLING************************************************
                    *******MAKE SURE NO ERRORS IN NULL VALUES REPLACEMENT***********
3.Determine how to remove nulls in 1st Year Policy.
4.Regression GrossMthSalary-BirthYear. Drop Birth Year
5. KNN for has a child (X train should have no nulls if points 1 and 2 performed)
6. Decision Tree for EduDegree 
***************************************************************************************************************************************
Remember to have arguments on why u use method X to replace the values. 
Do Exploratory Data Analysis like in IP.
****************************************************************MODELLING*****************************************************************
K Means: numeric columns only! Make sure you managed LOB Splitting.
K Modes: categorical columns only!!!
Hierarchical
DBSCAN

DO SCATTERPLOT FOR EACH VARIABLE COLORED BY CLASS LABEL TO SEE DEPENDENCIES.
DO PCA OUT OF THE COLUMNS WHICH DONT HAVE PARTICULAR DEPENDENCIES (OVERLAPPING)

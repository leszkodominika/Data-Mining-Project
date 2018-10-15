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

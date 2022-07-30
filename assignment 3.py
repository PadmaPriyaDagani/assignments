# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:34:10 2022

@author: hp
"""

#question 1
# A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units
#A randomly selected sample of cutlets was collected from both units and measured? 
#Analyze the data and draw inferences at 5% significance level.
#Please state the assumptions and tests that you carried out to check validity of the assumptions.


#H0= mu1 = mu2
#H1 = mu1 != mu2
import pandas as pd
df=pd.read_csv("Cutlets.csv")
df

df["Unit A"].mean()
df["Unit B"].mean()

# mean of Unit A =7.01909142857143
# mean of Unit B = 6.964297142857142

from scipy import stats
zcal, pval=stats.ttest_ind(df["Unit A"],df["Unit B"])

#zcal, pval =(0.7228688704678063, 0.4722394724599501)

print("Zcalculated value is", zcal.round(4))
print("P-value is", pval.round(4))

#Zcalculated value is 0.7229
#P-value is 0.4722

if pval<0.05:
    print("reject Null hypothesis,accept Alternative hypothesis")
else:
    print("accept Null hypothesis,reject Alternative hypothesis")

#accept Null hypothesis,reject Alternative hypothesis
# from the above result we can say that there is a similarity between Unit A and Unit B
--------------------------------------------------------------------------------------------    
#question 2
#A hospital wants to determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories on their preferred list. 
#They collected a random sample and recorded TAT for reports of 4 laboratories. TAT is defined as sample collected to report dispatch.
#  Analyze the data and determine whether there is any difference in average TAT among the different laboratories at 5% significance level.
 
import pandas as pd
df=pd.read_csv("LabTAT.csv")
df

#H0= mean values of all the 4 Laboratories are similar
#H1= mean values of all the 4 Laboratories are not similar
# Anova test
import scipy.stats as stats
stat, pval = stats.f_oneway(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],df.iloc[:,3])

#stat, pval= (118.70421654401437, 2.1156708949992414e-57)
 if pval<0.05:
     print("reject Null hypothesis,accept Alternative hypothesis")
 else:
     print("accept Null hypothesis,reject Alternative hypothesis")

#reject Null hypothesis,accept Alternative hypothesis
#mean of all the 4 Laboratories are notvsame as there is a Variance
-------------------------------------------------------------------------------------------
#question 3
#find the  similarity in th ebuyer ration of sales between male and female

import pandas as pd
df=pd.read_csv("BuyerRatio.csv")
df

#H0 = all proportions are equal
#H1 = all proportions are not euqual

alpha=0.05
Male = [50,142,131,70]
Female=[435,1523,1356,750]
Sales=[Male,Female]

#Contingency table is done
from scipy.stats import chi2_contingency
stat,p,dof,expected=chi2_contingency(Sales)
print("stat,p,dof,expected",chi2_contingency(Sales))
#(1.595945538661058,
# 0.6603094907091882,
 #3,
# array([[  42.76531299,  146.81287862,  131.11756787,   72.30424052],
#        [ 442.23468701, 1518.18712138, 1355.88243213,  747.69575948]]))

# compare with p value
 if p<0.05:
     print("reject Null hypothesis,accept Alternative hypothesis")
 else:
     print("accept Null hypothesis,reject Alternative hypothesis")

#accept Null hypothesis,reject Alternative hypothesis
#proportions of male and female across the regions are same

---------------------------------------------------------------------------------------
#Question 4
#4 centers of the TeleCall audit a certain %  of the customer order forms. 
#Any error in order form renders it defective and has to be reworked before processing. 
#The manager wants to check whether the defective %  varies by centre. 
#Please analyze the data at 5% significance level and help the manager draw appropriate inferences

import pandas as pd
df=pd.read_csv("Costomer+OrderForm.csv")
df

#H0= defective % does not varies by center
#H1= defective % does not varies by center


alpha=0.05
df.Phillippines.value_counts()
#Error Free    271
#Defective      29
#Name: Phillippines, dtype: int64
df.Indonesia.value_counts()
 #Error Free    267
 #Defective      33
#Name: Indonesia, dtype: int64
df.Malta.value_counts()
#Error Free    269
#Defective      31
#Name: Malta, dtype: int64
df.India.value_counts()
#Error Free    280
#Defective      20
3Name: India, dtype: int64

#contingency table
import numpy as np
from scipy.stats import chi2_contingency
obs=np.array([[271,267,269,280],[29,33,31,20]])
obs

#chi2_contingency
stat,p,dof,expected=chi2_contingency(obs)
 print("stat,p,dof,expected",chi2_contingency(obs))
#stat,p,dof,expected (3.858960685820355, 0.2771020991233135, 3, array([[271.75, 271.75, 271.75, 271.75],
 #      [ 28.25,  28.25,  28.25,  28.25]]))
 
 #compare p value with alpha
 if p<0.05:
     print("reject Null hypothesis,accept Alternative hypothesis")
 else:
     print("accept Null hypothesis,reject Alternative hypothesis")
#accept Null hypothesis,reject Alternative hypothesis  
# Thus,Customer order forms defective% does not varries with center


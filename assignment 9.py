# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:25:43 2022

@author: hp
"""
#Prepare rules for the all the data sets 
#1) Try different values of support and confidence. Observe the change in number of rules for different support,confidence values
#2) Change the minimum length in apriori algorithm
#3) Visulize the obtained rules using different plots 

pip install mlxtend
import pandas as pd
df=pd.read_csv("book.csv")
df

df.head()
df.shape
df.info()
df.describe()
df.corr()

#Apriori algorithm
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
#Data preprocessing is not required as it is already in Transaction format
# 1)association rules with 10%support and 70%confidence
frequent_itemsets=apriori(df,min_support=0.1,use_colnames=True)
frequent_itemsets
#association rules with 70%confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules
#A leverage value of 0 indicates independence.
#Range will be A high conviction value means that the consequent is highly depending on the antecedent and range [0 inf]

rules.sort_values('lift',ascending=False)

#Lift Ratio > 1 is a good influential rule in selecting the associated transactions.
rules[rules.lift>1]

#visyalization of obtained rules
import matplotlib.pyplot as plt
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

import seaborn as sns
sns.boxplot(x='support',y='confidence',data=rules)

# 2)association rules with 20% support and 60% confidence

frequent_itemsets2=apriori(df,min_support=0.2,use_colnames=True)
frequent_itemsets2
#association rules with 60%confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.6)
rules2

rules2.sort_values('lift',ascending=False)

#Lift Ratio > 1 is a good influential rule in selecting the associated transactions.
rules2[rules2.lift>1]

#visyalization of obtained rules
import matplotlib.pyplot as plt
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

import seaborn as sns
sns.boxplot(x='support',y='confidence',data=rules2)

# 3)association rules with 5%support and 80% confidence
frequent_itemsets3=apriori(df,min_support=0.05,use_colnames=True)
frequent_itemsets3
#association rules with 80%confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.8)
rules3

rules3.sort_values('lift',ascending=False)

#Lift Ratio > 1 is a good influential rule in selecting the associated transactions.
rules3[rules3.lift>1]

#visyalization of obtained rules
import matplotlib.pyplot as plt
plt.scatter(rules3['support'],rules3['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

import seaborn as sns
sns.boxplot(x='support',y='confidence',data=rules3)
--------------------------------------------------------------------------------------------
#movies data set
import pandas as pd
movies=pd.read_csv("my_movies.csv")
movies

movies.head()
movies.shape
movies.info()
movies.describe()
movies.corr()

movies2= movies.iloc[:,5:]
movies2

#Apriori algorithm
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
#Data preprocessing is not required as it is already in Transaction format
# 1)association rules with 10%support and 70%confidence
frequent_itemsets=apriori(movies2,min_support=0.1,use_colnames=True)
frequent_itemsets
#association rules with 70%confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules
#A leverage value of 0 indicates independence.
#Range will be A high conviction value means that the consequent is highly depending on the antecedent and range [0 inf]

rules.sort_values('lift',ascending=False)

#Lift Ratio > 1 is a good influential rule in selecting the associated transactions.
rules[rules.lift>1]

#visyalization of obtained rules
import matplotlib.pyplot as plt
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

import seaborn as sns
sns.boxplot(x='support',y='confidence',data=rules)

# 2)association rules with 20% support and 60% confidence

frequent_itemsets2=apriori(movies2,min_support=0.2,use_colnames=True)
frequent_itemsets2
#association rules with 60%confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.6)
rules2

rules2.sort_values('lift',ascending=False)

#Lift Ratio > 1 is a good influential rule in selecting the associated transactions.
rules2[rules2.lift>1]

#visyalization of obtained rules
import matplotlib.pyplot as plt
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

import seaborn as sns
sns.boxplot(x='support',y='confidence',data=rules2)

# 3)association rules with 30%support and 50% confidence
frequent_itemsets3=apriori(movies2,min_support=0.3,use_colnames=True)
frequent_itemsets3
#association rules with 50%confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.5)
rules3

rules3.sort_values('lift',ascending=False)

#Lift Ratio > 1 is a good influential rule in selecting the associated transactions.
rules3[rules3.lift>1]

#visyalization of obtained rules
import matplotlib.pyplot as plt
plt.scatter(rules3['support'],rules3['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

import seaborn as sns
sns.boxplot(x='support',y='confidence',data=rules3)
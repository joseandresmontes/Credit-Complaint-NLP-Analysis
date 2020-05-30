# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:00:03 2020

@author: JoseA
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt

#df = pd.pandas.read_csv('./data/complaints.csv')
#subset = df.iloc[:250,0:18]
#subset.to_csv("./data/subset.csv")
#df = pd.read_csv("./data/complaints.csv", index_col = False)

#(might take a while)
df = pd.read_csv('http://files.consumerfinance.gov/ccdb/complaints.csv.zip', 
                 compression='zip', header=0, sep=',', quotechar='"',
                 index_col = False)

# selecting only relavent columns and renaming columns for easier manipulation
df = df.drop(columns=[
            'Sub-issue',
            'Sub-product',
            'Tags', 
            'Consumer disputed?',
            'Complaint ID'])

df.rename(columns = {'Date received':'date', 
                     'Consumer complaint narrative': 'Complaint'}, 
              inplace = True)

# filtering to Top 3 Consumer Credit Rating Agencies
df = df[df.Company.isin(['EQUIFAX, INC.', 
                         'Experian Information Solutions Inc.',
                         'TRANSUNION INTERMEDIATE HOLDINGS, INC.'])]

index = df.loc[:,('Company')].map({'TRANSUNION INTERMEDIATE HOLDINGS, INC.': 'Transunion', 'Experian Information Solutions Inc.':'Experian', 'EQUIFAX, INC.':'Equifax'}).values
df.loc[:,('Company')] = index
df = df.set_index(index)

# conducting analysis on a per year basis (loop to be done soon)

df['date'] = pd.to_datetime(df.date)
df['year'] = df.date.dt.year

df.groupby('year').Company.value_counts().unstack(level=0).mean().plot();

plt.title('Annual Complaints', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Complaints', fontsize=12)

df.groupby('year').Company.value_counts().to_frame().unstack(level=1).plot(legend=False, kind='bar');

plt.title('Annual Complaints', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Complaints', fontsize=12)
plt.legend(('Equifax','Experian','Transunion'))

df_group = df.groupby("year")
df_dict = {year: df_group.drop("year", axis=1) 
          for year, df_group in df_group}
df_nas = pd.DataFrame()
result = []
na = []
filed = []
for year in df_dict:
    result.append(year)
    na.append(df_dict[year].Complaint.isna().sum())
    filed.append(df_dict[year].Complaint.value_counts().sum())
df_nas["Year"] = result
df_nas["NA"] = na
df_nas["Filed"] = filed
df_nas.set_index("Year").plot(legend=True, kind='bar');

plt.title('Annual Filed Complaints', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Count', fontsize=12)

##########################################################Data Setup (Pickle)#
df = df[["date","year","Product","Complaint"]]
df.dropna(inplace=True)

filename = 'df'
outfile = open(filename,'wb')
pickle.dump(df,outfile)
outfile.close()
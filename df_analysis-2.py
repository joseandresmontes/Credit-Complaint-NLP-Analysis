# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:35:11 2020

@author: JoseA
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nlp import nlp_functions
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#####################################################Upload Pickled Data######
filename = 'df'
infile = open(filename, 'rb')
df = pickle.load(infile)
infile.close()

####################################################Pre-processing Cleaning###
clean = nlp_functions
preclean = lambda x: clean.clean_text_preclean(x)
stopremoval = lambda x: clean.stopRemove(x)
stem = lambda x: clean.stemSentence(x)

df['Complaint'] = df['Complaint'].apply(stopremoval)
df['Complaint'] = df['Complaint'].apply(preclean)
df['Complaint_stem'] = df['Complaint'].apply(stem)

df_group = df.groupby("year")
df_dict = {year: df_group.drop("year", axis=1) 
          for year, df_group in df_group}

################################################################WordCloud#####
wc = WordCloud(background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)

for year in df_dict:
    text = "".join(df_dict[year].Complaint_stem)
    wc = wc.generate(text)
    plt.title(str(year) + " Complaint Term Prevalence")
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
###############################################################DTM############

vect = TfidfVectorizer(min_df=.10,max_df=.95, ngram_range=(1, 2) , stop_words='english')

vect_dict = []
dtm_dict = []

for year in df_dict:
    complaint_vect = vect.fit_transform(df_dict[year].Complaint_stem)
    complaint_dtm = pd.DataFrame(complaint_vect.toarray(), columns=vect.get_feature_names())
    complaint_dtm.index = df_dict[year].Complaint_stem.index
    vect_dict.append(complaint_vect)
    dtm_dict.append(complaint_dtm)
    

cv = CountVectorizer(min_df=.10,max_df=.95, ngram_range=(1, 2) , stop_words='english')

cv_dict = []
dtm_cv_dict = []

for year in df_dict:
    complaint_cv = cv.fit_transform(df_dict[year].Complaint_stem)
    complaint_dtm_cv = pd.DataFrame(complaint_cv.toarray(), columns=cv.get_feature_names())
    complaint_dtm_cv.index = df_dict[year].Complaint_stem.index
    cv_dict.append(complaint_cv)
    dtm_cv_dict.append(complaint_dtm_cv)

year=[2015,2016,2017,2018,2019,2020]
vect_dict = dict(zip(year,vect_dict))
dtm_dict = dict(zip(year,dtm_dict))
cv_dict = dict(zip(year,cv_dict))
dtm_cv_dict = dict(zip(year,dtm_cv_dict))

########################################################Sentiment Analysis####

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

for year in df_dict:
    df_dict[year] = df_dict[year].assign(polarity=df_dict[year].Complaint_stem.apply(pol))
    df_dict[year] = df_dict[year].assign(subjectivity= df_dict[year].Complaint_stem.apply(sub))
    df_dict[year] = df_dict[year].assign(Company = df_dict[year].index)

for year in df_dict:
    sns.scatterplot(x="polarity", y="subjectivity",
              hue="Company",alpha=0.4,               
              data= df_dict[year])
    plt.title(str(year)+' Sentiment Analysis')
    plt.xlabel('< Negative (Polarity) Positive >')
    plt.ylabel('< Facts (Subjectivity) Opinions >')
    plt.legend(loc=2)
    plt.show()

for year in df_dict:
    mean = df_dict[year].groupby('Company').mean()
    sns.scatterplot(data = mean,x='polarity',y='subjectivity',hue=mean.index, s=80)
    plt.xlabel('< Negative (Polarity) Positive >')
    plt.ylabel('< Facts (Subjectivity) Opinions >')
    plt.title(str(year)+" Complaint Mean")
    plt.legend(loc=2)
    plt.show()
    
filename = 'df_dict'
outfile = open(filename,'wb')
pickle.dump(df_dict,outfile)
outfile.close()
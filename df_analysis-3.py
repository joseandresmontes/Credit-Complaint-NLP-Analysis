# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:47:08 2020

@author: JoseA
"""

from nlp import nlp_functions
from gensim import matutils, models
import scipy.sparse
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer


tag = nlp_functions

filename = 'df_dict'
infile = open(filename, 'rb')
df_dict = pickle.load(infile)
infile.close()

cv = CountVectorizer(min_df=.10,max_df=.95, ngram_range=(1, 2) , stop_words='english')

tag = nlp_functions

from nlp import nlp_functions
tag = nlp_functions

for year in df_dict:
    data_nouns_adj = pd.DataFrame(df_dict[year].Complaint_stem.apply(tag.nouns_adj))
    data_cvna = cv.fit_transform(data_nouns_adj.Complaint_stem)
    data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cv.get_feature_names())
    data_dtmna.index = data_nouns_adj.index
    corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))
    id2wordna = dict((v, k) for k, v in cv.vocabulary_.items())
    ldana = models.LdaModel(corpus=corpusna, num_topics=4, id2word=id2wordna, passes=10)
    print('\n')
    print('Year: ' + str(year) )
    print(ldana.print_topics())
    print('\n')
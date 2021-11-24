#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import string
import os
import sys
import re
from collections import Counter, defaultdict
from nltk import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
import itertools
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# In[19]:


qrels = pd.read_csv("D:\\Courses\\Sem 7 2021-22\\COL764\\COL764 Project\\AILA_2019_dataset\\relevance_judgments_priorcases.txt", sep = ' ', header=None)


# In[20]:


# is this stopword list the correct one ? can we try other stopword collections or may be custom ?
stopword_set = set(stopwords.words('english'))
def stopword_removal(word_list):

    return [word for word in word_list if word not in stopword_set ]    


# do we make the case lower for legal documents?
def tokenize_document(doc, lowercase = False, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
                      tokenizer = 'RegExpTokeinzer', lemmatize = True):
    
    # punctuations are removed with empty string
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    
    # do we convert to lowercase?
    if(lowercase):
        doc = doc.lower()
    
    
    # variety of tokenizers could be experimented with
    if(tokenizer == 'RegExpTokeinzer'):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(doc)
    
    # word tokenizer of nltk - a bit slow since based on NLP techniques
    elif(tokenizer == 'Word_tokenizer'):
        tokens = word_tokenize(doc)
    
        
    if(removeStopwords == True):
        tokens = stopword_removal(tokens)
    
        
    #variety of stemmers could be experimented with
    if(stemming):
        if(stemmer == 'PorterStemmer'):
            tokens = [ps.stem(token) for token in tokens ]
        
    
    # We can also perform lemmatization if we want
    # write function for lemmatizer here.
    if(lemmatize == True):
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    
    return tokens
        


# In[21]:


# this function is for processing the corpus
def process_corpus(filename, lowercase = False, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
                    tokenizer = 'RegExpTokeinzer'):
    corpus_files = os.listdir(filename)
    
    list_tokens = []
    
    indexes_of_doc = defaultdict(lambda:0)
    
    count = 0
    for corpus_file in corpus_files:
        filepath = os.path.join(filename, corpus_file)
        
        indexes_of_doc[count] = corpus_file[:-4]
        print(corpus_file)
        with open(filepath, 'r') as f:
            doc_text = f.readlines()
            
            
            doc_text = ''.join(doc_text)
            
            # this considers all words
            tokens = tokenize_document(doc_text, lowercase = lowercase, removeStopwords = removeStopwords , stemming = stemming, 
                                       stemmer = stemmer , tokenizer =  tokenizer)
            
            list_tokens.append(tokens)
        
        count +=1
    
    return list_tokens, indexes_of_doc
            

            


# In[22]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\AILA_2019_dataset\Object_casedocs", lowercase = True, stemming = True)


# In[23]:


def process_full_queries(filename):
    with open(filename, 'r') as f:
        doc_text = f.readlines() 
        doc_text = ''.join(doc_text)
        
    indexes = [x.start() for x in re.finditer('\|\|', doc_text)]
    
    print(len(indexes))
    
    queries = defaultdict(lambda: '')
    for i in range(len(indexes)):
        if(i == len(indexes) -1):
            queries['AILA_' + "Q" + str(i+1)] = doc_text[indexes[i] + 2:]
            
        else:
            queries['AILA_' + "Q" + str(i+1)] = doc_text[indexes[i] + 2: indexes[i+1] -1 - 7]

    print(indexes)
    
    return queries


# In[24]:


queries = process_full_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\AILA_2019_dataset\Query_doc.txt")


# In[12]:


def bm25_query(queries, queryname, bm25, lowercase = False, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
               tokenizer = 'RegExpTokeinzer'):
    
    query_scores_doc = defaultdict(lambda:0)
    
    query = queries[queryname]

    query_tokens = tokenize_document(query, lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, 
                                         stemmer = stemmer, tokenizer = tokenizer)
        

    doc_scores = bm25.get_scores(query_tokens)
        
        # top 100 queries for each doc
    indices = np.argsort(doc_scores)[::-1]
        
    values = [doc_scores[i] for i in indices]
        
        
    for i in range(len(indices)):
        index = indices[i]
        score = values[i]
            
        query_scores_doc[index] = query_scores_doc[index]  if query_scores_doc[index] > score else score

     
    results = sorted(query_scores_doc.items(), key = lambda x: x[1], reverse = True)
    
    
    rankings = []
    
    for result in results:
        index = result[0]
        rankings.append(indexes_of_doc[index])

    return rankings


# In[25]:


# 2nd model (filtering out with idf)
def bm25_query_idf(queries, queryname, bm25, lowercase = False, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
               tokenizer = 'RegExpTokeinzer'):
    
    query_scores_doc = defaultdict(lambda:0)
    
    query = queries[queryname]

    query_tokens = tokenize_document(query, lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, 
                                         stemmer = stemmer, tokenizer = tokenizer)
    
   
    query_tokens = tokenize_document(query, lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, 
                                         stemmer = stemmer, tokenizer = tokenizer)
    
    full_doc_scores = bm25.get_scores(query_tokens)
    
    
    filtered_query = sorted(query_tokens, key = lambda x : bm25.idf[x] if x in bm25.idf else 0, reverse = True)
                

    # selecting top 50 % terms
    query_tokens = filtered_query[0: len(filtered_query)//2]
        
    idf_doc_scores = bm25.get_scores(query_tokens)
    

    doc_scores = [full_doc_scores[i] + idf_doc_scores[i] for i in range(len(full_doc_scores))]

    indices = np.argsort(doc_scores)[::-1]
        
    values = [doc_scores[i] for i in indices]         
        
    for i in range(len(indices)):
        index = indices[i]
        score = values[i]
            
        query_scores_doc[index] = query_scores_doc[index]  if query_scores_doc[index] > score else score

     
    results = sorted(query_scores_doc.items(), key = lambda x: x[1], reverse = True)
    
    
    rankings = []
    
    for result in results:
        index = result[0]
        rankings.append(indexes_of_doc[index])
        
        
    
    return rankings


# In[26]:


def AP(qrel, rankings, queryname):
    relevant_docs = qrels[qrels[0] == queryname][qrels[3] == 1][2].values
    
    
    relevant_docs_retrieved = 0
    
    precision = []
    for i in range(len(rankings)):
        doc = rankings[i]
        docs_retrieved = i + 1
        
        if doc in relevant_docs:
            relevant_docs_retrieved +=1
            
            precision.append(relevant_docs_retrieved/docs_retrieved)
            
    
    if(len(precision) == 0):
        return 0, 0
        
    
    return np.sum(precision)/len(relevant_docs), len(precision)


def MRR(qrel,  rankings, queryname):
    relevant_docs = qrels[qrels[0] == queryname][qrels[3] == 1][2].values
    
    rec = 0
    for i in range(len(rankings)):
        doc = rankings[i]
        if doc in relevant_docs:
            return 1/(i+1)
    
    return rec

def P_at_10(qrel,  rankings, queryname):
    relevant_docs = qrels[qrels[0] == queryname][qrels[3] == 1][2].values
    
    relevant = 0
    for i in range(10):
        doc = rankings[i]
        if doc in relevant_docs:
            relevant +=1
    
    return relevant/10

def Rec_at_100(qrel,  rankings, queryname):
    relevant_docs = qrels[qrels[0] == queryname][2].values
    
    relevant = 0
    for i in range(100):
        doc = rankings[i]
        if doc in relevant_docs:
            relevant +=1
    
    return relevant/len(relevant_docs)
        


# In[27]:


def obtain_relevant_docs(queries, bm25, lowercase = False, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
                      tokenizer = 'RegExpTokeinzer', model = 1):
    
    querynames = queries.keys()
    
    
    list_AP = []
    list_MRR = []
    list_P10 = []
    list_Rec100 = []
    for queryname in querynames:
        if(model == 1):
            rankings = bm25_query(queries, queryname, bm25, lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, stemmer = stemmer,
                              tokenizer = tokenizer)[:100]
        elif(model == 2):
            rankings = bm25_query_idf(queries, queryname, bm25, lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, stemmer = stemmer,
                              tokenizer = tokenizer)[:100]
        
        ap = AP(qrels, rankings, queryname)
        
        mrr = MRR(qrels, rankings, queryname)
        
        p10 = P_at_10(qrels, rankings, queryname)
        
        rec100 = Rec_at_100(qrels,  rankings, queryname)
        
        print(queryname, ap, mrr, p10)
        
        list_AP.append(ap[0])
        list_MRR.append(mrr)
        list_P10.append(p10)
        list_Rec100.append(rec100)
        
    return np.mean(list_AP), np.mean(list_MRR) , np.mean(list_P10), np.mean(list_Rec100)


# In[28]:


def tune_hyperparameters(lowercase = False, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
                      tokenizer = 'RegExpTokeinzer', model = 1):
    k1_list = np.linspace(1, 2, 5)
    b_list = np.linspace(0.70, 1, 6)
    
    parameter_list = [k1_list, b_list]
    hyperparameters = list(itertools.product(*parameter_list))
    
    MAP_list = defaultdict(lambda:0)
    for x in hyperparameters:
        print(x)
        bm25 = BM25Okapi(list_tokens, k1= x[0], b= x[1])
        
        map, mrr, p10, rec100 = obtain_relevant_docs(queries, bm25, lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, stemmer = stemmer,
                              tokenizer = tokenizer, model = model)
        
        MAP_list[x] = map
        print("MAP is:", map)
        print("MRR is:", mrr)
        print("MRR is:", p10)
        print("Rec100 is:", rec100)
    return MAP_list
        
        


# In[ ]:


tune_hyperparameters(lowercase = True, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
                      tokenizer = 'RegExpTokeinzer', model = 1)


# In[188]:


bm25 = BM25Okapi(list_tokens, k1= 1.2, b= 0.75)       
map, mrr, p10, rec100 = obtain_relevant_docs(queries, bm25, lowercase = True, model = 1, stemming = True)


# In[189]:


map, mrr, p10, rec100


# In[172]:


bm25 = BM25Okapi(list_tokens, k1= 1.2, b= 0.75)       
map, mrr, p10, rec100 = obtain_relevant_docs(queries, bm25, lowercase = True, stemming = True, model = 2)


# In[17]:


map, mrr, p10, rec100


# In[ ]:





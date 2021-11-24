#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import os
import string
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from nltk import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# In[2]:


qrels = pd.read_csv("D:\\Courses\\Sem 7 2021-22\\COL764\\COL764 Project\\AILA_2019_dataset\\relevance_judgments_priorcases.txt", sep = ' ', header=None)


# In[3]:


# this function is for processing the corpus
def process_corpus(filename, lowercase = True):
    corpus_files = os.listdir(filename)
    
    tagged_data = []
    
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
            tagged_data.append(TaggedDocument(words=word_tokenize(doc_text.lower()), tags=[str(count)]))
            
        count +=1
    
    return tagged_data, indexes_of_doc


# In[4]:


tagged_data, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\AILA_2019_dataset\Object_casedocs", lowercase = True)


# In[5]:


def doc2vec_model(vectorsize, mincount, epochs):
    return gensim.models.doc2vec.Doc2Vec(vector_size= vectorsize, min_count= mincount, epochs= epochs)


# In[6]:


model = doc2vec_model(300 , 2, 80)
model.build_vocab(tagged_data)


# In[7]:


# it took some time to train. Though the corpus is small
model.train(tagged_data, total_examples=model.corpus_count, epochs= 80)


# In[8]:


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


# In[9]:


queries = process_full_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\AILA_2019_dataset\Query_doc.txt")


# In[1]:


def lda_vec(token_list):
    return  lda_model.get_document_topics(token_list, minimum_probability=0.0)
def prior_distributions(corpus):
    vectors = []
    for i in range(len(corpus)):
        vectors.append(lda_vec(corpus[i]))
        
    return vectors   


# In[11]:


prior_vec, indexes_of_doc = prior_vectors("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\AILA_2019_dataset\Object_casedocs")


# In[15]:


def sim(a, b, metric = 'cosine'):
    if(metric == 'cosine'):
        return np.inner(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))
    elif(metric == 'euclidean'):
        return np.abs(np.linalg.norm(a-b))
def similarity_scores(prior_vec, doc_vec, metric = 'cosine'):
    sim_scores = []
    for vec in prior_vec:
        sim_scores.append(sim(vec, doc_vec, metric = metric))
        
    return sim_scores
        
        
def doc2vec_full_query(full_queries, queryname, model, metric = 'cosine'):
    
    query_scores_doc = defaultdict(lambda:0)
    
    query = full_queries[queryname]
        
    doc_vec = model.infer_vector(word_tokenize(query.lower()))
        
        
    doc_scores = similarity_scores(prior_vec, doc_vec, metric = metric)
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


# In[20]:


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


# In[21]:


def obtain_relevant_docs(queries, model, flag = 1):
    
    querynames = queries.keys()
    
    
    list_AP = []
    list_MRR = []
    list_P10 = []
    list_Rec100 = []
    for queryname in querynames:
        if(flag == 1):
            rankings = doc2vec_query(queries, queryname, model, metric = 'cosine')
        elif(flag == 2):
            rankings = doc2vec_query(queries, queryname, model, metric = 'euclidean')
        elif(flag == 3):
            rankings = doc2vec_full_query(queries, queryname, model, metric = 'cosine')
        elif(flag == 4):
            rankings = doc2vec_full_query(queries, queryname, model, metric = 'euclidean')
        
        ap = AP(qrels, rankings, queryname)
        
        mrr = MRR(qrels, rankings, queryname)
        
        p10 = P_at_10(qrels, rankings, queryname)
        
        rec100 = Rec_at_100(qrels, rankings, queryname)
        
        print(queryname, ap, mrr, p10, rec100)
        
        list_AP.append(ap[0])
        list_MRR.append(mrr)
        list_P10.append(p10)
        list_Rec100.append(rec100)
        
    return np.mean(list_AP), np.mean(list_MRR) , np.mean(list_P10), np.mean(list_Rec100)


# In[22]:


obtain_relevant_docs(queries, model, flag = 3)


# In[ ]:





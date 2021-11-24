#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


qrels = pd.read_csv("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\irled-qrel.txt", sep = ' ', header=None)


# In[5]:


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


# In[12]:


tagged_data, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True)


# In[13]:


def doc2vec_model(vectorsize, mincount, epochs):
    return gensim.models.doc2vec.Doc2Vec(vector_size= vectorsize, min_count= mincount, epochs= epochs)


# In[14]:


model = doc2vec_model(300 , 2, 80)
model.build_vocab(tagged_data)


# In[15]:


# it took some time to train. Though the corpus is small
model.train(tagged_data, total_examples=model.corpus_count, epochs= 80)


# In[16]:


model.save("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\\trained_model.model")


# In[17]:


model = gensim.models.Word2Vec.load("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\\trained_model.model")


# In[19]:


def findall(p, s):

    i = s.find(p)
    while i != -1:
        yield i
        
        i = s.find(p, i+1)
def process_query(doc):
    citation_marker = "[?CITATION?]"
    
    return findall(citation_marker, doc)


# In[20]:


# function to get the query text by selecting region around the marker, how much to take?

# feedback: why construct the whole list ls1 and ls2, rather find one by one upto 40 spaces on either sides.
# i guess it would be more efficient.
def process_markers(s, i, p, length = 100):
    
    ls1 = [(a.start(), a.end()) for a in list(re.finditer(' ', s[:i]))]
    
    # another hyper-parameter,take max 40 spaces
    index1 = min(length, len(ls1))
    

    

    # find spaces
    ls2 = [(a.start(), a.end()) for a in list(re.finditer(' ', s[i + len(p):]))]
    
    # another hyper-parameter,take max 40 spaces
    index2 = min(length - 1, len(ls2)-1)
    
    
    # only considering spaces for now
    # may be later add logic for full stops and \n chars.
    return s[:i][ls1[-index1][0]:] + s[i+ len(p):][0: ls2[index2][0]]


def process_queries(filename, length = 100):
    query_files = os.listdir(filename)
    
    case_queries = defaultdict(lambda:[])
    
    for query_file in query_files:
        
        filepath = os.path.join(filename, query_file)
        
        
        
        with open(filepath, 'r') as f:
            doc_text = f.readlines() 
            doc_text = ''.join(doc_text)
            
        citation_marker_indices = list(process_query(doc_text))
        
        print(query_file, len(citation_marker_indices))
        
        for index in citation_marker_indices:
            query_text = process_markers(doc_text, index, "[?CITATION?]", length = length)
            
            case_queries[query_file[:-4]].append(query_text)
    
    return case_queries


def process_full_queries(filename):
    
    query_files = os.listdir(filename)
    
    case_queries = defaultdict(lambda: '')
    
    for query_file in query_files:
        filepath = os.path.join(filename, query_file)
        
        with open(filepath, 'r') as f:
            doc_text = f.readlines() 
            doc_text = ''.join(doc_text)
            
        case_queries[query_file[:-4]] = doc_text
            
    return case_queries
    


# In[21]:


# remove the pattern "27\."
queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = 400)


# In[22]:


def prior_vectors(model, filename):
    corpus_files = os.listdir(filename)
    
    prior_vec = []
    
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
            doc_vec = model.infer_vector(word_tokenize(doc_text.lower()))
            prior_vec.append(doc_vec)

            
        count +=1
    
    return prior_vec , indexes_of_doc


# In[ ]:


prior_vec, indexes_of_doc = prior_vectors(model, "D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases")


# In[ ]:


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
        
        
def doc2vec_query(queries, queryname, model, metric = 'cosine'):
    
    query_scores_doc = defaultdict(lambda:0)
    
    for query in queries[queryname]:
        
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


# In[ ]:


def AP(qrel, rankings, queryname):
    relevant_docs = qrels[qrels[0] == queryname][2].values
    
    
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
        
    
    return np.sum(precision)/5, len(precision)


def MRR(qrel,  rankings, queryname):
    relevant_docs = qrels[qrels[0] == queryname][2].values
    
    rec = 0
    for i in range(len(rankings)):
        doc = rankings[i]
        if doc in relevant_docs:
            return 1/(i+1)
    
    return rec

def P_at_10(qrel,  rankings, queryname):
    relevant_docs = qrels[qrels[0] == queryname][2].values
    
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


# In[ ]:


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


# In[84]:


obtain_relevant_docs(queries, model, flag = 1)


# In[36]:


# length = 400
obtain_relevant_docs(queries, model, flag = 1)


# In[49]:


full_queries = process_full_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases")
obtain_relevant_docs(full_queries, model, flag = 3)


# In[80]:


queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = 200)
obtain_relevant_docs(queries, model, flag = 1)


# In[70]:


obtain_relevant_docs(queries, model, flag = 2)


# In[ ]:


# remove the pattern "27\."
queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = 200)


# In[15]:


def tune_citation_length():
    
    MAP_list = defaultdict(lambda:0)
    
    hyperparameters = np.arange(50, 501, 50)
    
    print(hyperparameters)
    for x in hyperparameters:
        print("length=", x)
        
        queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = x)
        
        map, mrr, p10, rec100 = obtain_relevant_docs(queries, model, flag = 1)
        MAP_list[x] = map
        
        print("MAP is:", map)
        print("MRR is:", mrr)
        print("MRR is:", p10)
    return MAP_list
        


# In[25]:


tune_citation_length()


# In[13]:


def tune_model_vec():
    MAP_list = defaultdict(lambda:0)
    
    hyperparameters = np.arange(50, 501, 50)
    
    queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = 150)
    print(hyperparameters)
    
    for x in hyperparameters:
        print("length=", x)
        
        model = doc2vec_model(x , 2, 80)
        model.build_vocab(tagged_data)
        
        # it took some time to train. Though the corpus is small
        model.train(tagged_data, total_examples=model.corpus_count, epochs= 80)
        
        prior_vec, indexes_of_doc = prior_vectors(model, "D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases")
        
        map, mrr, p10, rec100 = obtain_relevant_docs(queries, model, flag = 1)
        MAP_list[x] = map
        
        print("MAP is:", map)
        print("MRR is:", mrr)
        print("MRR is:", p10)
    return MAP_list


# In[14]:


tune_model_vec()


# In[ ]:


obtain_relevant_docs(queries, model, flag = 1)


# In[ ]:





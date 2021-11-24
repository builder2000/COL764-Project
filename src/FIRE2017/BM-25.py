#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from rank_bm25 import BM25Okapi
import itertools
ps = PorterStemmer()


# In[2]:


qrels = pd.read_csv("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\irled-qrel.txt", sep = ' ', header=None)


# In[3]:


# is this stopword list the correct one ? can we try other stopword collections or may be custom ?
stopword_set = set(stopwords.words('english'))
def stopword_removal(word_list):

    return [word for word in word_list if word not in stopword_set ]    


# do we make the case lower for legal documents?
def tokenize_document(doc, lowercase = False, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
                      tokenizer = 'RegExpTokeinzer'):
    
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
    
    
    
    return tokens
        


# In[4]:


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
            

            


# In[5]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True)


# In[6]:


bm25 = BM25Okapi(list_tokens)


# In[7]:


def findall(p, s):

    i = s.find(p)
    while i != -1:
        yield i
        
        i = s.find(p, i+1)


# In[8]:


def process_query(doc):
    citation_marker = "[?CITATION?]"
    
    return findall(citation_marker, doc)


# In[9]:


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
        
        #print(query_file, len(citation_marker_indices))
        
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
    


# In[10]:


# remove the pattern "27\."
queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = 150)


# In[12]:


# strangely one case doesn't have citation markers case 0199


# In[11]:


def bm25_query(queries, queryname, bm25, lowercase = False, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
               tokenizer = 'RegExpTokeinzer'):
    
    query_scores_doc = defaultdict(lambda:0)
    
    for query in queries[queryname]:
        
        # using all terms 
        query_tokens = tokenize_document(query, lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, 
                                         stemmer = stemmer, tokenizer = tokenizer)
        
        #print(query_tokens)
        
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


# In[12]:


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
        


# In[13]:


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
                              tokenizer = tokenizer)
        elif(model == 2):
            rankings = bm25_query_idf(queries, queryname, bm25, lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, stemmer = stemmer,
                              tokenizer = tokenizer)
        
        ap = AP(qrels, rankings, queryname)
        
        mrr = MRR(qrels, rankings, queryname)
        
        p10 = P_at_10(qrels, rankings, queryname)
        
        rec100 = Rec_at_100(qrels,  rankings, queryname)
        
        print(queryname, ap, mrr, p10, rec100)
        
        list_AP.append(ap[0])
        list_MRR.append(mrr)
        list_P10.append(p10)
        list_Rec100.append(rec100)
        
    return np.mean(list_AP), np.mean(list_MRR) , np.mean(list_P10), np.mean(list_Rec100)


# In[14]:


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
        
        


# In[17]:


MAP_list = tune_hyperparameters(lowercase = True)


# In[ ]:


pattern = re.compile("([^\s]+\s){0,40}fg")

word = pattern.match("bdbkhdb fkjbkf fg ygfv cjhbc fg ccbkcb")


# In[ ]:


queries


# In[ ]:


word


# In[18]:


# tomorrow implement IDF 
# tomorrow try to get the langauge model right


# In[26]:


# results for MAP calcualtion using all items
sorted(MAP_list.items(), key = lambda x: x[1], reverse = True)


# In[27]:


## new part


# In[15]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True)
bm25 = BM25Okapi(list_tokens, k1= 1.5, b= 0.94)
map = obtain_relevant_docs(queries, bm25, lowercase = True)


# In[17]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = False)
bm25 = BM25Okapi(list_tokens, k1= 1.5, b= 0.94)
map = obtain_relevant_docs(queries, bm25, lowercase = False)


# In[21]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True, tokenizer = 'Word_tokenizer')
MAP_list = tune_hyperparameters(lowercase = True, tokenizer = 'Word_tokenizer')


# In[13]:


# 2nd model (filtering out with idf)
def bm25_query_idf(queries, queryname, bm25, lowercase = False, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
               tokenizer = 'RegExpTokeinzer'):
    
    query_scores_doc = defaultdict(lambda:0)
    
    for query in queries[queryname]:
        
        # using all terms 
        query_tokens = tokenize_document(query, lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, 
                                         stemmer = stemmer, tokenizer = tokenizer)
        
        
        filtered_query = sorted(query_tokens, key = lambda x : bm25.idf[x] if x in bm25.idf else 0, reverse = True)
                
        
        query_tokens = filtered_query[0: len(filtered_query)//2]
        
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


# In[24]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = False)
bm25 = BM25Okapi(list_tokens, k1= 1.5, b= 0.94)
rankings = bm25_query_idf(queries, "current_case_0001", bm25, lowercase = False, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
               tokenizer = 'RegExpTokeinzer')


# In[39]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True)
MAP = tune_hyperparameters(lowercase = True, model = 2)


# In[41]:


MAP


# In[8]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True)
MAP = tune_hyperparameters(lowercase = True, model = 2)


# In[62]:


#list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True)
bm25 = BM25Okapi(list_tokens, k1= 1.75, b= 0.95)
map, mrr, p10 = obtain_relevant_docs(queries, bm25, lowercase = True, tokenizer = 'Word_tokenizer')


# In[64]:


map, mrr, p10


# In[66]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True, tokenizer = 'Word_tokenizer'
                                            ,removeStopwords = False)
bm25 = BM25Okapi(list_tokens, k1= 1.75, b= 0.95)
map, mrr, p10 = obtain_relevant_docs(queries, bm25, lowercase = True, removeStopwords = False, tokenizer = 'Word_tokenizer')


# In[69]:


map, mrr, p10


# In[70]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True, tokenizer = 'Word_tokenizer'
                                            ,stemming = True)
bm25 = BM25Okapi(list_tokens, k1= 1.75, b= 0.95)
map, mrr, p10 = obtain_relevant_docs(queries, bm25, lowercase = True, stemming = True, tokenizer = 'Word_tokenizer')


# In[73]:


map, mrr, p10


# In[77]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True, tokenizer = 'Word_tokenizer'
                                            ,removeStopwords = False, stemming = True)
bm25 = BM25Okapi(list_tokens, k1= 1.75, b= 0.95)
map, mrr, p10 = obtain_relevant_docs(queries, bm25, lowercase = True, stemming = True, removeStopwords = False, tokenizer = 'Word_tokenizer')


# In[79]:


map, mrr, p10


# In[83]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True, tokenizer = 'Word_tokenizer'
                                            ,removeStopwords = False, stemming = False)
bm25 = BM25Okapi(list_tokens, k1= 1.75, b= 0.95)
map, mrr, p10 = obtain_relevant_docs(queries, bm25, lowercase = True, stemming = False, removeStopwords = False, tokenizer = 'Word_tokenizer')


# In[84]:


map, mrr, p10


# In[ ]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True, tokenizer = 'Word_tokenizer'
                                            ,removeStopwords = True, stemming = True)
bm25 = BM25Okapi(list_tokens, k1= 1.75, b= 0.95)
map, mrr, p10 = obtain_relevant_docs(queries, bm25, lowercase = True, stemming = True, removeStopwords = True, tokenizer = 'Word_tokenizer')


# In[85]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True, tokenizer = 'Word_tokenizer')
bm25 = BM25Okapi(list_tokens, k1= 1.75, b= 0.95)
map, mrr, p10 = obtain_relevant_docs(queries, bm25, lowercase = True, tokenizer = 'Word_tokenizer', model = 2)


# In[87]:


map, mrr, p10


# In[88]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases",  removeStopwords = False, lowercase = True, tokenizer = 'Word_tokenizer')
bm25 = BM25Okapi(list_tokens, k1= 1.75, b= 0.95)
map, mrr, p10 = obtain_relevant_docs(queries, bm25, lowercase = True, removeStopwords = False, tokenizer = 'Word_tokenizer', model = 2)


# In[90]:


map, mrr, p10


# In[14]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases",  removeStopwords = False, stemming = True, lowercase = True, tokenizer = 'Word_tokenizer')
bm25 = BM25Okapi(list_tokens, k1= 1.75, b= 0.95)
map, mrr, p10 = obtain_relevant_docs(queries, bm25, lowercase = True, removeStopwords = False, stemming = True, tokenizer = 'Word_tokenizer', model = 2)


# In[15]:


map, mrr, p10


# In[17]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases",  removeStopwords = True, stemming = True, lowercase = True, tokenizer = 'Word_tokenizer')
bm25 = BM25Okapi(list_tokens, k1= 1.75, b= 0.95)
map, mrr, p10 = obtain_relevant_docs(queries, bm25, lowercase = True, removeStopwords = True, stemming = True, tokenizer = 'Word_tokenizer', model = 2)


# In[18]:


map, mrr, p10


# In[46]:


def tune_citation_length():
    
    MAP_list = defaultdict(lambda:0)
    
    hyperparameters = np.arange(50, 501, 50)
    
    print(hyperparameters)
    for x in hyperparameters:
        print("length=", x)
        bm25 = BM25Okapi(list_tokens, k1= 1.75, b= 0.95)
        
        queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = x)
        
        map, mrr, p10, rec100 = obtain_relevant_docs(queries, bm25, lowercase = True, model = 1)
        
        MAP_list[x] = map
        
        print("MAP is:", map)
        print("MRR is:", mrr)
        print("MRR is:", p10)
    return MAP_list
        
        


# In[47]:


MAP_list = tune_citation_length()


# In[48]:


MAP_list


# In[16]:


bm25 = BM25Okapi(list_tokens, k1= 1.75, b= 0.95)
map, mrr, p10, rec100 = obtain_relevant_docs(queries, bm25, lowercase = True)


# In[17]:


map, mrr, p10, rec100


# In[ ]:





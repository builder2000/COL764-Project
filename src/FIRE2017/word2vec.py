#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
from gensim.models import Word2Vec

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# In[5]:


qrels = pd.read_csv("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\irled-qrel.txt", sep = ' ', header=None)


# In[6]:


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
        


# In[7]:


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
            

            


# In[8]:


list_tokens, indexes_of_doc = process_corpus("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", lowercase = True, tokenizer = 'Word_tokenizer')


# In[16]:


model = gensim.models.Word2Vec(list_tokens, min_count=1, vector_size = 100, epochs = 20)


# In[17]:


model.save("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\\word2_vec_trained_model.model")


# In[9]:


model = gensim.models.Word2Vec.load("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\\word2_vec_trained_model.model")


# In[10]:


# is this stopword list the correct one ? can we try other stopword collections or may be custom ?
stopword_set = set(stopwords.words('english'))
def stopword_removal(word_list):

    return [word for word in word_list if word not in stopword_set ]    


# do we make the case lower for legal documents?
def tf_idf_tokenize_document(doc, lowercase = False, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
                      tokenizer = 'RegExpTokeinzer', train_flag = True):
    
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
    token_dict = Counter(tokens)
    if(train_flag):
        for token in token_dict.keys():
            DF[token] +=1
        
    return token_dict


# In[11]:


def tf_idf_process_corpus(filename):
    corpus_files = os.listdir(filename)
    
    terms_dict = defaultdict(lambda: {})
    
    
    indexes_of_doc = defaultdict(lambda:0)
    
    count = 0
    for corpus_file in corpus_files:
        filepath = os.path.join(filename, corpus_file)
        
        indexes_of_doc[count] = corpus_file[:-4]
        print(corpus_file)
        
        with open(filepath, 'r') as f:
            doc_text = f.readlines()
                
            doc_text = ''.join(doc_text)
            
        tokens = tf_idf_tokenize_document(doc_text, lowercase = True, tokenizer = 'Word_tokenizer')
            
        terms_dict[count] = tokens
        
        count +=1
   
    return terms_dict, indexes_of_doc


# In[12]:


DF = defaultdict(lambda:0)
terms_dict, indexes_of_doc = tf_idf_process_corpus("D:\\Courses\\Sem 7 2021-22\\COL764\\COL764 Project\\FIRE2017-IRLeD-track-data\\Task_2\\Prior_Cases")


# In[13]:


def word_to_dict_map(DF):
    word_indices = {}
    count = 0
    for key, value in DF.items():
        word_indices[key] = count
        count +=1
        
    return word_indices
word_indices = word_to_dict_map(DF)


# In[14]:


def document_embeddings(model, terms_dict):
    N = len(terms_dict)
    
    embeddings = []
    for name, dict in terms_dict.items():
        document_name = indexes_of_doc[name]
        
        doc_embedding = np.zeros((100,))
        
        total_weights = 0
        for word, freq in dict.items():
            idf = idf = np.log2(1 + (N/ DF[word]))
            tf = np.log2(1 + freq)
            
            weight = tf*idf
            
            total_weights += weight
            
            
            doc_embedding += weight* model.wv[word]
            
        doc_embedding /= total_weights
        
        embeddings.append(doc_embedding)
    
    return embeddings
            


# In[15]:


doc_embeddings = document_embeddings(model, terms_dict)


# In[16]:


def findall(p, s):

    i = s.find(p)
    while i != -1:
        yield i
        
        i = s.find(p, i+1)
def process_query(doc):
    citation_marker = "[?CITATION?]"
    
    return findall(citation_marker, doc)


# In[17]:


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
    


# In[33]:


# remove the pattern "27\."
queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = 200)


# In[18]:


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

def word2vec_query(queryname, model, metric = 'cosine'):
    
    query_scores_doc = defaultdict(lambda:0)
    
    for query in queries[queryname]:
        query_dict =tf_idf_tokenize_document(query, lowercase = True, tokenizer = 'Word_tokenizer', train_flag = False)
        
        query_embedding = np.zeros((100,))
        
        total_weights = 0
        
        N = len(terms_dict)
        for word, freq in query_dict.items():
            if(DF[word] != 0):
                idf = idf = np.log2(1 + (N/ DF[word]))
                tf = np.log2(1 + freq)

                weight = tf*idf

                total_weights += weight
                query_embedding += weight* model.wv[word]
            
        query_embedding /= total_weights

        doc_scores = similarity_scores(doc_embeddings , query_embedding, metric = metric)
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


# In[19]:


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
    


# In[20]:


def obtain_relevant_docs(queries, model, flag = 1):
    
    querynames = queries.keys()
    
    
    list_AP = []
    list_MRR = []
    list_P10 = []
    for queryname in querynames:
        if(flag == 1):
            rankings = word2vec_query(queryname, model, metric = 'cosine')
        elif(flag == 2):
            rankings = word2vec_query(queryname, model, metric = 'euclidean')
        
        
        ap = AP(qrels, rankings, queryname)
        
        mrr = MRR(qrels, rankings, queryname)
        
        p10 = P_at_10(qrels, rankings, queryname)
        
        print(queryname, ap, mrr, p10)
        
        list_AP.append(ap[0])
        list_MRR.append(mrr)
        list_P10.append(p10)
        
    return np.mean(list_AP), np.mean(list_MRR) , np.mean(list_P10)


# In[40]:


# remove the pattern "27\."
queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = 150)
obtain_relevant_docs(queries, model, flag = 1)


# In[21]:


# remove the pattern "27\."
queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = 1000)
obtain_relevant_docs(queries, model, flag = 1)


# In[ ]:





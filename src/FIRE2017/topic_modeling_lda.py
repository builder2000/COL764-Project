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

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# In[2]:


qrels = pd.read_csv("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\irled-qrel.txt", sep = ' ', header=None)


# In[3]:


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


id2word = corpora.Dictionary(list_tokens)
texts = list_tokens

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


# In[7]:


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[8]:


lda_model.print_topics()


# In[ ]:


coherence_model_lda = CoherenceModel(model=lda_model, texts=list_tokens, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()


# In[ ]:


print(coherence_lda)


# In[18]:


def lda_vec(token_list):
    return  lda_model.get_document_topics(token_list, minimum_probability=0.0)

lda_vec1 = lda_vec(corpus[3])
lda_vec2 = lda_vec(corpus[5])


# In[10]:


sim = gensim.matutils.cossim(lda_vec1, lda_vec2)


# In[10]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics= num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=50,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print(num_topics, coherencemodel.get_coherence())

    return model_list, coherence_values


# In[48]:


model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=list_tokens, start=2, limit=40, step=6)


# In[11]:


def findall(p, s):

    i = s.find(p)
    while i != -1:
        yield i
        
        i = s.find(p, i+1)


# In[12]:


def process_query(doc):
    citation_marker = "[?CITATION?]"
    
    return findall(citation_marker, doc)


# In[13]:


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
    


# In[17]:


# remove the pattern "27\."
queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = 1000)


# In[16]:


# remove the pattern "27\."
full_queries = process_full_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases")


# In[25]:


def lda_vec(token_list):
    return  lda_model.get_document_topics(token_list, minimum_probability=0.0)
def prior_distributions(corpus):
    vectors = []
    for i in range(len(corpus)):
        vectors.append(lda_vec(corpus[i]))
        
    return vectors
        


# In[37]:


prior_dist = prior_distributions(corpus)


# In[26]:


def sim_metric(lda_vec1 , lda_vec2 , sim ='cosine'):
    if(sim == 'cosine'):
        return gensim.matutils.cossim(lda_vec1, lda_vec2)
    
    elif(sim == 'hellinger'):
        return gensim.matutils.hellinger(lda_vec1, lda_vec2)

    
def lda_scores(prior_dist,  query_tokens, sim = 'cosine'):
    #print(query_tokens)
    query_vec = lda_vec(query_tokens)

    scores = []
    for i in range(len(prior_dist)):
        score = sim_metric(query_vec, prior_dist[i], sim = sim)
        scores.append(score)
    
    return scores


def topical_model__query(queryname, lda_model , lowercase = True, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
               tokenizer = 'RegExpTokeinzer' , sim = 'cosine'):
    
    query_scores_doc = defaultdict(lambda:0)
    
    for query in queries[queryname]:
        
        # using all terms 
        query_tokens = tokenize_document(query, lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, 
                                         stemmer = stemmer, tokenizer = tokenizer)
        
        #print(query_tokens)
        query_tokens =  id2word.doc2bow(query_tokens)
        
        doc_scores = lda_scores(prior_dist, query_tokens, sim = sim)
        
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


def topical_model_full_query(queryname, lda_model , lowercase = True, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
               tokenizer = 'RegExpTokeinzer' , sim = 'cosine'):
    
    query_scores_doc = defaultdict(lambda:0)
    
    
    query = full_queries[queryname]
    
    query_tokens = tokenize_document(query, lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, 
                                         stemmer = stemmer, tokenizer = tokenizer)
    query_tokens =  id2word.doc2bow(query_tokens)
        
    doc_scores = lda_scores(prior_dist, query_tokens, sim = sim)
        
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


# In[27]:


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


# In[34]:


def obtain_relevant_docs(queries, lda_model, lowercase = True, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
                      tokenizer = 'RegExpTokeinzer', model = 1):
    
    querynames = queries.keys()
    
    
    list_AP = []
    list_MRR = []
    list_P10 = []
    list_Rec100 = []
    for queryname in querynames:
        if(model == 1):
            rankings = topical_model__query(queryname, lda_model , lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, stemmer = stemmer,
                              tokenizer = tokenizer, sim = 'cosine')
        elif(model == 2):
            rankings = topical_model__query(queryname, lda_model , lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, stemmer = stemmer,
                              tokenizer = tokenizer, sim = 'hellinger')
            
        
        ap = AP(qrels, rankings, queryname)
        
        mrr = MRR(qrels, rankings, queryname)
        
        p10 = P_at_10(qrels, rankings, queryname)
        
        rec100 = Rec_at_100(qrels, rankings, queryname)
        
        print(queryname, ap, mrr, p10, rec100)
        
        list_AP.append(ap[0])
        list_MRR.append(mrr)
        list_P10.append(p10)
        list_Rec100.append(rec100)
        
    return np.mean(list_AP), np.mean(list_MRR) , np.mean(list_P10), np.mean(rec100)


def obtain_relevant_fullq_docs(queries, lda_model, lowercase = True, removeStopwords = True, stemming = False, stemmer = 'PorterStemmer',
                      tokenizer = 'RegExpTokeinzer', model = 1):
    
    querynames = queries.keys()
    
    
    list_AP = []
    list_MRR = []
    list_P10 = []
    
    list_Rec100 = []
    for queryname in querynames:
        if(model == 1):
            rankings = topical_model_full_query(queryname, lda_model , lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, stemmer = stemmer,
                              tokenizer = tokenizer, sim = 'cosine')
        elif(model == 2):
            rankings = topical_model_full_query(queryname, lda_model , lowercase = lowercase, removeStopwords = removeStopwords, stemming = stemming, stemmer = stemmer,
                              tokenizer = tokenizer, sim = 'hellinger')
            
        
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


# In[35]:


# remove the pattern "27\."
full_queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = 400)
map, mrr, p10 , rec100 = obtain_relevant_docs(queries, lda_model, lowercase = True)


# In[36]:


map, mrr, p10, rec100


# In[21]:


map, mrr, p10 = obtain_relevant_docs(queries, lda_model, lowercase = True, model = 2)


# In[30]:


# Training on full data
map, mrr, p10, rec100 = obtain_relevant_fullq_docs(full_queries, lda_model, lowercase = True)


# In[31]:


map, mrr, p10, rec100


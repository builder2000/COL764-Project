#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased", output_hidden_states = True)


# In[9]:


import re
import os
import string
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch


# In[3]:


qrels = pd.read_csv("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\irled-qrel.txt", sep = ' ', header=None)


# In[4]:


def get_word_indeces(tokenizer, text, word):
    '''
    Determines the index or indeces of the tokens corresponding to `word`
    within `text`. `word` can consist of multiple words, e.g., "cell biology".
    
    Determining the indeces is tricky because words can be broken into multiple
    tokens. I've solved this with a rather roundabout approach--I replace `word`
    with the correct number of `[MASK]` tokens, and then find these in the 
    tokenized result. 
    '''
    # Tokenize the 'word'--it may be broken into multiple tokens or subwords.
    word_tokens = tokenizer.tokenize(word)

    # Create a sequence of `[MASK]` tokens to put in place of `word`.
    masks_str = ' '.join(['[MASK]']*len(word_tokens))

    # Replace the word with mask tokens.
    text_masked = text.replace(word, masks_str)

    # `encode` performs multiple functions:
    #   1. Tokenizes the text
    #   2. Maps the tokens to their IDs
    #   3. Adds the special [CLS] and [SEP] tokens.
    input_ids = tokenizer.encode(text_masked)

    # Use numpy's `where` function to find all indeces of the [MASK] token.
    mask_token_indeces = np.where(np.array(input_ids) == tokenizer.mask_token_id)[0]

    return mask_token_indeces


# In[10]:


def get_embedding(b_model, b_tokenizer, text, word=''):
    '''
    Uses the provided model and tokenizer to produce an embedding for the
    provided `text`, and a "contextualized" embedding for `word`, if provided.
    '''

    # If a word is provided, figure out which tokens correspond to it.
    if not word == '':
        word_indeces = get_word_indeces(b_tokenizer, text, word)

    # Encode the text, adding the (required!) special tokens, and converting to
    # PyTorch tensors.
    encoded_dict = b_tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        return_tensors = 'pt',  # Return pytorch tensors.
                        max_length = 512, 
                        truncation = True
                )

    input_ids = encoded_dict['input_ids']
    
    b_model.eval()

    # Run the text through the model and get the hidden states.
    bert_outputs = b_model(input_ids)
    
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():

        outputs = b_model(input_ids)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    # `hidden_states` has shape [13 x 1 x <sentence length> x 768]

    # Select the embeddings from the second to last layer.
    # `token_vecs` is a tensor with shape [<sent length> x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    
    # Convert to numpy array.
    sentence_embedding = sentence_embedding.detach().numpy()

    # If `word` was provided, compute an embedding for those tokens.
    if not word == '':
        # Take the average of the embeddings for the tokens in `word`.
        word_embedding = torch.mean(token_vecs[word_indeces], dim=0)

        # Convert to numpy array.
        word_embedding = word_embedding.detach().numpy()
    
        return (sentence_embedding, word_embedding)
    else:
        return sentence_embedding


# In[11]:


text = 'fbfkjb\nnflknf\nuid'
embedding = get_embedding(model, tokenizer, text, word='')


# In[104]:


text2 = 'gleeful'
embedding2 = get_embedding(model, tokenizer, text2, word='')


# In[105]:


np.inner(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))


# In[12]:


def sentence_embeddings(filename, model, tokenizer):
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
        
        doc_vec = get_embedding(model, tokenizer, doc_text, word='')
        
        prior_vec.append(doc_vec)

            
            
        count +=1
    
    return prior_vec , indexes_of_doc


# In[13]:


sent_embeddings, indexes_of_doc = sentence_embeddings("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Prior_Cases", model, tokenizer)


# In[14]:


def findall(p, s):

    i = s.find(p)
    while i != -1:
        yield i
        
        i = s.find(p, i+1)
def process_query(doc):
    citation_marker = "[?CITATION?]"
    
    return findall(citation_marker, doc)


# In[15]:


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
    


# In[87]:


# remove the pattern "27\."
queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = 400)


# In[17]:


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
        
        
def bert_query(queryname, model, tokenizer, metric = 'cosine'):
    
    query_scores_doc = defaultdict(lambda:0)
    
    for query in queries[queryname]:
        
        doc_vec = get_embedding(model, tokenizer, query , word='')
        
        doc_scores = similarity_scores(sent_embeddings, doc_vec, metric = metric)

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



# In[18]:


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
    


# In[19]:


def obtain_relevant_docs(queries, model, tokenizer, flag = 1):
    
    querynames = queries.keys()
    
    
    list_AP = []
    list_MRR = []
    list_P10 = []
    for queryname in querynames:
        if(flag == 1):
            rankings = bert_query(queryname, model, tokenizer, metric = 'cosine')
        elif(flag == 2):
            rankings = bert_query(queryname, model, tokenizer, metric = 'euclidean')
        
        ap = AP(qrels, rankings, queryname)
        
        mrr = MRR(qrels, rankings, queryname)
        
        p10 = P_at_10(qrels, rankings, queryname)
        
        print(queryname, ap, mrr, p10)
        
        list_AP.append(ap[0])
        list_MRR.append(mrr)
        list_P10.append(p10)
        
    return np.mean(list_AP), np.mean(list_MRR) , np.mean(list_P10)


# In[192]:


obtain_relevant_docs(queries, model, tokenizer, flag = 1)


# In[20]:


# remove the pattern "27\."
queries = process_queries("D:\Courses\Sem 7 2021-22\COL764\COL764 Project\FIRE2017-IRLeD-track-data\Task_2\Current_Cases", length = 150)


# In[21]:


obtain_relevant_docs(queries, model, tokenizer, flag = 1)


# In[ ]:





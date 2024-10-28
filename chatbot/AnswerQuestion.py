import numpy as np
import spacy
from bert_serving.client import BertClient
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from scipy.spatial.distance import cdist
import sys
sys.path.append('../../MyModules/')
import KleptoFunctions as kf

# NLP libraries
nlp = spacy.load("en_core_web_sm")

# vectorization tools
model_file = 'enwiki_dbow/doc2vec.bin'
enwiki = get_tmpfile(model_file)
enwiki_model = Doc2Vec.load(enwiki)

# Start Bert server in terminal
# bert-serving-start -model_dir /tmp/uncased_L-24_H-1024_A-16/
bc = BertClient()

# already vectorized documents
page_vectors = kf.puking_file('page_vectors_dict', '../kf_data/')
sent_vectors = kf.puking_file('sent_vectors_dict', '../kf_data/')

# process question
def nlp_text(page):
    text = nlp(page)
    tokens = [token.lemma_ for token in text]
    return tokens

def vectorize_tokens(model, tokens):
    model.random.seed(42)
    vector = model.infer_vector(tokens)
    return vector

# score documents
def score_page_to_any_question(vectors_dict, model, q_string):
    q_tokens = nlp_text(q_string)
    q_vector = np.array(vectorize_tokens(model, q_tokens)).reshape(1,-1)
    for filename in vectors_dict.keys():
        vector = vectors_dict[filename]['vector']
        score = 1-cdist(q_vector, vector, 'cosine')
        vectors_dict[filename]['score'] = score
    return vectors_dict

def get_related_documents(scores_dict, cutoff):
    flatten_scores = {}
    for index, filename in enumerate(score_dict.keys()):
        flatten_score[filename] = scores_dict[filename]['score']
    sliced_sorted_pagenames = sorted(\
                                     flatten_page_scores, 
                                     key=flatten_page_scores.get, 
                                     reverse = True)[:cutoff]
    return flatten_page_scores, sliced_scorted_pagenames

# score sentences
def score_sent_to_any_question(vectors_dict, q_string):
    q_vector = bc.encode([q_string])
    for pagename in vectors_dict.keys():
        for sent_i in vectors_dict[pagename].keys():
            for version in ['nonNLP', 'NLP']:
                text = vectors_dict[pagename][sent_i][version]['text']
                if len(text) >0:
                    vector = vectors_dict[pagename][sent_i][version]['vector']
                    score = 1-cdist(q_vector, vector, 'cosine')
                    vectors_dict[pagename][sent_i][version]['score'] = score
                else:
                    vectors_dict[pagename][sent_i][version]['score'] = 0.0
    return vectors_dict

def what_is_the_answer(scores_dict, version):
    flatten_scores = {}
    for docname in scores_dict.keys():
        for sent_i in scores_dict[docname].keys():
            flatten_scores['{}__{}'.format(docname, sent_i)] = \
            scores_dict[docname][sent_i][version]['score']
    answer_key = max(flatten_scores, key=flatten_scores.get)
    answer_score = flatten_scores[answer_key]
    print('Score: ', answer_score)
    
    answer_doc = answer_key.split('__')[0]
    answer_sent = int(answer_key.split('__')[-1])
    answer_text = scores_dict[docname][sent_i]['nonNLP']['text']
    
    answer_set = {
        'score': answer_score,
        'doc': answer_doc,
        'sent_index': answer_sent,
        'sent_text': answer_text}
    return answer_set, flatten_sent_scores

# Get answer as a string
def answer_any_question(q_string, page_vectors, sent_vectors):
    pretrained_doc_model = enwiki_model
    relevance_cutoff = 50
    page_scores = score_page_to_any_question(page_vectors, pretrained_doc_model, q_string)
    flatten_page_scores, relevant_filenames = get_related_documents(page_scores, relevance_cutoff)
    relevant_sent_vectors = {k:v for k, v in sent_vectors.items() if k in relevant_filenames}
    
    sent_scores = score_sent_to_any_question(relevant_sent_vectors, q_string)
    answer_set, flatten_sent_scores = what_is_the_answer(sent_scores, 'nonNLP')
    
    return answer_set, flatten_page_cores, flatten_sent_scores, sent_scores

# Get other possible answers as strings
def get_other_answers(flatten_sent_scores, scores_dict, cutoff):
    sliced_sorted_sentnames = sorted(flatten_sent_scores, key=flatten_sent_scores.get,
                                     reverse=True)[1:cutoff+1]
    other_answers = []
    for s in sliced_sorted_sentnames:
        answer_score = flatten_sent_scores[s]
        answer_doc = s.split('__')[0]
        answer_sent = int(s.split('__')[-1])
        answer_text = score_dict[answer_doc][answer_sent]['nonNLP']['text']
        answer_set = {
            'score': answer_score,
            'doc': answer_doc,
            'sent_index': answer_sent,
            'sent_text': answer_text}
        other_answers.append(answer_set)
    return other_answers

def string_page(pagename):
    if 'COLUMNS' in pagename:
        doc = '_'.join(pagename[:-12].split('_')[:-1])+'.pdf'
        page = pagename[:-12].split('_')[-1][4:]
    else:
        doc = pagename
        page = '01'
    page_string = '• Doc: {}\tPage: {}'.format(doc, page)
    return page_string

def string_answer(answer_set):
    text=answer_set['sent_text']
    page_string = string_page(answer_set['doc'])
    sentence = answer_set['sent_index']
    answer_string = '• "{}"\n{}\tSentence: {}'.format(text, page_string, sentence)
    return answer_string

# get other possible pages as strings
def get_other_pages(flatten_page_scores, cutoff):
    sliced_sorted_pagename = sorted(flatten_page_scores,
                                    key=flatten_page_scores.get,
                                    reverse=True)[:cutoff]
    return sliced_sorted_pagenames

# get all string for any question
def make_slack_answers(q_string):
    answer_set, flatten_page_scores, flatten_sent_scores, sent_scores = answer_any_question(q_string, page_vectors, sent_vectors)
    answer_string = string_answer(answer_set)
    
    other_answers = get_other_answers(flatten_sent_socres, sent_scores, 3)
    other_answer_string = '\n\n'.join([string_answer(s) for s in other_answers])
    
    other_pages = get_other_pages(flatten_page_scores, 2)
    other_pages_string = '\n\n'.join([string_page(p) for p in other_pages])
    
    return answer_string, other_answer_string, other_pages_string


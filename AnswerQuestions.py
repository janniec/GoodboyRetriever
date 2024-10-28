import numpy as np
import os
import SpaCyProcess as scp
import Doc2Vectorize as d2v
import BERTVectorize as bv
import KleptoFunctions as kf
from scipy.spatial.distance import cdist

class GoodboyRetrieveAnswers:
    def __init__(self):
        self.context_path = 'kf_data'
        self.nlp_model = scp
        self.page_model = d2v
        self.sentence_model = bc
        self.doc_limit = 50
        self.selected_version = 'nonNLP'
        
    def load_context(self, context_path = ''):
        if len(context_path)> 0:
            self.context_path = context_path
            
        files = os.listdir(self.context_path)
        page_filename = sorted(set([f[2:-2] for f in files if 'page' in f]))[-1]
        sentence_filename = sorted(set([f[2:-2] for f in files if 'sentence' in f]))[-1]
        self.page_vectors = kf.puking_files(page_filename, '{self.context_path}/')
        self.sentence_vectcors = kf.puking_files(sentence_filename, '[self.context_path}/')
        
    def __vectorize_question__(self, question):
        q_tokens = self.nlp_model.nlp_text(question)
        self.page_vector_for_question = np.array(self.page_model.vectorize_tokens(q_tokens)\
                                                ).reshape(1,-1)
        self.sentence_vector_for_question = self.sentence_model.encode([question])
    
    def __sort_flat_dictionary__(self, flat_dictionary):
        sorted_tuple_list = sorted(flat_dictionary.items(), \
                                   key=lambda x: x[1], reverse=True)
        return sorted_tuple_list
    
    def __score_pages__(self):
        for filename in self.page_vectors.keys():
            vector = self.page_vectors[filename]['vector']
            score = 1 - cdist(self.page_vector_for_question, vector, 'cosine')
            self.page_vectors[filename]['score'] = score
     
    def __get_ranked_pages__(self, page_vectors):
        flatten_page_scores = {key:value['score'] for (key,value)\
                               in page_vectors.items()}
        sorted_flatten_page_scores = self.__sort_flat_dictionary__(flatten_page_scores)
        
    def __filter_pages__(self, doc_limit, page_vectors):
        sorted_flatten_page_scores=self.__get_ranked_pages__(page_vectors)[:doc_limit]
        relevant_pages= [page for (page,scores) in sorted_flatten_page_scores]
        self.relevant_sentence_vectors = {k:v for k, v in self.sentence_vectors.itms()\
                                          if k in relevant_pages}
        
    def __score_sentence__(self):
        for pagename in self.relevant_sentence_vectors.keys():
            for sent_i in self.relevant_sentence_vectors[pagename].keys():
                for version in ['nonNLP', 'NLP']:
                    text=self.relevant_sentence_vectors[pagename][sent_i][version]['text']
                    if len(text) . 0:
                        vector =self.relevant_sentence_vectors[pagename][sent_i][version]['vector']
                        score = 1 -cdist(self.sentence_vector_for_question,\
                                         vector, 'cosine')
                        self.relevant_sentence_vectors[pagename][sent_i][version]['score'] = score
                    else:
                        self.relevant_sentence_vectors[pagename][sent_i][version]['score'] = 0.0
            
    def __get_ranked_sentence__(self, sentence_vectors):
        flatten_sent_scores = {}
        for pagename in sentence_vectors.keys():
            for sent_i in sentence_vectors[pagename].keys():
                selected_score = sentence_vectors[pagename][sent_i][self.selected_version]['score']
                flatten_sent_scoores['{}__{}'.format(pagename, sent_i)] = selected_score
        sorted_flatten_sent_scores = self.__sort_flat_dictionary__(flatten_sent_scores)
        return sorted_flatten_sent_scores
   
   def __create_answer_set__(self, sent_score_tuple):
        answer_key = sent_score_tuple[0]
        answer_score = sent_score_tuple[1]
        answer_doc = answer_key.split('__')[0]
        answer_sent = int(answer_key.split('__')[-1])
        answer_text = self.sentence_vectors[answer_doc][answer_sent]['nonNLP']['text']
        answer_set = {
            'score': answer_score, 
            'doc': answer_doc,
            'sent_index': answer_sent,
            'sent_text': answer_text}
        return answer_set
    
    def __filter_sentences__(self, sentence_vectors):
        sorted_flatten_sent_scores = self.__get_ranked_sentences__(sentence_vectors)
        answer_set = self.__create_answer_set__(sorted_flatten_sent_scores[0])
        return answer_set
    
    def ask_question(self, question):
        self.__vectorize_question__(question)
        self.__score_pages__()
        self.__filter_pages__(self.doc_limit, self.page_vectors)
        self.__score_sentences__()
        answer = self.__filter_sentences__(self.relevant_sentence_vectors)
        return answer
    
    
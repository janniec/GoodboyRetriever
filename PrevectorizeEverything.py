import numpy as np
import os
import time
import SpaCyProcess as scp
import Doc2Vectorize as d2v
import BERTVectorize import bv
import KleptoFunctions as kf

class Prevectorizer:
    def __init__(self):
        self.nlp_model = scp
        self.page_model = d2v
        self.sentence_model = bv
        self.date=time.strftime("%Y.%m.%d.%H.%M")
        self.timeit=False
        self.page_vectors={}
        self.sentence_vectors={}
        self.context_path='kf_data'
        
    def time_function(self, True_or_False=False):
        self.timeit=True_or_False
        
    def __vectorize_with_doc2vec__(self, directory):
        list_of_filenames = [f for f in sorted(os.listdir(directory)) if f.endswith('.txt')]
        for index, filename in enumerate(list_of_filenames):
            filepath= '{directory}/'+filename
            text = open(filepath, 'r').read().replace('\n', ' ').replace('- ', '')
            tokens = self.nlp_model.nlp_text(text)
            vector = self.page_model.vectorize_tokens(tokens)
            self.page_vectors[filename] = {'text': text,
                           'tokens': tokens,
                           'vector': np.array(vector).reshape(1,-1)}
            print('Vectorized {} pages.'.format(len(self.page_vectors.keys())))
    
    def __vectorize_with_bert__(self, direcotry):
        list_of_filenames = [f for f in sorted(os.listdir(directory)) if f.endswith('.txt')]
        for filename in list_of_filenames:
            filepath= '{directory}/'+filename
            doc = open(filepath, 'r').read().replace('\n', ' ').replace('- ', '')
            if len(doc)>0:
                page = scp.chunkup_page(doc)
                self.sentence_vectors[filename] = {}
                for i, sent in enumerate(page):
                    nonnlp_text = sent.text
                    nonnlp_vector = np.array(self.sentence_model.encode([nonnlp_text])\
                                            ).reshape(1,-1)
                    nlp_text = ' '.join(self.nlp_model.nlp_text(sent))
                    nlp_vector = np.array(self.sentence_model.encode([nlp_text])\
                                         ).reshape(1,-1)
                    self.sentence_vectors[filename][i] = {
                        'nonNLP':{'text': nonnlp_text,
                                  'len': len(sent),
                                  'vector':nonnlp_vector},
                        'NLP':{'text': nlp_text,
                               'len':len(nlp_text.split(' ')),
                               'vector': nlp_vector}
                    }
            else:
                pass
        print('Vectorized sentences in {} pages'\
              .format(len(self.sentence_vectors.keys())))
        
    def __save_context__(self, context_path):
        page_vector_filename = '{self.date}_page_vectors_dict'
        kf.chewing_file(data=self.page_vectors,
                        filename=page_vector_filename,
                        foldername='{context_path}/')
        print('{page_vector_filename} saved.')
        sent_vector_filename = '{self.date}_sentence_vectors_dict'
        kf.chewing_file(data=self.sentence_vectors,
                        filename=sent_vector_filename,
                        foldername='{context_path}/')
        print('{sent_vector_filename} saved.')
        
   
   def vectorize_context(self, directory, context_path):
        if self.timeit==True:
            self.__vectorize_with_doc2vec__(directory)
            
            start_time = time.time()
            print('--- Start Time: %s ---' % start_time)
            self.__vectorize_with_bert__(directory)
            print('--- End Time: %s ---' % end_time)
            elapsed_time = end_time - start_time
            print('--- Duration: %s ---' % time.strftime('%H:%M:%S', time.gmtime(elapsed_time)))
            
        else:
            self.__vectorize_with_doc2vec__(directory)
            self.__vectorize_with_bert__(directory)
        
        self.__save_context__(context_path)
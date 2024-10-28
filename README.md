# GoodboyRetriever  
GoodboyRetriever was a pet (pun intended!) project of mine from 2020. I had been waiting to share him until he was further developed, but ... 2020. I've finally decided to just clean him up and post.   
  
![GoodboyRetriever](https://github.com/janniec/GoodboyRetriever/blob/main/images/golden_retriever.jpg)  
image source: [Wag!](https://wagwalking.com/training/train-a-golden-retriever-to-fetch)    
  
GoodboyRetriever is a customizable information retrieval model (a RAG without text generation) utilizing pretrained embeddings from large language models (LLMs) and deployable to Slack. Instead of a chatbot, he was intended to be a reference tool customizable to any knowledge base. When GoodboyRetriever is fed a prompt, he will return the most relevant sentences from that knowledge base.  Sometimes you want to see the source, not read a curated response from a chat bot. Possible applications I had in mind were...   
- retrieve all passages related to a symptom from a collection of medical records to review a patient's history with that health condition  
- retrieve the paragraph about pet insurance from an employer's benefit handbook so that an employee can refresh their memory on a vet visit co-pay  
- retrieve an instruction page from a car manual to fix something    
  
## GoodboyRetriever is basically RAG  
![RAG-ish](https://github.com/janniec/GoodboyRetriever/blob/main/images/RAG_WIP.png)  
image source: [Medium](https://ai.plainenglish.io/a-brief-introduction-to-retrieval-augmented-generation-rag-b7eb70982891) 
  
RAG (Retrieval-Augmented Generation) is a cost-effective system to improve the output of LLMs, allowing them to reference knowledge outside of their training data, and focusing the LLMs to specific domains without the need to retrain the model. The key steps of a RAG are:    
1. Prepare a corpus of text data as a searchable knowledge base, transforming raw data into consistent "documents".
2. Create a vector database through data extraction, data chunking, vectorizing the chunks using embeddings.
3. Set up a retrieval process, where the system can ingest a query from the user, search the vector database and retrieve relevant context.
4. Send the query and context to an LLM for response generation.   
  
I developed GoodboyRetriever to do all of these steps except the last. Instead he'll send the context back to the user. 
  
For more information on RAGs, click [here](https://aws.amazon.com/what-is/retrieval-augmented-generation/).  
    
  
  
## How does Goodboy Retrieve Answers?  
  
1. GoodboyRetriever needs to be fed a corpus of text data as the knowledge base.  
2. The corpus will be chunked into pages and sentences. Chunking is performed hierarchically, utilizing SpaCy's POS (part of speech), punctuation, and lastly a limit on the number of tokens.  
3. The pages will be vectorized using a pretrained Doc2Vec embeddings based on wikipedia data.  
4. The sentences will be vectorized using an engine running on a large pretrained BERT model.   
5. All the page vectors and sentences vectors are stored using Klepto.  
6. When GoodboyRetriever is fed a query, it will vectorize the query twice. Once with Doc2Vec to produce a "page vector" version of the query. Then, with the BERT engine to produce a "sentence vector" version of the query.   
7. Using the page vectors from the corpus, GoodboyRetriever will find the 50 most relevant pages to the query's page vector using cosine similiarity between vectors.   
8. Only considering the sentence vectors associated with those 50 pages, GoodboyRetriever will then find the most relevant sentence to the query's sentence vector using cosine similarity between the vectors. 
9. GoodboyRetriever will send the most relevatn sentence to the user as an answer to the query.  
  
### Why Doc2Vec?  
To optimize GoodboyRetriever to function in near real time, I wanted him to first narrow the focus to a few relevant pages before it looked through the sentences. For this, I leveraged a publically available pretrained Doc2Vec DBOW embeddings trained on wikipedia text.    
  
![Doc2Vec](https://github.com/janniec/GoodboyRetriever/blob/main/images/doc2vec.png)  
image source: [Medium](https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e) 
  
Doc2Vec is deep learning model that learned vector representations (aka document embeddings) of documents, mapping each document, regardless of length, to a fixed-length vector in a high-dimensional space.   
  
There are 2 variations of the Doc2Vec model. 

- A Distributed Bag of Words (DBOW) Doc2Vec model starts with a unique Document ID for each document. The words in the document are not inputs but labels.  The model will learn to predict the probability of each word in the document based on the document vector only. The embeddings from this model treat the documents as a "bag of words" without capturing the order of the words. But these embeddings are simplier and faster to train.   
- In a Distributed Memory (DM) Doc2Vec model, words from the documents and a unique Document ID for each document are the inputs. The DM model will create a unique vector representation for each document based on the order of the context words. This approach allows the vectors to capture the semantic meaning of documents.  
   
For GoodboyRetriever, embeddings from a DM model would have been preferred, but none were available.  As next steps, I'd like to train a DM Doc2Vec model myself instead of relying on a pretrained DBOW model.  
  
For more information on Doc2Vec, click [here](https://www.geeksforgeeks.org/doc2vec-in-nlp/).  
  
### Why BERT?  
GoodboyRetriever is ultimately performing a sentence-level task. And for those, BERT is still the most successful and widely used LLM and a good place to start.   
  
![BERT Masked Language Model](https://github.com/janniec/GoodboyRetriever/blob/main/images/BERT_MLM.png)  
image source: [AnalyticsVidhya](https://www.analyticsvidhya.com/blog/2022/02/analyzing-semantic-equivalence-of-sentences-using-bert/) 
  
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based deep learning model that uses attention to capture the relationship between words without sequential processing. It produces vector representations of sentences (aka sentence embeddings) via 2 tasks in the training process.     
 - Masked Lanuage Model -  BERT takes in 2 inputs, the words in a sentence and a unique identifier for the sentence, a CLS token. It learns to predict missing words in a sentence by considering the surrounding words from both directions.
 - Next Sentence Prediction - At the same time, BERT is trained to predict whether 2 sentences will appear consecutively or not. The "words in a sentence" input I just mentioned above is actually the words from 2 sentences with a separater token in between.  
  
![BERT Next Sentence Prediction](https://github.com/janniec/GoodboyRetriever/blob/main/images/BERT_NSP.jpg)  
image source: [GeeksforGeeks](https://www.geeksforgeeks.org/understanding-bert-nlp/)   
   
As a result of these simultaneous tasks, the learned weights associated with the CLS token become the vector representations of sentences that capture the syntactic and semantic structures.   
   
Training a BERT model for custom embeddings is not recommended as it requires massive amounts of data and is a long and expensive process. But fine tuning an pretrained model for specific tasks is approachable and something I'd like to tackle as next steps.    
   
For more information on BERT, click [here](https://medium.com/@shaikhrayyan123/a-comprehensive-guide-to-understanding-bert-from-beginners-to-advanced-2379699e2b51) and [here](https://osanseviero.github.io/hackerllama/blog/posts/sentence_embeddings/).
  
  
  
## Modules  
* [kleptoFunctions.py](https://github.com/janniec/too_big_to_.pkl) (not included here) - Functions utilizing Kelpto to save dictionaries, lists, and dataframes in batches.  
* OCRpipeline.py - For consistent & wrapped text, OCR pdfs and images using AWS's textract in column mode. 
SpaCyProcess.py - Functions utilizing SpaCy to preprocess text data and hierarchically chunk based on POS (Part of speech) and tokenization.  
* Doc2Vectorize.py - Function utilizing a pretrained Doc2Vec embedding to vectorize pages.  
* BERTVectcorize.py - Function utilizing a pretrained BERT embedding to vectorize sentences.  
* PrevectorizeEverything.py - Class to vectorize all pages with Doc2Vec, vectorize all sentences with BERT and save organized dictionaries containing text and vectors with Klepto.  
* AnswerQuestions.py - Class to vectorize the query, unpack the vectors, filter the pages to the most relevant subset, and find the sentence "closest" to the query.   
* AnswerQuestion.py (in chatbot folder) - Unpacks dictionaries of vectors using Klepto, vectorizes querys Doc2Vec & BERT, filters pages, and finds closest sentence.   
* slackbot.py (in chatbot folder) - Runs the GoodboyRetriever engine as a slackbot in a slack workspace by typing `python slackbot.py` in terminal.  
  

  
# Set up  
  
## Virtual Enironment  
I used [pipenv](https://pipenv.pypa.io/en/latest/) to set up a virtual environment to test the requirements for just the PrevectorizeEverything.py and AnswerQuestion.py.     
Warning: I was able to install things fine but the 'locking' failed on gensim and scipy. So the 'Pipfile.lock' may not install everything you need.   
  
## Requirements
  
### To Prevectorize the Corpus  
* python 3.7  
* screen (To run the BERT server in the background. This is installed with `sudo apt install screen`. It might not be available on Mac.)  
* os   
* sys  
* pandas   
* numpy  
* time (You might not need this. I just timed my functions while experimenting.)  
* [spacy](https://spacy.io/usage)  
* [gensim](https://radimrehurek.com/gensim/models/doc2vec.html)  
* [bert-serving-server](https://github.com/hanxiao/bert-as-service)  
* bert-serving-client  
* [klepto](https://github.com/uqfoundation/klepto)  
* [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)  
  
### To OCR  
* [boto3](https://aws.amazon.com/blogs/machine-learning/automatically-extract-text-and-structured-data-from-documents-with-amazon-textract/) (NOTE: there's an error in the source code. Please use `from trp import Document`)  
* [pdf2image](https://github.com/Belval/pdf2image)  
* poppler (which might require conda)  
* json   
  
### To set up a Slackbot  
* [slackclient](https://medium.com/@nidhog/how-to-make-a-chatbot-on-slack-with-python-82015517f19c)  
* random  
  
   
  
## Pretrained Embeddings / Models (LLMs)  
Caution: There are likely some differences because a Ubuntu Subsystem on Windows vs a Mac. I will document how I setup these pretrained embeddings on my Ubuntu subsystem.   
  
### SpaCy    
#### To download & set it up   
1. Download the english model through the terminal.   
`python -m spacy download en_core_web_sm`   
or from pipenv, run this in the terminal  
`pipenv run python -m spacy download en_core_web_sm`
  
#### To use the pretrained model  
1. In your code load the english model into a function.   
`nlp = spacy.load("en_core_web_sm")`   
2. Switch out 'doc' with a string and process with the function.   
`spacy_doc = nlp(doc)`  

### Doc2Vec   
#### To download & set it up   
1. Click this [link](https://github.com/jhlau/doc2vec) to find a list of pretrained Doc2Vec embeddings.  
2. Scroll down to 'Pre-Trained Doc2Vec Models'.  
3. Click 'English Wikipedia DBOW (1.4GB)' link to go to 'sharepoint.com'.  
4. Click 'doc2vec' to download the 'enwiki_dbow.tgz.  
5. Move the 'enwiki_dbow' folder to your tmp directory.  
   
#### To use the pretrained embedding   
1. In your code, load the pretrained embedding into a variable with this code:   
```  
model_file = 'enwiki_dbow/doc2vec.bin'  
enwiki = get_tmpfile(model_file)  
enwikis_model = Doc2Vec.load(enwiki)  
```  
2. Switch out 'list_of_tokens' with a list of strings and infer a vector.  
```  
enwiki_model.random.seed(42)  
vector = enwiki_model.infer_vector(list_of_tokens)  
```  
NOTE: These pretrained embeddings have a random component. The model must be stablized with a seed before EVERY TIME its used to infer a vector.  

### BERT   
#### To download & set it up   
1. Click this [link](https://github.com/jina-ai/clip-as-service/tree/bert-as-service) to the Bert-as-service documentation.  
2. Scroll down to '1. Download a Pre-trained BERT Model' and click the dropdown.  
3. From the list, click 'BERT-Large, Uncased' to download 'uncased_L-24_H-1024_A-16.zip' zipfile.  
4. Move 'uncased_L-24_H-1024_A-16.zip' zipfile to your tmp directory.  
5. Unzip the zipfile in the tmp directory and discard the zipfile. You should be left with a 'uncased_L-24_H-1024_A-16' folder.  
  
#### To use the pretrained embedding  
In order to use bert-as-service, you need to keep the BERT server running. I prefer to run this in the background in a detached `screen` session.   
1. In the terminal, start a screen session.  
`screen`  
2. Start the BERT server.   
`bert-serving-start -model_dir /tmp/uncased_L-24_H-1024_A-16/`   
3. Detach that screen session. The BERT server will continue to run in the background.  
`ctrl` + `a` + `d`     
4. In your code, load the server into a variable.  
`bc = BertClient()`  
5. Switch out 'string' and encode a vector.   
`vector = bc.encode([string])`  
NOTE: The server is going to make a bunch of annoying 'tmp' folders. I don't know why. I don't know what for. After I'm done with the BERT server, I delete them.    
`rm -rf tmp*`  



# How to use GoodboyRetriever  
  
## to make the context  
```  
import AnswerQuestions as aq  
GBR = aq.GoodboyRetrieveAnswers()  
GBR.load_context()  
answer = GBR.ask_question('Put your question here?')  
print(answer)  
```  
  
## to ask the a question in the terminal  
```  
import PrevectorizeEverything as pe  
PV = pe.Prevectorizor()  
PV.__time_function__(True)  
PV.vectorize_context('OCR_output_directory', 'path/to/save/context')  
```  
  
## to run the slackbot  
The slackbot.py module will annoyingly/helpfully print slack events in the terminal. So I recommend running this in the background in a detached `screen` session.    
1. In the terminal, start a screen session.    
    `screen`   
2. Start the GoodboyRetriever slackbot.  
    `python slackbot.py`  
3. Detach that screen session. The slackbot app will continue to run in the background.  
    `ctrl` + `a` + `d`  
4. In your slack workspace locate the GoodboyRetriever app & ask your questions. 
  
 
  
# Next Steps   
![Retrieve more context](https://github.com/janniec/GoodboyRetriever/blob/main/images/retrieve_more.jpg)  
image source: [Pinterest](https://ru.pinterest.com/pin/847450854852769236/)  
  
- I'd like to standardize and automate some sort of an evaluation metric to measure GoodboyRetriever's performance as I update him. Maybe Precision@K or Mean Reciprocal Rank.   
- To improve performance, I'd like to add steps to train a Doc2Vec Distributed Memory (DM) model and fine tune the BERT embeddings.  
- I think more context around the answers would be helpful for the user. When returning the answer, I want to try including the sentence before and sentence after the answer.   
- Maybe I'll add text generation to clean up the returned answers to turn GoodboyRetriever into a RAG and give users a Chatbot experience.   
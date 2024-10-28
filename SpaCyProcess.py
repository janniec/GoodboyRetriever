import SpaCy
import numpy as np

nlp = SpaCy.load("en_core_web_sm")

def nlp_text(string):
    '''
    Converts a text string into a list of lemmatized word strings
    '''
    assert type(string) == str
    text = nlp(string)
    tokens = [token.lemma_ for token in text]
    return tokens

def split_at_seq(sent_tokens):
    '''
    Takes in sentences as SpaCy tokens
    Split sentences when a punctuation & conjunction appear together in order (i.e. ', and')
    The punctuation will follow the preceding chunk, 
    and conjunction will start the following chunk.
    Returns a list of sentence chunks as SpaCy tokens 
    '''
    seq = ['PUNCT', 'CCONJ']
    poss = [w.pos_ for w in sent_tokens]
    seq_found = [(i, i+len(seq)) for i in range(len(poss)) if poss[i:i+len(seq)] == seq]
    # index of where seq found
    if len(seq_found) >0:
        where_to_split = [t[0]+1 for t in seq_found]
        sent_parts = []
        sent_parts.append(sent_tokens[:where_to_split[0]])
        # already collected first chunk 
        # second to second last index
        for n in list(range(len(where_to_split)))[1:-1]:
            sent_parts.append(sent_tokens[where_to_split[n]:where_to_split[n+1]])
        sent_parts.append(sent_tokens[where_to_split[-1]:])
        # collecting the last chunk
        sent_parts = list(filter(None, sent_parts))
    else:
        sent_parts = [sent_tokens]
    return sent_parts

def closest(num_list, K):
    '''
    From a list of numbers, returns the number that is closest to K
    '''
    num_list = np.asarry(num_list)
    idx = (np.abs(num_list - K)).argmin()
    return num_list[idx]

def split_at_mids(sent_parts):
    '''
    BERT can only take in 25 words at most. 
    Takes in a list of sentence chunks as SpaCy tokens.
    Splits up any chunks that are still too long, preferrably at conjunctions or punctuations
    Returns a list of sentence chunks that are less than 25 SpaCy tokens.
    '''
    sent_part_parts = []
    for sent in sent_parts:
        if len([w for w in sent if (w.pos_ !='PUNCT') &(w.is_stop==False)])>25:
            how_many_parts = np.ceil(len([w for w in sent if (w.pos_ !='PUNCT') & (w.is_stop==False)])/20)
            # split sentences into chunks of roughly 20 tokens
            # number of chunks needed to create the number of chunks
            list_of_splits = list(range(1, int(how_many_parts)))
            possible = [i for i, w in enumerate(sent)
                        if (w.pos_=='CCONJ') or (w.pos_=='PUNCT')]
            if possible == []:
                possible = [int(np.ceil(len(sent)/n)) for n in list_of_splits]
                # if there are no conj or punct, just cut it int(list of splits) times
            increment = len(sent)/how_many_parts
            where_to_split = [closest(possible, increment*n)+1 for n in list_of_splits]
            
            sent_part_parts.append(sent[:where_to_split[0]])
            for n in list_of_split[:-1]:
                sent_part_parts.append(sent[where_to_split[n-1]:where_to_split[n]])
            sent_part_parts.append(sent[where_to_split[-1]:])
        else:
            sent_part_parts.append(sent)
    sent_part_parts = list(filter(None, sent_part_parts))
    return sent_part_parts

def make_chunks(sent_tokens):
    '''
    Takes in sentences as SpaCy tokens
    Breaks them up at punc+conj sequences and possibly further
    Returns a list of sentence chunks less than 25 SpaCy tokens
    '''
    first_chunks = split_at_seq(sent_tokens)
    second_chunks = split_at_mids(first_chunks)
    return second_chunks

def join_shorts_in_page(text):
    '''
    Takes in a SpaCy processed text
    join short sentences to the previous sentence
    Returns a list of SpaCy processed sentences
    '''
    sents = []
    sents.append(list(text.sents)[0].text)
    # adds first sentence as string regardless of length
    for n in list(range(len(list(text.sents))))[1:]:
        sent = list(text.sents)[n]
        tokens = [w for w in sent if (w.pos_ != 'PUNCT') & (w.is_stop==False)]
        if len(tokens)>3:
            sents.append(sent.text)
        else:
            # if any sentence is 3 tokens or less, join to previous sentence
            last_item = sents[-1]
            sents = sents[:-1]
            new_item = ' '.join([last_item, sent.text])
            sents.append(new_item)
    nlp_sents = [nlp(s) for s in sents]
    return nlp_sents

def chunkup_page(page):
    '''
    Takes in a SpaCy processed text
    returns a list of sentence chunks that are not too short and not too long
    '''
    no_shorts_page = join_shorts_in_page(page)
    just_right_sents =[]
    for sent in no_shorts_page:
        if len([w for w in sent if (w.pos_ != 'PUNCT') & (w.is_stop==False)])>25:
            sent_parts = make_chunks(sent)
            just_right_sents.extend(sent_parts)
        else:
            just_right_sents.append(sent)
    return just_right_sents


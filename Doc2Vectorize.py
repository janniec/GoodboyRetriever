from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile

model_file = 'enwiki_dbow/doc2vec.bin'
enwiki = get_tmpfile(model_file)
enwiki_model = Doc2Vec.load(enwiki)

def vectorize_tokens(tokens, models=enwiki_model):
    model.random.seed(42)
    vector = model.infer_vector(tokens)
    return vector
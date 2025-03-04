#!/usr/bin/env python
# coding: utf-8
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from stUtil import rndrCode

data = ["I love machine learning. Its awesome.", "I love coding in python", "I love building chatbots", "they chat amagingly well"]
tagged_data=[TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
max_epochs, vec_size, alpha=100, 20, .025
mdlLM = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=.00025, min_count=1, dm=1)
mdlLM.build_vocab(tagged_data)
mdlLM.save('colonUGI.mdlLM')
test_data = word_tokenize("I love chatbots".lower())
mdlLM.infer_vector(test_data)
similar_doc = mdlLM.docvecs.most_similar('1')
rndrCode(similar_doc)

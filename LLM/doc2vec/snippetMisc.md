from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
from gensim.test.utils import get_tmpfile
fname = get_tmpfile("my_doc2vec_model")
model.save(fname)
model = Doc2Vec.load(fname)  # you can continue training with the loaded model!
If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:

model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
Infer vector for a new document:

vector = model.infer_vector(["system", "response"])
*************************************************
doc2vec是指將句子、段落或者文章使用向量來表示，這樣可以方便的計算句子、文章、段落的相似度。

【二】使用方法介紹

1. 預料準備
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

def read_corpus(fname, tokens_only=False):
    with open(fname, encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if tokens_only:
                yield simple_preprocess(line)
            else:
                # For training data, add tags 利用gensim進行doc2vec時，語料庫是一個TaggedDocument，其包括原始語料（句子、段落、篇章）和對應的id（如句子id，段落id，篇章id）即語料標識
                yield TaggedDocument(simple_preprocess(line), [i])

2. 模型訓練

方法一：
from gensim.models.doc2vec import Doc2Vec

def train_doc2vec2():
    train_file = "E:/nlp_data/in_the_name_of_people/in_the_name_of_people.txt"
    train_corpus = list(read_corpus(train_file))
    mdlLM = Doc2Vec(documents=train_corpus, vector_size=50, min_count=2, epochs=10)
    mdlLM.save("doc2vec2.model")

方法二：

def train_doc2vec():
    train_file = "E:/nlp_data/in_the_name_of_people/in_the_name_of_people.txt"
    train_corpus = list(read_corpus(train_file))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=10)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs = model.epochs)
    model.save("doc2vec.model")

3. 模型使用

3.1 推測句子、段落或者文章的向量表示

model = doc2vec.Doc2Vec.load("doc2vec.model")
# 基於已有模型，來推測新文件或者句子或者段落的向量
print(model.infer_vector(["李達康是市委書記"]))

3.2 求解句子或者段落或者文章相似的內容

model = doc2vec.Doc2Vec.load("doc2vec2.model")
inferred_vector = model.infer_vector(["沙瑞金是省委書記"])
# 求解句子或者段落或者文章的相似性
sims = model.docvecs.most_similar([inferred_vector], topn=3)

train_file = "E:/nlp_data/in_the_name_of_people/in_the_name_of_people.txt"
train_corpus = list(read_corpus(train_file))
for docid, sim in sims:
    print(docid)
    print(sim)
    print(train_corpus[docid])
*****************************************************
DOC2VEC gensim tutorial Deepak Mishra
Today I am going to demonstrate a simple implementation of nlp and doc2vec. The idea is to implement doc2vec model training and testing using gensim 3.4 and python3. The new updates in gensim makes the implemention of doc2vec easier. Let’s start implementing #Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
Let’s prepare data for training our doc2vec model
data = ["I love machine learning. Its awesome.", "I love coding in python", "I love building chatbots", "they chat amagingly well"]

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
Here we have a list of four sentences as training data. Now I have tagged the data and its ready for training. Lets start training our model.
max_epochs, vec_size, alpha=100, 20, .025
mdlLM = Doc2Vec(size=vec_size, alpha=alpha, min_alpha=.00025, min_count=1, dm=1)
mdlLM.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    mdlLM.train(tagged_data, total_examples=mdlLM.corpus_count, epochs=mdlLM.iter)
    # decrease the learning rate
    mdlLM.alpha -= .0002
    # fix the learning rate, no decay
    mdlLM.min_alpha = mdlLM.alpha
mdlLM.save("d2v.mdlLM")
print("Model Saved")
Note: dm defines the training algorithm. If dm=1 means ‘distributed memory’ (PV-DM) and dm =0 means ‘distributed bag of words’ (PV-DBOW). Distributed Memory model preserves the word order in a document whereas Distributed Bag of words just uses the bag of words approach, which doesn’t preserve any word order.
So we have saved the model and it’s ready for implementation. Lets play with it.
from gensim.models.doc2vec import Doc2Vec

mdlLM= Doc2Vec.load("d2v.mdlLM")
#to find the vector of a document which is not in training data
test_data = word_tokenize("I love chatbots".lower())
v1 = mdlLM.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = mdlLM.docvecs.most_similar('1')
print(similar_doc)

# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(mdlLM.docvecs['1'])
So doc2vec is as simple as this. Hope you guys liked it. Feel free to comment . Here is link to my blog for older version of gensim, you guys can also view that. https://medium.com/@mishra.thedeepak/doc2vec-in-a-simple-way-fa80bfe81104
**********************************************
DocumentEmbeddingTechniquesAreviewOfnotableLiterature.html
class MyLabeledSentences(object):
    def __init__(self, dirname, dataDct={}, sentList=[]):
        self.dirname = dirname
        self.dataDct = {}
        self.sentList = []
    def ToArray(self):       
        for fname in os.listdir(self.dirname):            
            with open(os.path.join(self.dirname, fname)) as fin:
                for item_no, sentence in enumerate(fin):
                    self.sentList.append(LabeledSentence([w for w in sentence.lower().split() if w in stopwords.words('english')], [fname.split('.')[0].strip() + '_%s' % item_no]))
        return sentList

class MyTaggedDocument(object):
    def __init__(self, dirname, dataDct={}, sentList=[]):
        self.dirname = dirname
        self.dataDct = {}
        self.sentList = []
    def ToArray(self):       
        for fname in os.listdir(self.dirname):            
            with open(os.path.join(self.dirname, fname)) as fin:
                for item_no, sentence in enumerate(fin):
                    self.sentList.append(TaggedDocument([w for w in sentence.lower().split() if w in stopwords.words('english')], [fname.split('.')[0].strip() + '_%s' % item_no]))
        return sentList

sentences = MyLabeledSentences(some_dir_name)
model_l = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=7)
sentences_l = sentences.ToArray()
model_l.build_vocab(sentences_l )
for epoch in range(15): # 
    random.shuffle(sentences_l )
    model.train(sentences_l )
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model_l.alpha 

sentences = MyTaggedDocument(some_dir_name)
model_t = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=7)
sentences_t = sentences.ToArray()
model_l.build_vocab(sentences_t)
for epoch in range(15): # 
    random.shuffle(sentences_t)
    model.train(sentences_t)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model_l.alpha

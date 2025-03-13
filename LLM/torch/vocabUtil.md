from torchtext.vocab import build_vocab_from_iterator, Vectors
#from torchtext.data import Field, LabelField
from torch import long as trchLong
from torchtext.data.utils import get_tokenizer
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
symNdx=(UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX )
# Make sure the tokens are in order of their indices to properly insert them in vocab
特殊符 = ['<unk>', '<pad>', '<bos>', '<eos>']
象=dict(zip(特殊符, symNdx))    #象={'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}

def 計算彙(cntxt):
  TEXT = Field(tokenize=tokenizer, use_vocab=True, lower=True, batch_first=True, include_lengths=True)
  LABEL = LabelField(dtype=trchLong, batch_first=True, sequential=False)
  fields = [('text', TEXT), ('label', LABEL)]

  象彙 = get_tokenizer('spacy', language='en_core_web_sm')
  向量 = Vectors(name='glove.6B.50d.txt')
  TEXT.build_vocab(cntxt, vectors=向量, max_size=10000, min_freq=1)
  LABEL.build_vocab(cntxt)
  return TEXT, LABEL
**********************  製作vocab  *****************************
from torchtext.vocab import vocab, build_vocab_from_iterator
from collections import Counter, OrderedDict
from stUtil import rndrCode
from io import open as ioOPEN
url = {'42B':'http://nlp.stanford.edu/data/glove.42B.300d.zip', '840B':'http://nlp.stanford.edu/data/glove.840B.300d.zip', 'twitter.27B':'http://nlp.stanford.edu/data/glove.twitter.27B.zip', '6B':'http://nlp.stanford.edu/data/glove.6B.zip' }
pretrained_aliases = {"charngram.100d":partial(CharNGram), 
"fasttext.en.300d":partial(FastText, language="en"), 
"fasttext.simple.300d":partial(FastText, language="simple"), 
"glove.42B.300d":partial(GloVe, name="42B", dim="300"), 
"glove.840B.300d":partial(GloVe, name="840B", dim="300"), 
"glove.twitter.27B.25d":partial(GloVe, name="twitter.27B", dim="25"), 
"glove.twitter.27B.50d":partial(GloVe, name="twitter.27B", dim="50"), 
"glove.twitter.27B.100d":partial(GloVe, name="twitter.27B", dim="100"), 
"glove.twitter.27B.200d":partial(GloVe, name="twitter.27B", dim="200"), 
"glove.6B.50d":partial(GloVe, name="6B", dim="50"), 
"glove.6B.100d":partial(GloVe, name="6B", dim="100"), 
"glove.6B.200d":partial(GloVe, name="6B", dim="200"), 
"glove.6B.300d":partial(GloVe, name="6B", dim="300") }

def vocabGlove():
    examples = ['chip', 'baby', 'Beautiful']
    vec = text.vocab.GloVe(name='6B', dim=50)
    ret = vec.get_vecs_by_tokens(tokens, lower_case_backup=True)

def 產詞(file_path):
    with ioOPEN(file_path, encoding = 'utf-8') as fin:
      for line in fin:
        yield line.strip().split()

def vocabIter():
  #generating vocab from text file
  vocab = build_vocab_from_iterator(產詞(file_path), specials=["<unk>", '<CLS>', '<SEP>'])

def vocabDemo():
  from torchtext.vocab import vocab
  from collections import Counter, OrderedDict
  counter = Counter(["a", "a", "b", "b", "b"])
  sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=true)
  ordered_dict = OrderedDict(sorted_by_freq_tuples)
  v1 = vocab(ordered_dict)
  print(v1['a']) #prints 1
  print(v1['out of vocab']) #raise runtimeerror since default index is not set
  tokens = ['e', 'd', 'c', 'b', 'a']
  v2 = vocab(OrderedDict([(token, 1) for token in tokens]))
  #adding <unk> token and default index
  unk_token = '<unk>'
  default_index = -1
  if unk_token not in v2: v2.insert_token(unk_token, 0)
  v2.set_default_index(default_index)
  print(v2['<unk>']) #prints 0
  print(v2['out of vocab']) #prints -1
  #make default index same as index of unk_token
  v2.set_default_index(v2[unk_token])
  v2['out of vocab'] is v2[unk_token] #prints true


def mkVocab(TKN):
  counter = Counter(TKN)   #["a", "a", "b", "b", "b"]
  stCode(['counter=', counter])     #Counter({'is': 4, 'to': 3, 'the': 3, ',': 3, 'it': 3, 'input': 3, 'user': 2, 'i': 2, 'this': 2, 'and': 2, '.': 2, 'then': 2, 'which': 1, 'a': 1, 'way': 1, 'receive': 1, 'inputs': 1, 'from': 1, '!': 1, 'personally': 1, 'did': 1, 'not': 1, 'know': 1, 'how': 1, 'works': 1, 'before': 1, 'reading': 1, 'file': 1, 'frankly': 1, 'think': 1, 'an': 1, 'awesome': 1, 'feature': 1, 'in': 1, 'python': 1, 'basically': 1, 'what': 1, 'function': 1, 'does': 1, 'print': 1, 'string': 1, 'given': 1, 'by': 1, 'request': 1, 'raw_text': 1, 'encoded': 1, 'into': 1, 'tokens': 1, 'using': 1})
  sortedFreq=sorted(counter.items(), key=lambda x: x[1], reverse=True)
  彙表=OrderedDict(sortedFreq)
  stCode(['彙表=', 彙表])   #OrderedDict({'is': 4, 'to': 3, 'the': 3, ',': 3, 'it': 3, 'input': 3, 'user': 2, 'i': 2, 'this': 2, 'and': 2, '.': 2, 'then': 2, 'which': 1, 'a': 1, 'way': 1, 'receive': 1, 'inputs': 1, 'from': 1, '!': 1, 'personally': 1, 'did': 1, 'not': 1, 'know': 1, 'how': 1, 'works': 1, 'before': 1, 'reading': 1, 'file': 1, 'frankly': 1, 'think': 1, 'an': 1, 'awesome': 1, 'feature': 1, 'in': 1, 'python': 1, 'basically': 1, 'what': 1, 'function': 1, 'does': 1, 'print': 1, 'string': 1, 'given': 1, 'by': 1, 'request': 1, 'raw_text': 1, 'encoded': 1, 'into': 1, 'tokens': 1, 'using': 1})
  彙IDs = vocab(彙表)
  #Vocab(dictionary_items, unk_token=unk_replacement)
  stCode(['彙.vocab, TKN=', dir(彙IDs.vocab), TKN])#prints 1
  #彙.vocab==>'append_token', 'default_index_', 'get_default_index', 'get_itos', 'get_stoi', 'insert_token', 'itos_', 'lookup_indices', 'lookup_token', 'lookup_tokens', 'set_default_index'
  stCode(['get_itos=', 彙.get_itos()]) #['is', 'to', 'the', ',', 'it', 'input', 'user', 'i', 'this', 'and', '.', 'then', 'which', 'a', 'way', 'receive', 'inputs', 'from', '!', 'personally', 'did', 'not', 'know', 'how', 'works', 'before', 'reading', 'file', 'frankly', 'think', 'an', 'awesome', 'feature', 'in', 'python', 'basically', 'what', 'function', 'does', 'print', 'string', 'given', 'by', 'request', 'raw_text', 'encoded', 'into', 'tokens', 'using']
  #tokens = ['e', 'd', 'c', 'b', 'a']
  #adding <unk> token and default index
  symNdx=UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX=0, 1, 2, 3
  #(UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX )
  # Make sure the tokens are in order of their indices to properly insert them in vocab
  未知, 特殊符='<unk>', ['<unk>', '<pad>', '<bos>', '<eos>']
  象, 預設Ndx=dict(zip(特殊符, symNdx)), -1    #象={'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
  #特殊符=[未知]
  rndrCode(['象', 象])
  彙II = vocab(OrderedDict([(token, 1) for token in TKN]), specials=特殊符)     #[unk_token]
  彙II.set_default_index(預設Ndx)
  #make default index same as index of unk_token
  彙II.set_default_index(彙II[未知])
  rndrCode(['彙II=', 彙II.__dict__])
  彙II['out of vocab'] is 彙II[未知] #prints True
  rndrCode(['[unk]=', 彙II['<unk>']]) #prints 0
  rndrCode(['[out of vocab]=', 彙II['out of vocab']]) #prints -1

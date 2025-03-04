使用Keras和預訓練的詞向量訓練新聞文字分類模型
其他 · 發表 2019-02-20
from __future__ import print_function
import os
import sys
from numpy import array as npAsarray
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
BASE_DIR = "/data/machine_learning/"
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'news20/20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000  # 每個文字或者句子的截斷長度，只保留1000個單詞
MAX_NUM_WORDS = 20000  # 用於構建詞向量的詞彙表數量
EMBEDDING_DIM = 100  # 詞向量維度
VALIDATION_SPLIT = 0.2
"""
基本步驟：
1.資料準備：
預訓練的詞向量檔案:下載地址：http://nlp.stanford.edu/data/glove.6B.zip
用於訓練的新聞文字檔案:下載地址:http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
2.資料預處理
1)生成文字檔案詞彙表:這裡詞彙表長度為20000，只取頻數前20000的單詞
2)將文字檔案每行轉為長度為1000的向量，多餘的截斷，不夠的補0。向量中每個值表示單詞在詞彙表中的索引
3）將文字標籤轉換為one-hot編碼格式
4）將文字檔案劃分為訓練集和驗證集
3.模型訓練和儲存
1）構建網路結構
2）模型訓練
3）模型儲存
"""
# 構建詞向量索引
print("Indexing word vectors.")
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]  # 單詞
        coefs = npAsarray(values[1:], dtype='float32')  # 單詞對應的向量
        embeddings_index[word] = coefs  # 單詞及對應的向量
# print('Found %s word vectors.'%len(embeddings_index))#400000個單詞和詞向量
print('預處理文字資料集')
texts = []  # 訓練文字樣本的list
labels_index = {}  # 標籤和數字id的對映
labels = []  # 標籤list
# 遍歷資料夾，每個子資料夾對應一個類別
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    # print(path)
    if os.path.isdir(path):
        labels_id = len(labels_index)
        labels_index[name] = labels_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')  ##遮蔽檔案頭
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(labels_id)
print("Found %s texts %s label_id." % (len(texts), len(labels)))  # 19997個文字檔案
# 向量化文字樣本
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
# fit_on_text(texts) 使用一系列文件來生成token詞典，texts為list類，每個元素為一個文件。就是對文字單詞進行去重後
tokenizer.fit_on_texts(texts)
# texts_to_sequences(texts) 將多個文件轉換為word在詞典中索引的向量形式,shape為[len(texts)，len(text)] -- (文件數，每條文件的長度)
sequences = tokenizer.texts_to_sequences(texts)
print(sequences[0])
print(len(sequences))  # 19997
word_index = tokenizer.word_index  # word_index 一個dict，儲存所有word對應的編號id，從1開始
print("Founnd %s unique tokens." % len(word_index))  # 174074個單詞
# ['the', 'to', 'of', 'a', 'and', 'in', 'i', 'is', 'that', "'ax"] [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(list(word_index.keys())[0:10], list(word_index.values())[0:10])  #
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # 長度超過MAX_SEQUENCE_LENGTH則截斷，不足則補0
labels = to_categorical(npAsarray(labels))
print("訓練資料大小為：", data.shape)  # (19997, 1000)
print("標籤大小為:", labels.shape)  # (19997, 20)
# 將訓練資料劃分為訓練集和驗證集
indices = np.arange(data.shape[0])
np.random.shuffle(indices)  # 打亂資料
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
# 訓練資料
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
# 驗證資料
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
# 準備詞向量矩陣
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)  # 詞彙表數量
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))  # 20000*100
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:  # 過濾掉根據頻數排序後排20000以後的詞
        continue
    embedding_vector = embeddings_index.get(word)  # 根據詞向量字典獲取該單詞對應的詞向量
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# 載入預訓練的詞向量到Embedding layer
embedding_layer = Embedding(input_dim=num_words,  # 詞彙表單詞數量
                            output_dim=EMBEDDING_DIM,  # 詞向量維度
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,  # 文字或者句子截斷長度
                            trainable=False)  # 詞向量矩陣不進行訓練
print("開始訓練模型.....")
# 使用
sequence_input=Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')  # 返回一個張量，長度為1000，也就是模型的輸入為batch_size*1000
embedded_sequences = embedding_layer(sequence_input)  # 返回batch_size*1000*100
x = Conv1D(128, 5, activation='relu')(embedded_sequences)  # 輸出的神經元個數為128，卷積的視窗大小為5
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))
model.summary()
model.save("../data/textClassifier.h5")
模型結構如下：
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 1000)              0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 1000, 100)         2000000   
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 996, 128)          64128     
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 199, 128)          0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 195, 128)          82048     
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 39, 128)           0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 35, 128)           82048     
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512     
_________________________________________________________________
dense_2 (Dense)              (None, 20)                2580      
=================================================================
Total params: 2,247,316
Trainable params: 247,316
Non-trainable params: 2,000,000

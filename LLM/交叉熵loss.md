文本分析 神經網絡訓練 不要使用 one-hot encoding 採用適當的embedding 進行分類
在文本分析中，如果不使用 one-hot encoding，可以釆用詞嵌入（embedding）的方法進行神經網絡的訓練。以下是適合的流程：

1. 准備數據
文本清理：去除噪音（如標點、停用詞等），或者根據任務的需要保留特定信息。
分詞：將句子或段落分割成單詞，或在某些任務中直接處理字符級別的輸入。
2. 詞嵌入（Embedding）
詞嵌入通過將詞映射到一個固定長度的向量空間中來表示詞的語義。常用的嵌入方法有：

預訓練嵌入模型：使用預先訓練好的嵌入，例如 Word2Vec、GloVe 或 FastText。
自定義嵌入層：在神經網絡中使用嵌入層并通過訓練數據自動學習詞向量表示。
Keras/TensorFlow 中的 Embedding 層，PyTorch 中的 nn.Embedding 都可以用於這個目的。
3. 神經網絡架搆選擇
在分類任務中，合適的神經網絡架搆可以提高分類性能：

簡單模型：多層感知器（MLP）可以應用於短文本或特徵量較少的任務。
循環神經網絡（RNN）：處理時序數據（例如文本）的首選。LSTM 或 GRU 是經典的變種。
卷積神經網絡（CNN）：適用於捕捉局部特徵，也在文本分類中常用。
Transformer模型：如 BERT、GPT，能夠很好地處理長文本，并且在各種文本分類任務中取得了領先的效果。
4. 訓練流程
Embedding 初始化：將嵌入層初始化為預訓練的詞嵌入（如 Word2Vec 或 BERT），或隨機初始化后在任務中學習。
輸入文本嵌入：將每個詞或字符嵌入表示傳遞到神經網絡中。
分類層：在模型的最后一層，可以釆用 softmax（多分類）或 sigmoid（二分類）作為輸出層，輸出預測概率。
5. 損失函數和優化
損失函數：通常使用交叉熵損失函數（categorical_crossentropy 或 binary_crossentropy）。
優化器：常用優化器如 Adam、SGD、RMSprop，可以根據模型訓練時的表現進行調整。
實現示例（使用Keras）
python
複製程式碼
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# 假設我們有預處理后的訓練數據 X_train 和標簽 y_train

# 參數設定
vocab_size = 10000  # 詞匯表大小
max_length = 100  # 每條文本序列的最大長度
embedding_dim = 100  # 詞向量維度

# 建立一個簡單的LSTM分類器
model = Sequential()
# Embedding層：嵌入詞匯表中的單詞
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
# LSTM層：處理時間序列特徵
model.add(LSTM(128))
# 全連接層
model.add(Dense(1, activation='sigmoid'))  # 二分類問題

# 編譯模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
6. 評估模型
使用測試集進行模型的評估，報告准確率、F1 分數等關鍵分類指標。
如果使用預訓練的嵌入，可以比較與隨機初始化嵌入的性能差異。
這種方式有效避免了 one-hot encoding 的稀疏性問題，并且通過詞嵌入捕捉到詞語之間的語義信息。






tokenize 分詞 n-grams 並自動找出n-labels
要對文本進行分詞、生成 n-grams，并自動找出文本的 n 個類別標簽，可以按照以下步驟來實現。我們將分詞、n-grams 和多標簽分類的過程逐步細化。整個過程可以使用 Python 的 nltk、sklearn 和 Keras 等庫來完成。

1. 文本分詞和 n-grams 生成
在文本處理中，分詞是將文本切分為單獨的詞語，而n-grams是將連續的 n 個詞組合在一起，這可以捕捉到更丰富的上下文信息。

實現步驟：
分詞：可以使用 nltk 或 spaCy 來處理。
n-grams：可以使用 nltk 或 scikit-learn 的 CountVectorizer 或 TfidfVectorizer 生成。
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# 文本示例
corpus = [
    "機器學習是人工智能的一個分支",
    "神經網絡是一種機器學習的模型",
    "深度學習是機器學習的一個重要領域"
]

# 分詞
def tokenize(text):
    return nltk.word_tokenize(text)

# 生成 n-grams
vectorizer = CountVectorizer(ngram_range=(1, 3), tokenizer=tokenize)
X = vectorizer.fit_transform(corpus)

# 打印 n-grams 詞匯表
print(vectorizer.get_feature_names_out())
2. 自動識別文本的 n-labels（多標簽分類）
n-labels 分類是指文本可以屬於多個類別。對於多標簽分類問題，我們可以使用機器學習算法，結合詞向量和 n-grams 生成的特徵，來自動預測多個類別標簽。

文本向量化：使用 TfidfVectorizer 或 CountVectorizer 將文本轉換為特徵向量。
多標簽分類器：使用 scikit-learn 的 MultiLabelBinarizer 和多標簽分類算法，比如 RandomForestClassifier、LogisticRegression 或 Keras 中的神經網絡。
示例：使用 TfidfVectorizer 和多標簽分類

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# 假設我們有以下文本和對應的多標簽
texts = [
    "機器學習和深度學習是人工智能的一部分",
    "自然語言處理涉及文本分析和生成",
    "深度學習使用神經網絡來進行學習",
]

# 每個文本可能屬於多個類別
labels = [
    ['AI', 'ML', 'DL'],
    ['NLP', 'Text Analysis'],
    ['DL', 'NN']
]

# 將文本向量化
vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 3))
X = vectorizer.fit_transform(texts)

# 使用 MultiLabelBinarizer 進行標簽二值化
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 選擇分類模型（邏輯回歸 + 多輸出分類器）
model = MultiOutputClassifier(LogisticRegression())
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 查看預測結果
predicted_labels = mlb.inverse_transform(y_pred)
print(predicted_labels)
3. 使用神經網絡進行多標簽分類
我們也可以用深度學習模型來處理這個問題，尤其當數據規模較大時，神經網絡在文本分類任務中表現尤為強大。

Keras 實現多標簽分類

from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

# 示例數據
texts = [
    "機器學習和深度學習是人工智能的一部分",
    "自然語言處理涉及文本分析和生成",
    "深度學習使用神經網絡來進行學習",
]

labels = [
    ['AI', 'ML', 'DL'],
    ['NLP', 'Text Analysis'],
    ['DL', 'NN']
]

# 使用Tokenizer對文本進行分詞和序列化
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, padding='post', maxlen=10)

# 標簽二值化
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)

# 創建神經網絡模型
model = Sequential()
model.add(Dense(128, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(mlb.classes_), activation='sigmoid'))  # 使用 sigmoid 進行多標簽分類

# 編譯模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 訓練模型
model.fit(X, Y, epochs=10, batch_size=32)

# 預測新數據
new_texts = ["機器學習是人工智能"]
new_X = tokenizer.texts_to_sequences(new_texts)
new_X = pad_sequences(new_X, padding='post', maxlen=10)

# 預測類別
y_pred = model.predict(new_X)
predicted_labels = mlb.inverse_transform(y_pred > 0.5)  # 使用 0.5 作為閾值進行多標簽分類
print(predicted_labels)
4. 總結
分詞與 n-grams：使用 nltk、scikit-learn 的工具可以輕松生成 n-grams，捕捉文本的局部特徵。
多標簽分類：可以用邏輯回歸、隨機森林或深度學習模型進行多標簽分類。深度學習特別適合處理大規模數據集。
自動找出 n-labels：通過模型訓練，可以自動為每個文本預測多個標簽。
通過這種方法，可以結合 n-grams 特徵和神經網絡或傳統機器學習模型來實現多標簽分類任務。

深度學習模型進行多標簽分類 n-labels torch範例
使用 PyTorch 進行多標簽分類時，可以通過 sigmoid 激活函數為每個類別輸出一個概率值。每個類別的預測值獨立於其他類別，這使得我們可以實現多標簽分類任務。以下是一個 PyTorch 中實現多標簽分類的完整示例。

1. 數據准備
我們將假設有一組文本數據，每個文本對應多個類別標簽。首先需要對文本進行分詞、生成 n-grams，并將標簽二值化。然后，我們將使用神經網絡來訓練模型，并輸出多個類別標簽（n-labels）。

2. PyTorch 實現步驟 導入所需庫
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# 分詞工具
nltk.download('punkt')
def tokenize(text):
    return nltk.word_tokenize(text)
示例數據
假設我們有以下文本和對應的多標簽數據：

# 示例文本
texts = [
    "機器學習和深度學習是人工智能的一部分",
    "自然語言處理涉及文本分析和生成",
    "深度學習使用神經網絡來進行學習"
]

# 標簽，文本對應的多個標簽
labels = [
    ['AI', 'ML', 'DL'],
    ['NLP', 'Text Analysis'],
    ['DL', 'NN']
]
數據預處理
我們需要將文本轉換為向量，并將標簽二值化。

# 使用TfidfVectorizer進行特徵提取（n-grams）
vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(texts).toarray()

# 標簽二值化
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 轉換為 PyTorch 張量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
3. 定義深度學習模型
使用簡單的全連接神經網絡來實現多標簽分類。輸出層使用 sigmoid 激活函數，每個節點獨立輸出一個概率值。

class MultiLabelClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 隱藏層1
        self.fc2 = nn.Linear(128, 64)        # 隱藏層2
        self.fc3 = nn.Linear(64, output_dim)  # 輸出層
        self.sigmoid = nn.Sigmoid()          # 用於多標簽分類

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 激活函數為ReLU
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # 輸出為0~1之間的概率值
        return x
4. 訓練模型
我們使用二元交叉熵損失函數（BCELoss），它適用於多標簽分類問題，并使用 Adam 作為優化器。

# 模型參數
input_dim = X_train.shape[1]  # 輸入維度是TF-IDF特徵的數量
output_dim = y_train.shape[1]  # 輸出維度是標簽的數量

# 創建模型實例
model = MultiLabelClassificationModel(input_dim, output_dim)

# 定義損失函數和優化器
criterion = nn.BCELoss()  # 二元交叉熵損失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 10
batch_size = 2

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    # Mini-batch 梯度下降
    for i in range(0, X_train.size(0), batch_size):
        inputs = X_train[i:i + batch_size]
        labels = y_train[i:i + batch_size]
        
        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(X_train):.4f}")
5. 評估模型
我們可以通過在測試集上進行預測來評估模型的性能，使用 0.5 作為閾值來確定每個類別的標簽。

model.eval()  # 進入評估模式
with torch.no_grad():
    outputs = model(X_test)
    predicted = (outputs > 0.5).float()  # 使用 0.5 作為分類閾值
    print("Predicted Labels:")
    print(predicted)

    print("True Labels:")
    print(y_test)
6. 完整代碼總結
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# 分詞工具
nltk.download('punkt')
def tokenize(text):
    return nltk.word_tokenize(text)

# 示例文本
texts = [
    "機器學習和深度學習是人工智能的一部分",
    "自然語言處理涉及文本分析和生成",
    "深度學習使用神經網絡來進行學習"
]

# 標簽，文本對應的多個標簽
labels = [
    ['AI', 'ML', 'DL'],
    ['NLP', 'Text Analysis'],
    ['DL', 'NN']
]

# 使用TfidfVectorizer進行特徵提取（n-grams）
vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(texts).toarray()

# 標簽二值化
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 轉換為 PyTorch 張量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 模型定義
class MultiLabelClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 隱藏層1
        self.fc2 = nn.Linear(128, 64)        # 隱藏層2
        self.fc3 = nn.Linear(64, output_dim)  # 輸出層
        self.sigmoid = nn.Sigmoid()          # 用於多標簽分類

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 激活函數為ReLU
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # 輸出為0~1之間的概率值
        return x

# 模型參數
input_dim = X_train.shape[1]  # 輸入維度是TF-IDF特徵的數量
output_dim = y_train.shape[1]  # 輸出維度是標簽的數量

# 創建模型實例
model = MultiLabelClassificationModel(input_dim, output_dim)

# 定義損失函數和優化器
criterion = nn.BCELoss()  # 二元交叉熵損失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 10
batch_size = 2

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    # Mini-batch 梯度下降
    for i in range(0, X_train.size(0), batch_size):
        inputs = X_train[i:i + batch_size]
        labels = y_train[i:i + batch_size]
        
        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(X_train):.4f}")

# 模型評估
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted = (outputs > 0.5).float()
    print("Predicted Labels:", predicted)
    print("True Labels:", y_test)
7. 總結
通過 PyTorch 實現多標簽分類，可以使用 sigmoid 函數獨立預測每個類別的概率。
使用二元交叉熵損失函數適合多標簽分類任務。
該示例展示了如何進行文本的特徵提取、標簽二值化以及多標簽分類模型的訓練與評估。

上述例子是三個labels 若是在不確定下  找出文本中的n-labels 
如果在多標簽分類任務中，不確定有多少個標簽（n-labels），我們可以通過模型來預測每個文本的標簽數量，或者使用一種基於概率的動態方法來確定文本中的標簽數量。具體流程可以通過以下方式來實現：

模型架搆保持不變：輸出層繼續使用 sigmoid 激活函數，并預測每個可能的標簽概率。
動態決定標簽數量：通過預測的概率值，我們可以設定一個動態的閾值（比如 0.5），來決定哪些標簽被選中。這種方式不需要預定義固定的標簽數量，而是讓模型根據每個文本的具體情況來確定。
步驟
1. 數據准備
在數據准備階段，依然需要多標簽的二值化編碼，但標簽的數量是由訓練數據中出現的所有標簽來決定。最終模型預測時，將會為每個標簽輸出一個概率，表示該標簽是否應該被選擇。

2. 動態確定 n-labels
對於每個樣本的輸出概率，我們可以設置不同的閾值來決定標簽。例如：

固定閾值：可以簡單地設置一個固定閾值（如 0.5），如果某個類別的預測概率超過該值，則認為該類別屬於當前樣本。
動態閾值：如果想動態決定標簽數量，可以基於每個樣本的輸出，選擇前 n 個最大的概率值，或者根據整個輸出的概率分布來設定閾值。
修改代碼以支持動態標簽選擇
下面將展示如何在 PyTorch 代碼中處理這種情況。

1. 修改模型部分保持不變
我們依然使用原先定義的 MultiLabelClassificationModel 模型。

2. 動態確定 n-labels 的推斷方法
我們可以在模型預測時，動態決定每個文本的 n-labels。可以通過如下策略：

根據輸出的概率來選擇所有大於 0.5 的標簽（或調整這個閾值）。
選擇概率最大的前 n 個標簽作為預測結果。
示例代碼（使用閾值和動態選擇）
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# 分詞工具
nltk.download('punkt')
def tokenize(text):
    return nltk.word_tokenize(text)

# 示例文本
texts = [
    "機器學習和深度學習是人工智能的一部分",
    "自然語言處理涉及文本分析和生成",
    "深度學習使用神經網絡來進行學習"
]

# 標簽，文本對應的多個標簽
labels = [
    ['AI', 'ML', 'DL'],
    ['NLP', 'Text Analysis'],
    ['DL', 'NN']
]

# 使用TfidfVectorizer進行特徵提取（n-grams）
vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(texts).toarray()

# 標簽二值化
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 轉換為 PyTorch 張量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 模型定義
class MultiLabelClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 隱藏層1
        self.fc2 = nn.Linear(128, 64)        # 隱藏層2
        self.fc3 = nn.Linear(64, output_dim)  # 輸出層
        self.sigmoid = nn.Sigmoid()          # 用於多標簽分類

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 激活函數為ReLU
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # 輸出為0~1之間的概率值
        return x

# 模型參數
input_dim = X_train.shape[1]  # 輸入維度是TF-IDF特徵的數量
output_dim = y_train.shape[1]  # 輸出維度是標簽的數量

# 創建模型實例
model = MultiLabelClassificationModel(input_dim, output_dim)

# 定義損失函數和優化器
criterion = nn.BCELoss()  # 二元交叉熵損失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 10
batch_size = 2

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    # Mini-batch 梯度下降
    for i in range(0, X_train.size(0), batch_size):
        inputs = X_train[i:i + batch_size]
        labels = y_train[i:i + batch_size]
        
        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(X_train):.4f}")

# 模型評估
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    
    # 方法1: 固定閾值（如0.5）來確定標簽
    fixed_threshold_pred = (outputs > 0.5).float()
    
    # 方法2: 動態確定n-labels（選擇最大概率的n個標簽）
    n_labels = 2  # 假設我們想要找到n個最可能的標簽
    top_n_pred = torch.topk(outputs, n_labels, dim=1).indices
    top_n_pred_labels = torch.zeros_like(outputs)
    for i, idx in enumerate(top_n_pred):
        top_n_pred_labels[i, idx] = 1

    print("Predicted Labels with Fixed Threshold (0.5):")
    print(fixed_threshold_pred)

    print("Predicted Labels with Dynamic Top-n (2 labels):")
    print(top_n_pred_labels)

    print("True Labels:")
    print(y_test)
解釋
方法 1: 固定閾值
使用固定的閾值（如 0.5）來決定哪些標簽屬於當前樣本。如果某個類別的預測概率超過閾值，則認為該類別屬於當前樣本。這種方法適合當你認為標簽數量較為固定時。

fixed_threshold_pred = (outputs > 0.5).float()
方法 2: 動態確定 n-labels
這是一種動態的標簽選擇方式。假設你不知道具體的標簽數量，可以根據每個樣本的預測概率，選取前 n 個最高的概率值對應的標簽。例如，假設我們希望選擇前 2 個最有可能的標簽：

n_labels = 2  # 假設我們想要找到n個最可能的標簽
top_n_pred = torch.topk(outputs, n_labels, dim=1).indices
top_n_pred_labels = torch.zeros_like(outputs)
for i, idx in enumerate(top_n_pred):
    top_n_pred_labels[i, idx] = 1
這里，我們選取每個樣本中概率最大的 n 個標簽，并將其作為預測結果。

結論
固定閾值法 適合預定義好閾值的情況下使用，通常選擇 0.5 作為默認值，但你也可以調整該值以獲得更好的分類結果。
動態 n-labels 選擇法 適合標簽數量不確定的場景。你可以根據每個樣本的預測情況，選擇概率最大的 n 個標簽。

訓練分類 給出以CrossEntropy的torch範例
在 PyTorch 中，交叉熵損失（CrossEntropy Loss）通常用於單標簽分類問題。以下是一個使用交叉熵損失進行訓練分類模型的完整示例，重點是單標簽分類任務。

1. 數據准備
我們首先需要准備數據，假設我們有一個文本分類任務，其中每個文本都有一個單一的標簽。

2. 實現步驟
導入所需庫
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
示例數據
假設我們有以下文本數據和對應的單標簽：

python
複製程式碼
# 示例文本
texts = [
    "機器學習是人工智能的一部分",
    "深度學習是機器學習的一種",
    "自然語言處理是人工智能的重要應用",
    "計算機視覺是機器學習的一個重要領域",
    "強化學習涉及學習最優策略"
]

# 標簽（每個文本對應的單一標簽）
labels = [0, 0, 1, 1, 2]  # 0: AI, 1: ML, 2: RL
數據預處理
我們需要對文本進行特徵提取，并將標簽轉換為 PyTorch 張量。

python
複製程式碼
# 使用TfidfVectorizer進行特徵提取
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts).toarray()

# 轉換標簽為numpy數組
Y = np.array(labels)

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 轉換為 PyTorch 張量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # 需要使用long類型
y_test = torch.tensor(y_test, dtype=torch.long)
3. 定義深度學習模型
我們使用簡單的全連接神經網絡進行文本分類。

python
複製程式碼
class TextClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 隱藏層1
        self.fc2 = nn.Linear(128, output_dim)  # 輸出層

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU激活
        x = self.fc2(x)               # 輸出層
        return x
4. 訓練模型
我們使用交叉熵損失函數（CrossEntropyLoss），并選擇 Adam 作為優化器。

python
複製程式碼
# 模型參數
input_dim = X_train.shape[1]  # 輸入維度是TF-IDF特徵的數量
output_dim = len(set(labels))  # 輸出維度是標簽的數量

# 創建模型實例
model = TextClassificationModel(input_dim, output_dim)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()  # 交叉熵損失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 10
batch_size = 2

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    # Mini-batch 梯度下降
    for i in range(0, X_train.size(0), batch_size):
        inputs = X_train[i:i + batch_size]
        labels = y_train[i:i + batch_size]
        
        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(X_train):.4f}")
5. 評估模型
訓練結束后，我們可以在測試集上進行評估，查看模型的准確率。

model.eval()  # 進入評估模式
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)  # 選擇概率最大的類
    accuracy = (predicted == y_test).float().mean()  # 計算准確率
    print(f"Accuracy: {accuracy:.4f}")
6. 完整代碼總結
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 示例文本
texts = [
    "機器學習是人工智能的一部分",
    "深度學習是機器學習的一種",
    "自然語言處理是人工智能的重要應用",
    "計算機視覺是機器學習的一個重要領域",
    "強化學習涉及學習最優策略"
]

# 標簽（每個文本對應的單一標簽）
labels = [0, 0, 1, 1, 2]  # 0: AI, 1: ML, 2: RL

# 使用TfidfVectorizer進行特徵提取
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts).toarray()

# 轉換標簽為numpy數組
Y = np.array(labels)

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 轉換為 PyTorch 張量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # 需要使用long類型
y_test = torch.tensor(y_test, dtype=torch.long)

# 模型定義
class TextClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 隱藏層1
        self.fc2 = nn.Linear(128, output_dim)  # 輸出層

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU激活
        x = self.fc2(x)               # 輸出層
        return x

# 模型參數
input_dim = X_train.shape[1]  # 輸入維度是TF-IDF特徵的數量
output_dim = len(set(labels))  # 輸出維度是標簽的數量

# 創建模型實例
model = TextClassificationModel(input_dim, output_dim)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()  # 交叉熵損失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 10
batch_size = 2

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    # Mini-batch 梯度下降
    for i in range(0, X_train.size(0), batch_size):
        inputs = X_train[i:i + batch_size]
        labels = y_train[i:i + batch_size]
        
        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(X_train):.4f}")

# 模型評估
model.eval()  # 進入評估模式
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)  # 選擇概率最大的類
    accuracy = (predicted == y_test).float().mean()  # 計算准確率
    print(f"Accuracy: {accuracy:.4f}")
7. 總結
該示例展示了如何使用 PyTorch 和交叉熵損失進行文本分類。
通過 TF-IDF 提取特徵，搆建簡單的全連接神經網絡模型進行訓練與評估。
可以通過調整模型架搆、學習率和批量大小等參數來進一步優化模型表現。

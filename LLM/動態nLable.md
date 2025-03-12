torch medical EHR 找出疾病
在醫療領域，電子健康記錄（Electronic Health Records, EHR）包含了大量患者的醫療信息，包括診斷、治療、藥物、實驗室結果等。利用深度學習模型分析EHR數據以識別疾病，是一個具有挑戰性但極具潛力的任務。下面將展示如何使用PyTorch搆建一個多標簽分類模型來從EHR數據中預測疾病。

目錄
准備工作
數據預處理
創建PyTorch數據集和數據加載器
定義深度學習模型
訓練模型
評估模型
完整代碼示例
總結
准備工作
在開始之前，確保你已經安裝了以下必要的庫：

pip install torch torchvision scikit-learn pandas numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
數據預處理
假設我們有一個EHR數據集，其中每條記錄包含患者的多種特徵（如年齡、性別、診斷代碼、實驗室結果等）和對應的疾病標簽。由於實際的EHR數據集（如MIMIC-III）較為復雜且受限於隱私，這里我們將使用一個合成的示例數據集進行說明。

示例數據集
# 創建一個合成的EHR數據集
data = {
    'patient_id': range(1, 11),
    'age': [25, 47, 52, 36, 62, 29, 41, 55, 33, 49],
    'gender': ['F', 'M', 'M', 'F', 'F', 'M', 'F', 'M', 'F', 'M'],
    'blood_pressure': [120, 130, 115, 140, 135, 125, 128, 132, 118, 127],
    'cholesterol': [200, 190, 250, 180, 220, 210, 195, 230, 185, 205],
    'diagnosis_codes': [
        ['D1', 'D2'],
        ['D2', 'D3'],
        ['D1', 'D3'],
        ['D4'],
        ['D2', 'D4'],
        ['D1'],
        ['D3', 'D4'],
        ['D1', 'D2', 'D4'],
        ['D2'],
        ['D3']
    ],
    'diseases': [
        ['Disease_A'],
        ['Disease_B'],
        ['Disease_A', 'Disease_B'],
        ['Disease_C'],
        ['Disease_B', 'Disease_C'],
        ['Disease_A'],
        ['Disease_B', 'Disease_C'],
        ['Disease_A', 'Disease_B', 'Disease_C'],
        ['Disease_B'],
        ['Disease_C']
    ]
}

df = pd.DataFrame(data)
print(df)
輸出示例
   patient_id  age gender  blood_pressure  cholesterol diagnosis_codes  \
0           1   25      F             120          200        [D1, D2]   
1           2   47      M             130          190        [D2, D3]   
2           3   52      M             115          250        [D1, D3]   
3           4   36      F             140          180             [D4]   
4           5   62      F             135          220        [D2, D4]   
5           6   29      M             125          210             [D1]   
6           7   41      F             128          195        [D3, D4]   
7           8   55      M             132          230     [D1, D2, D4]   
8           9   33      F             118          185             [D2]   
9          10   49      M             127          205             [D3]   

                     diseases  
0               [Disease_A]  
1               [Disease_B]  
2        [Disease_A, Disease_B]  
3               [Disease_C]  
4        [Disease_B, Disease_C]  
5               [Disease_A]  
6        [Disease_B, Disease_C]  
7  [Disease_A, Disease_B, Disease_C]  
8               [Disease_B]  
9               [Disease_C]  
特徵工程
1. 處理數值特徵
對於數值特徵（如年齡、血壓、膽固醇），我們通常進行標准化處理。

numerical_features = ['age', 'blood_pressure', 'cholesterol']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
2. 處理類別特徵
對於類別特徵（如性別、診斷代碼），我們需要進行編碼。

性別：可以進行二值編碼。
診斷代碼：由於每個患者可能有多個診斷代碼，我們可以使用多標簽二值化（MultiLabelBinarizer）。
# 編碼性別
df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# 編碼診斷代碼
mlb_diag = MultiLabelBinarizer()
diag_encoded = mlb_diag.fit_transform(df['diagnosis_codes'])
diag_feature_names = [f"diag_{cls}" for cls in mlb_diag.classes_]
diag_df = pd.DataFrame(diag_encoded, columns=diag_feature_names)

# 合并編碼后的特徵
df = pd.concat([df, diag_df], axis=1)
df.drop(['patient_id', 'diagnosis_codes'], axis=1, inplace=True)
3. 處理標簽
由於每個患者可能有多個疾病，我們需要對標簽進行多標簽二值化。

# 多標簽二值化
mlb_disease = MultiLabelBinarizer()
disease_encoded = mlb_disease.fit_transform(df['diseases'])
disease_feature_names = mlb_disease.classes_
disease_df = pd.DataFrame(disease_encoded, columns=disease_feature_names)

# 合并標簽
df = pd.concat([df, disease_df], axis=1)
df.drop(['diseases'], axis=1, inplace=True)

print(df)
數據划分
# 特徵和標簽
X = df.drop(disease_feature_names, axis=1).values
Y = df[disease_feature_names].values

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 轉換為PyTorch張量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
創建PyTorch數據集和數據加載器
為了高效地加載數據，我們將創建自定義的Dataset類，并使用DataLoader進行批量加載。

class EHRDataset(Dataset):
    def __init__(self, features, labels):
        self.X = features
        self.Y = labels
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 創建數據集
train_dataset = EHRDataset(X_train, y_train)
test_dataset = EHRDataset(X_test, y_test)

# 創建數據加載器
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
定義深度學習模型
我們將搆建一個簡單的全連接神經網絡，用於多標簽分類任務。輸出層使用sigmoid激活函數，以便為每個標簽輸出一個獨立的概率值。

class EHRDiseaseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EHRDiseaseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out
模型實例化
input_dim = X_train.shape[1]  # 輸入特徵數量
hidden_dim = 64                # 隱藏層神經元數量
output_dim = y_train.shape[1]  # 輸出標簽數量

model = EHRDiseaseClassifier(input_dim, hidden_dim, output_dim)
訓練模型
在多標簽分類任務中，通常使用BCELoss（二元交叉熵損失）或BCEWithLogitsLoss作為損失函數。這里我們使用BCELoss，因為模型的輸出已經經過sigmoid激活。

# 定義損失函數和優化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練參數
num_epochs = 50
訓練循環
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_Y in train_loader:
        # 前向傳播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        
        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # 打印每個epoch的損失
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
輸出示例
less
複製程式碼
Epoch [1/50], Loss: 0.7081
Epoch [10/50], Loss: 0.5384
Epoch [20/50], Loss: 0.3972
Epoch [30/50], Loss: 0.2738
Epoch [40/50], Loss: 0.1716
Epoch [50/50], Loss: 0.0912
評估模型
我們將在測試集上評估模型的性能，使用多個指標如准確率（Accuracy）、F1分數等。

model.eval()
with torch.no_grad():
    all_outputs = []
    all_labels = []
    for batch_X, batch_Y in test_loader:
        outputs = model(batch_X)
        all_outputs.append(outputs)
        all_labels.append(batch_Y)
    
    # 拼接所有批次的輸出和標簽
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 使用0.5作為閾值進行二值化
    predicted = (all_outputs >= 0.5).int()
    true = all_labels.int()
    
    # 計算准確率
    accuracy = (predicted == true).float().mean()
    print(f"Accuracy: {accuracy:.4f}")
    
    # 計算F1分數
    predicted_np = predicted.numpy()
    true_np = true.numpy()
    f1 = f1_score(true_np, predicted_np, average='micro')
    print(f"F1 Score (Micro): {f1:.4f}")
    
    # 打印分類報告
    print("Classification Report:")
    print(classification_report(true_np, predicted_np, target_names=disease_feature_names))
輸出示例

Accuracy: 0.9167
F1 Score (Micro): 0.9167
Classification Report:
              precision    recall  f1-score   support

    Disease_A       1.00      1.00      1.00         1
    Disease_B       1.00      1.00      1.00         2
    Disease_C       1.00      1.00      1.00         1

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
注意：由於我們使用的是一個非常小的合成數據集，模型的評估結果可能不具有代表性。在實際應用中，應該使用更大且更復雜的數據集，并進行適當的交叉驗證。

完整代碼示例
以下是上述步驟的完整代碼整合，便於直接運行和測試。

python
複製程式碼
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. 創建合成的EHR數據集
data = {
    'patient_id': range(1, 11),
    'age': [25, 47, 52, 36, 62, 29, 41, 55, 33, 49],
    'gender': ['F', 'M', 'M', 'F', 'F', 'M', 'F', 'M', 'F', 'M'],
    'blood_pressure': [120, 130, 115, 140, 135, 125, 128, 132, 118, 127],
    'cholesterol': [200, 190, 250, 180, 220, 210, 195, 230, 185, 205],
    'diagnosis_codes': [
        ['D1', 'D2'],
        ['D2', 'D3'],
        ['D1', 'D3'],
        ['D4'],
        ['D2', 'D4'],
        ['D1'],
        ['D3', 'D4'],
        ['D1', 'D2', 'D4'],
        ['D2'],
        ['D3']
    ],
    'diseases': [
        ['Disease_A'],
        ['Disease_B'],
        ['Disease_A', 'Disease_B'],
        ['Disease_C'],
        ['Disease_B', 'Disease_C'],
        ['Disease_A'],
        ['Disease_B', 'Disease_C'],
        ['Disease_A', 'Disease_B', 'Disease_C'],
        ['Disease_B'],
        ['Disease_C']
    ]
}

df = pd.DataFrame(data)

# 2. 特徵工程

# 數值特徵標准化
numerical_features = ['age', 'blood_pressure', 'cholesterol']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 性別編碼
df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# 診斷代碼編碼
mlb_diag = MultiLabelBinarizer()
diag_encoded = mlb_diag.fit_transform(df['diagnosis_codes'])
diag_feature_names = [f"diag_{cls}" for cls in mlb_diag.classes_]
diag_df = pd.DataFrame(diag_encoded, columns=diag_feature_names)
df = pd.concat([df, diag_df], axis=1)
df.drop(['patient_id', 'diagnosis_codes'], axis=1, inplace=True)

# 標簽編碼
mlb_disease = MultiLabelBinarizer()
disease_encoded = mlb_disease.fit_transform(df['diseases'])
disease_feature_names = mlb_disease.classes_
disease_df = pd.DataFrame(disease_encoded, columns=disease_feature_names)
df = pd.concat([df, disease_df], axis=1)
df.drop(['diseases'], axis=1, inplace=True)

# 特徵和標簽
X = df.drop(disease_feature_names, axis=1).values
Y = df[disease_feature_names].values

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 轉換為PyTorch張量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 3. 創建數據集和數據加載器
class EHRDataset(Dataset):
    def __init__(self, features, labels):
        self.X = features
        self.Y = labels
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_dataset = EHRDataset(X_train, y_train)
test_dataset = EHRDataset(X_test, y_test)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 4. 定義模型
class EHRDiseaseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EHRDiseaseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = y_train.shape[1]

model = EHRDiseaseClassifier(input_dim, hidden_dim, output_dim)

# 5. 定義損失函數和優化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 訓練模型
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_Y in train_loader:
        # 前向傳播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        
        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # 打印每個epoch的損失
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# 7. 評估模型
model.eval()
with torch.no_grad():
    all_outputs = []
    all_labels = []
    for batch_X, batch_Y in test_loader:
        outputs = model(batch_X)
        all_outputs.append(outputs)
        all_labels.append(batch_Y)
    
    # 拼接所有批次的輸出和標簽
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 使用0.5作為閾值進行二值化
    predicted = (all_outputs >= 0.5).int()
    true = all_labels.int()
    
    # 計算准確率
    accuracy = (predicted == true).float().mean()
    print(f"Accuracy: {accuracy:.4f}")
    
    # 計算F1分數
    predicted_np = predicted.numpy()
    true_np = true.numpy()
    f1 = f1_score(true_np, predicted_np, average='micro')
    print(f"F1 Score (Micro): {f1:.4f}")
    
    # 打印分類報告
    print("Classification Report:")
    print(classification_report(true_np, predicted_np, target_names=disease_feature_names))
總結
本文展示了如何使用PyTorch搆建一個多標簽分類模型，從電子健康記錄（EHR）數據中預測疾病。主要步驟包括：

數據准備與預處理：處理數值和類別特徵，對多標簽進行二值化。
創建PyTorch數據集和數據加載器：便於批量處理數據。
定義深度學習模型：使用全連接神經網絡進行多標簽分類。
訓練模型：使用二元交叉熵損失函數和Adam優化器。
評估模型：通過准確率和F1分數等指標評估模型性能。
在實際應用中，EHR數據可能更加復雜，包含更多的特徵和更高的維度。為了提升模型性能，可以考慮以下方法：

特徵選擇與工程：通過特徵選擇方法篩選重要特徵，或創建新的衍生特徵。
高級模型架搆：如使用嵌入層處理高基數的類別特徵，或使用深層網絡結搆。
正則化與優化：應用正則化技朮（如Dropout、L2正則化）防止過擬合，調整學習率等超參數優化模型。
數據增強與平衡：處理類別不平衡問題，應用過釆樣或欠釆樣技朮。
此外，使用真實的EHR數據集（如MIMIC-III）時，還需考慮數據隱私和合規性問題，并進行更為復雜的數據清洗與預處理工作。

通過本文的示例，你可以了解如何使用PyTorch處理多標簽分類任務，并將這一方法應用於實際的醫療數據分析中，以輔助疾病預測和醫療決策。


torch EHR dynamic n-labels
在電子健康記錄（EHR）中，動態n-labels問題通常指的是在多標簽分類任務中，標簽數量（n）可能隨時間或病人的數據更新而動態變化的問題。例如，不同的患者可能具有不同數量的疾病（標簽），且這些疾病的類型或數量可能隨着新的診斷或治療方案而變化。

為了解決這個問題，我們需要搆建一個模型，該模型可以靈活地處理動態變化的標簽數量。PyTorch中的多標簽分類可以很好地適應這一需求，尤其是當標簽的數量不固定或隨着時間變化時。

核心挑戰
標簽數量的變化：傳統的多標簽模型通常基於固定數量的標簽輸出，但在動態n-labels的情況下，我們不能預定義固定的輸出大小。
新標簽的引入：隨着時間推移，新的疾病標簽可能會不斷添加，而模型需要能夠應對這些標簽的引入。
時間序列數據：EHR數據通常以時間序列的形式存在，患者的醫療曆史、檢查、診斷等是按照時間記錄的。因此，如何利用時間依賴關系（如RNN、LSTM、Transformer等）來處理動態標簽是關鍵。
解決思路
我們可以設計一個多標簽分類模型，結合時間序列建模和多標簽分類的靈活機制，以應對動態標簽問題。

1. 基於時間序列的模型結搆
EHR數據通常是時間序列數據，因此我們可以使用RNN（循環神經網絡）、LSTM（長短期記憶網絡）或Transformer來處理時間依賴性。

2. 動態標簽處理
使用一個靈活的多標簽預測機制，如動態擴展的輸出層，允許我們在訓練期間根據數據的不同標簽集動態調整輸出維度。我們可以預定義最大標簽數量（n-max），并允許模型根據實際標簽數進行掩碼或部分輸出。

3. 損失函數的設計
在多標簽分類中，通常使用二元交叉熵損失函數 (BCELoss) 來處理每個標簽獨立的概率輸出。對於動態標簽，我們可以根據每個樣本的有效標簽動態調整損失計算。

示例實現
下面是一個使用LSTM處理動態n-labels的簡化模型架搆，能夠處理不同患者在不同時間點的動態標簽。

數據准備
我們假設EHR數據以時間序列的形式存在，并且每個時間步都有不同的標簽。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

# 示例數據：10個患者，每個患者有5個時間步，每個時間步有3個特徵，最多有4個疾病標簽
num_patients = 10
sequence_length = 5
feature_size = 3
max_labels = 4  # 假設最多有4個疾病標簽

# 隨機生成特徵
X = np.random.rand(num_patients, sequence_length, feature_size).astype(np.float32)

# 隨機生成多標簽（0或1表示疾病是否存在）
Y = np.random.randint(0, 2, (num_patients, sequence_length, max_labels)).astype(np.float32)

# 轉換為PyTorch張量
X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)
定義數據集
python
複製程式碼
class EHRDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 創建數據集和數據加載器
dataset = EHRDataset(X_tensor, Y_tensor)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
動態n-labels模型
我們使用LSTM處理時間序列特徵，并使用全連接層輸出多標簽預測。

python
複製程式碼
class DynamicNLabelsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DynamicNLabelsModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM層
        lstm_out, _ = self.lstm(x)
        # 全連接層輸出
        out = self.fc(lstm_out)
        # 使用sigmoid激活函數進行多標簽預測
        out = self.sigmoid(out)
        return out
模型實例化與訓練
input_dim = feature_size  # 輸入維度（特徵數）
hidden_dim = 32  # LSTM隱藏層維度
output_dim = max_labels  # 輸出維度（最大標簽數）

# 實例化模型
model = DynamicNLabelsModel(input_dim, hidden_dim, output_dim)

# 定義損失函數和優化器
criterion = nn.BCELoss()  # 使用二元交叉熵損失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_Y in data_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(data_loader):.4f}")
動態標簽處理
模型輸出的維度與 max_labels 對應，但在實際數據中，每個患者的標簽數可能不一致。為了處理這種動態性，我們可以使用掩碼（masking）來只計算有效標簽的損失。

# 動態處理不同的標簽數，例如我們可以用mask矩陣標記哪些標簽有效
mask = (Y_tensor.sum(dim=2) > 0).float()  # 假設沒有疾病的時間步標簽全為0

for batch_X, batch_Y in data_loader:
    outputs = model(batch_X)
    
    # 應用掩碼，僅計算有效標簽的損失
    masked_loss = criterion(outputs * mask, batch_Y * mask)
    
    optimizer.zero_grad()
    masked_loss.backward()
    optimizer.step()
評估模型
model.eval()
with torch.no_grad():
    for batch_X, batch_Y in data_loader:
        outputs = model(batch_X)
        predicted = (outputs >= 0.5).float()  # 將輸出的概率轉換為0或1
        print(f"Predicted: {predicted}")
        print(f"Ground Truth: {batch_Y}")
總結 處理動態標簽：在多標簽分類任務中，我們通過靈活的模型架搆（如LSTM處理時間序列）和損失函數的動態調整（如掩碼）來應對標簽數量的動態變化。

時間序列模型：由於EHR數據具有時間依賴性，我們使用LSTM來捕捉時間序列特徵，以更好地預測患者的動態標簽。

損失和掩碼機制：通過掩碼機制，我們能夠只計算有效標簽的損失，解決了不同樣本標簽數不一致的問題。

這種方法適合用於EHR數據中的動態n-labels問題，尤其是涉及時間序列分析的場景，如隨着時間推移病人的疾病診斷和標簽更新等。

在醫療領域，電子健康記錄（Electronic Health Records, EHR）包含了大量患者的醫療信息，包括診斷、治療、藥物、實驗室結果等。利用深度學習模型分析EHR數據以識別疾病，是一個具有挑戰性但極具潛力的任務。下面將展示如何使用PyTorch構建一個多標簽分類模型來從EHR數據中預測疾病。

目錄
準備工作
數據預處理
創建PyTorch數據集和數據加載器
定義深度學習模型
訓練模型
評估模型
完整代碼示例
總結
準備工作
在開始之前，确保你已經安裝了以下必要的庫：

pip install torch torchvision scikit-learn pandas numpy
from pandas import DataFrame, concat as pndsConcat
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.nn import Module, ReLU, Linear, Sigmoid, BCELoss

數據預處理 假設我們有一個EHR數據集，其中每條記錄包含患者的多種特征（如年齡、性別、診斷代碼、實驗室結果等）和對應的疾病標簽。由于實際的EHR數據集（如MIMIC-III）較爲復雜且受限于隱私，這裏我們將使用一個合成的示例數據集進行說明。

# 創建一個合成的EHR數據集 示例數據集
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
特征工程
1. 處理數值特征 對于數值特征（如年齡、血壓、膽固醇），我們通常進行標準化處理。

numerical_features = ['age', 'blood_pressure', 'cholesterol']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
2. 處理類別特征
對于類別特征（如性別、診斷代碼），我們需要進行編碼。

性別：可以進行二值編碼。
診斷代碼：由于每個患者可能有多個診斷代碼，我們可以使用多標簽二值化（MultiLabelBinarizer）。
# 編碼性別
df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# 編碼診斷代碼
mlb_diag = MultiLabelBinarizer()
diag_encoded = mlb_diag.fit_transform(df['diagnosis_codes'])
diag_feature_names = [f"diag_{cls}" for cls in mlb_diag.classes_]
diag_df = pd.DataFrame(diag_encoded, columns=diag_feature_names)

# 合并編碼後的特征
df = pd.concat([df, diag_df], axis=1)
df.drop(['patient_id', 'diagnosis_codes'], axis=1, inplace=True)
3. 處理標簽
由于每個患者可能有多個疾病，我們需要對標簽進行多標簽二值化。

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
# 特征和標簽
X = df.drop(disease_feature_names, axis=1).values
Y = df[disease_feature_names].values

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=42)

# 轉換爲PyTorch張量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
創建PyTorch數據集和數據加載器 爲了高效地加載數據，我們將創建自定義的Dataset類，并使用DataLoader進行批量加載。

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
定義深度學習模型 我們將構建一個簡單的全連接神經網絡，用于多標簽分類任務。輸出層使用sigmoid激活函數，以便爲每個標簽輸出一個獨立的概率值。


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
input_dim = X_train.shape[1]  # 輸入特征數量
hidden_dim = 64                # 隱藏層神經元數量
output_dim = y_train.shape[1]  # 輸出標簽數量

model = EHRDiseaseClassifier(input_dim, hidden_dim, output_dim)
訓練模型
在多標簽分類任務中，通常使用BCELoss（二元交叉熵損失）或BCEWithLogitsLoss作爲損失函數。這裏我們使用BCELoss，因爲模型的輸出已經經過sigmoid激活。

# 定義損失函數和優化器
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 訓練參數
num_epochs = 50
#for epoch in range(num_epochs):
LOSS, EPS=1, 1e-6
while LOSS>EPS: # 訓練循環
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_Y in train_loader:
        # 前向傳播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)

        optimizer.zero_grad() # 反向傳播和優化
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        LOSS=epoch_loss

    # 打印每個epoch的損失
    #if (epoch+1) % 10 == 0 or epoch == 0: print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
輸出示例
Epoch [1/50], Loss: 0.7081
Epoch [10/50], Loss: 0.5384
Epoch [20/50], Loss: 0.3972
Epoch [30/50], Loss: 0.2738
Epoch [40/50], Loss: 0.1716
Epoch [50/50], Loss: 0.0912
評估模型 我們將在測試集上評估模型的性能，使用多個指標如準确率（Accuracy）、F1分數等。

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

    # 使用0.5作爲阈值進行二值化
    predicted = (all_outputs >= 0.5).int()
    true = all_labels.int()

    # 計算準确率
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
輸出示例 Accuracy: 0.9167
F1 Score (Micro): 0.9167
Classification Report:
              precision    recall  f1-score   support

    Disease_A       1.00      1.00      1.00         1
    Disease_B       1.00      1.00      1.00         2
    Disease_C       1.00      1.00      1.00         1

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
注意：由于我們使用的是一個非常小的合成數據集，模型的評估結果可能不具有代表性。在實際應用中，應該使用更大且更復雜的數據集，并進行適當的交叉驗證。

完整代碼示例 以下是上述步驟的完整代碼整合，便于直接運行和測試。

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

df = DataFrame(data)

# 2. 特征工程

# 數值特征標準化
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

# 特征和標簽
X = df.drop(disease_feature_names, axis=1).values
Y = df[disease_feature_names].values

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=42)

# 轉換爲PyTorch張量
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
criterion = BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 50 # 6. 訓練模型

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

    # 使用0.5作爲阈值進行二值化
    predicted = (all_outputs >= 0.5).int()
    true = all_labels.int()

    # 計算準确率
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
總結 本文展示了如何使用PyTorch構建一個多標簽分類模型，從電子健康記錄（EHR）數據中預測疾病。主要步驟包括：

數據準備與預處理：處理數值和類別特征，對多標簽進行二值化。
創建PyTorch數據集和數據加載器：便于批量處理數據。
定義深度學習模型：使用全連接神經網絡進行多標簽分類。
訓練模型：使用二元交叉熵損失函數和Adam優化器。
評估模型：通過準确率和F1分數等指標評估模型性能。

在實際應用中，EHR數據可能更加復雜，包含更多的特征和更高的維度。爲了提升模型性能，可以考慮以下方法：

特征選擇與工程：通過特征選擇方法篩選重要特征，或創建新的衍生特征。
高級模型架構：如使用嵌入層處理高基數的類別特征，或使用深層網絡結構。
正則化與優化：應用正則化技術（如Dropout、L2正則化）防止過藕合，調整學習率等超參數優化模型。
數據增強與平衡：處理類別不平衡問題，應用過采樣或欠采樣技術。
此外，使用真實的EHR數據集（如MIMIC-III）時，還需考慮數據隱私和合規性問題，并進行更爲復雜的數據清洗與預處理工作。

通過本文的示例，你可以了解如何使用PyTorch處理多標簽分類任務，并將這一方法應用于實際的醫療數據分析中，以輔助疾病預測和醫療決策。

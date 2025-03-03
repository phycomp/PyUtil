from pandas import read_csv
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from torch.nn import Module, Linear, ReLU, CrossEntropyLoss

# 1. 加載 ICD-11 數據表和 EHR 數據
icd_data = read_csv('icd11_codes.csv')  # ICD-11 代碼和描述
ehr_data = read_csv('ehr_records.csv')  # EHR 記錄文本

# 2. 使用 TfidfVectorizer 對 EHR 文本進行特徵提取
vectorizer = TfidfVectorizer(max_features=5000)  # 將文本轉換為特徵向量
X = vectorizer.fit_transform(ehr_data['diagnosis_text']).toarray()

# 3. 標簽編碼，將 ICD-11 代碼轉換為數字標簽
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(ehr_data['icd_code'])

# 4. 拆分訓練和測試數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 定義一個簡單的神經網絡模型來預測 ICD-11 代碼
class ICD分類(Module):
    def __init__(self, input_size, num_classes):
        super(ICD分類, self).__init__()
        self.fc1 = Linear(input_size, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 定義超參數
input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)
learning_rate, batch_size, num_epochs=.001, 64, 10

from torch import tensor as trchTnsr
# 6. 創建 DataLoader
class EHRDataset(Dataset):
    def __init__(self, X, y):
        self.X = trchTnsr(X, dtype=torch.float32)
        self.y = trchTnsr(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EHRDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 7. 定義模型、損失函數和優化器

from torch.optim import Adam
model = ICD分類(input_size, num_classes)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# 8. 訓練模型
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 9. 評估模型
from torch import no_grad as trchNograd, max as trchMax
with trchNograd():
    test_inputs = trchTnsr(X_test, dtype=torch.float32)
    test_labels = trchTnsr(y_test, dtype=torch.long)

    outputs = model(test_inputs)
    _, predicted = trchMax(outputs, 1)

    accuracy = (predicted == test_labels).sum().item() / len(test_labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')

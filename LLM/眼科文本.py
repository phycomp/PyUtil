"""text
在醫療領域，SOAP 是一種標準的病歷記錄格式，代表著四個部分：Subjective（主訴）、Objective（客觀檢查）、Assessment（評估） 和 Plan（計畫）。對于眼科檢查，OD（右眼）、OS（左眼）、IOP（眼內壓）等是常見的術語。我們可以設計一個 PyTorch 的 Prompt，用于從 SOAP 格式的文本中提取出與 OD、OS、IOP 相關的信息。

使用 PyTorch 結合自然語言處理（NLP）技術，我們可以訓練模型自動從 SOAP 文檔中提取這些關鍵信息。以下是一個基于 BERT 的命名實體識別（NER）任務示例，用于提取 OD、OS 和 IOP 信息。

1. 數據準備

我們首先需要有帶標注的 SOAP 文本數據集，其中 OD、OS 和 IOP 被標注爲不同的實體類別。例如，假設以下文本是標注數據的一部分：

SOAP: 
Subjective: The patient reports blurry vision.
Objective: OD 24 mmHg, OS 22 mmHg, IOP within normal limits.
Assessment: Glaucoma suspect.
Plan: Monitor IOP levels.

標注爲：

OD: B-OD
24 mmHg: I-OD
OS: B-OS
22 mmHg: I-OS
IOP: B-IOP
within normal limits: I-IOP
"""

"""
2. 基于 BERT 的命名實體識別 (NER)

爲了從文本中提取 OD、OS 和 IOP 信息，我們可以使用 Hugging Face 的 `transformers` 庫，加載預訓練的 BERT 模型并進行微調。

#安裝依賴庫

pip install transformers
"""

#代碼示例：提取 OD、OS 和 IOP 信息

import torch
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline

# 加載預訓練的BERT模型和分詞器（帶NER任務）
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=3)  # 3表示標簽數量，OD, OS, IOP

text = "Objective: OD 24 mmHg, OS 22 mmHg, IOP within normal limits."   # 示例SOAP文本

輸入 = tokenizer(text, return_tensors="pt", truncation=True, padding=True) # 分詞和編碼

outputs = model(輸入) # 模型推理

predictions = torch.argmax(outputs.logits, dim=2) # 獲取預測的實體類別

tokens = tokenizer.convert_ids_to_tokens(輸入["input_ids"].squeeze().tolist()) # 獲取標注的實體名稱
labels = predictions.squeeze().tolist()

for token, label in zip(tokens, labels): # 輸出提取的結果
    if label == 1: entity = "OD"
    elif label == 2: entity = "OS"
    elif label == 3: entity = "IOP"
    else: entity = "O"
    print(f"Token: {token}, Entity: {entity}")
"""
在這裏，`num_labels` 是你的數據集中實體類別的數量。你可以根據不同的標注體系設置不同數量的標簽（比如 OD、OS、IOP），并通過 `BertForTokenClassification` 進行命名實體識別。

3. 自定義 Prompt 示例

Prompt 可以幫助你通過預訓練的模型快速提取所需的信息，以下是一些可能的自定義 Prompt：

- Prompt 1: `"Extract the OD, OS, and IOP values from the following SOAP note: {SOAP 文本}"`  
  用于從完整的 SOAP 文本中提取 OD、OS 和 IOP 值。

- Prompt 2: `"Find the IOP measurements for the left eye (OS) and right eye (OD) in this text: {SOAP 文本}"`  
  用于直接獲取眼內壓（IOP）的測量值。

4. 模型微調 你可以通過監督學習微調 BERT 模型，使用包含 OD、OS 和 IOP 標注的訓練數據。
通過 PyTorch 和 BERT 模型的結合，您可以輕松設計 prompt 來提取 SOAP 文本中的 OD（右眼）、OS（左眼）和 IOP（眼內壓）等關鍵信息。這對于自動化醫療文本處理、病歷分析或診斷輔助系統非常有用。
"""

from transformers import AdamW
from torch.utils.data import DataLoader

optimizer = AdamW(model.parameters(), lr=5e-5) # 定義優化器

train_dataset = CustomNERDataset(texts, labels, tokenizer, max_len=128) # 假設我們有一個自定義的數據集
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model.train() # 訓練過程
for epoch in range(3):  # 訓練3個epoch
    for batch in train_loader:
        optimizer.zero_grad()
        輸入 = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=輸入, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed.")

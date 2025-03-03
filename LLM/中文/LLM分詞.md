torch 訓練中文語言模型

訓練中文語言模型可以使用 PyTorch 這個深度學習框架。以下是一個基本的步驟指南，幫助你使用 PyTorch 訓練中文語言模型：
1. 環境準備 首先，你需要安裝必要的庫：

pip install torch transformers datasets

2. 數據集準備 選擇一個中文語料庫，例如中文維基百科或其他文本數據集。你可以使用 Hugging Face 的 datasets 庫來輕鬆下載和使用數據集。

from datasets import load_dataset

# 載入中文文本數據集（例如中文維基百科）
dataset = load_dataset('wikipedia', 'zh')

3. 數據預處理 將文本數據進行預處理，例如分詞、轉換為 ID 等。你可以使用 transformers 庫中的 Tokenizer。

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese') # 使用 BERT Tokenizer

def tokenize_function(examples):
    return tokenizer(examples['text']) # 對數據集進行分詞

tokenized_datasets = dataset.map(tokenize_function, batched=True)

4. 訓練模型 使用 transformers 中的 Trainer 來訓練模型。

from transformers import BertForMaskedLM, Trainer, TrainingArguments

model = BertForMaskedLM.from_pretrained('bert-base-chinese') # 載入預訓練的模型

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
) # 訓練參數設置

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
) # 初始化 Trainer

trainer.train() # 開始訓練

5. 評估和保存模型 訓練完成後，可以評估模型的性能並保存模型。

trainer.evaluate() # 評估模型

model.save_pretrained('./my_model') # 保存模型
tokenizer.save_pretrained('./my_model')

6. 使用模型進行預測 可以加載訓練好的模型進行文本生成或其他任務。
from transformers import pipeline

model = BertForMaskedLM.from_pretrained('./my_model') # 加載模型和 Tokenizer
tokenizer = BertTokenizer.from_pretrained('./my_model')

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer) # 創建文本生成管道
result = fill_mask("今天是[MASK]天。") # 進行預測
print(result)

**************************************
使用BERT進行中文斷詞是一個較爲常見的任務。BERT通過其上下文語義理解能力，可以很好地捕捉文本中的斷詞邊界。中文斷詞是一個序列標注問題，通常采用BERT進行序列標注，每個字對應一個標簽。標簽可以是`B`（詞的開始）、`I`（詞的內部）、`O`（其他）。

接下來，我們使用預訓練的中文BERT模型來進行中文斷詞任務。

基本步驟：
1. 加載預訓練的BERT模型：使用`transformers`庫中的預訓練BERT模型，如`bert-base-chinese`。
2. 數據預處理：將句子轉化爲BERT的輸入格式，包括tokenization、attention masks等。
3. 構建模型：在BERT的基礎上添加一個線性分類層，用于預測每個字的標簽（B、I、O）。
4. 訓練和推理：使用訓練數據進行模型微調，然後對新句子進行斷詞。

代碼實現步驟：

1. 安裝依賴庫
首先需要安裝`transformers`和`torch`：
```bash
pip install torch transformers

2. 加載預訓練BERT模型和數據預處理

from transformers import BertTokenizer, BertModel, BertForTokenClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW as optimAdamW
from torch.nn import CrossEntropyLoss

# 加載預訓練的BERT模型和分詞器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=3)  # 3個標簽: B, I, O

# 標簽映射
tag2idx = {'B': 0, 'I': 1, 'O': 2}
idx2tag = {0: 'B', 1: 'I', 2: 'O'}

# 示例數據集
class WordSegmentationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# 示例輸入文本（已進行BERT tokenization的token ID）
texts = [
    tokenizer.encode("我愛北京天安門", add_special_tokens=True),  # [CLS]我愛北京天安門[SEP]
    tokenizer.encode("今天天氣很好", add_special_tokens=True)    # [CLS]今天天氣很好[SEP]
]
# 對應標簽
labels = [
    [tag2idx['B'], tag2idx['I'], tag2idx['B'], tag2idx['I'], tag2idx['I'], tag2idx['B'], tag2idx['O']],  # 我 B 愛 I 北京 B 天 I 安 I 門 B
    [tag2idx['B'], tag2idx['I'], tag2idx['B'], tag2idx['I'], tag2idx['O']]  # 今天 B 天 I 氣 B 很 I 好 O
]

# 數據集與DataLoader
dataset = WordSegmentationDataset(texts, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

3. 構建并訓練模型

# 損失函數和優化器
optimizer = optimAdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()
from torch.nn.utils.rnn import pad_sequence
# 訓練模型
for epoch in range(3):
    model.train()
    for batch in dataloader:
        輸入, labels = batch
        輸入 = pad_sequence([trchTnsr(t) for t in inputs], batch_first=True)  # 填充序列
        labels = pad_sequence([trchTnsr(l) for l in labels], batch_first=True)  # 填充標簽
        
        輸出 = model(輸入, labels=labels) # 前向傳播
        loss = 輸出.loss

        optimizer.zero_grad() # 反向傳播和優化
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


from torch import no_grad as trchGrad, tensor as trchTnsr
# 推理函數 4. 推理與分詞 在推理階段，給定輸入句子，模型將預測每個字的分詞標簽。
def predict(sentence):
    model.eval()
    with trchGrad():
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        輸入 = trchTnsr([tokens])
        輸出 = model(輸入).logits
        predictions = 輸出.argmax(dim=-1).tolist()[0]

        # 去掉 [CLS] 和 [SEP] 特殊token的標簽
        predictions = predictions[1:-1]
        tokens = tokenizer.convert_ids_to_tokens(tokens[1:-1])

        # 輸出分詞結果
        segmented_sentence = ""
        for token, tag in zip(tokens, predictions):
            if idx2tag[tag] == 'B':
                segmented_sentence += " " + token
            else:
                segmented_sentence += token
        return segmented_sentence.strip()

# 示例推理
sentence = "我愛北京天安門"
segmented_sentence = predict(sentence)
print("分詞結果:", segmented_sentence)

輸出：
分詞結果: 我 愛 北京 天安門

代碼要點：
1. BERT模型與數據預處理：通過`transformers`庫中的`BertTokenizer`進行分詞，并使用`BertForTokenClassification`進行序列標注任務。
2. 模型微調：在BERT的基礎上添加一個分類頭（線性層），用于預測每個字的分詞標簽（B、I、O），然後用中文標注數據集進行微調。
3. 推理與分詞：推理時對每個字進行分詞標簽預測，根據預測結果還原分詞後的句子。

注意事項：
1. 數據集：你可以使用開源的中文分詞數據集，如MSRA或PKU分詞數據集來進行訓練。
2. 預訓練模型：可以使用更多預訓練的BERT模型，如`roberta-base-chinese`等，以提升模型效果。
3. 微調：BERT需要通過適當的數據集進行微調，才能在特定的任務上取得良好表現。

通過這個過程，你可以使用BERT進行高質量的中文斷詞。如果數據規模較大，可以調整超參數如學習率、batch size等來提升模型性能。
*************************************************
PyTorch 中文斷詞 自動作出 labels
在 PyTorch 中進行中文斷詞（分詞）任務時，可以使用預訓練模型（如 BERT）來自動化標注（labels）的生成。一般的分詞任務是將連續的文本划分為多個詞語，因此常見的標注方案是使用 BIO（Begin, Inside, Outside） 編碼或類似的標注方式，以識別每個字在詞語中的角色：
    • B（Begin）：表示詞語的開頭。
    • I（Inside）：表示詞語的中間部分。
    • O（Outside）：表示不屬於任何詞語的字符（例如標點符號等）。
以下是如何使用 PyTorch 和中文預訓練 BERT 模型來進行自動標注和分詞的具體步驟。
步驟概覽
    1. 准備數據：加載中文文本并預處理。
    2. 分詞模型：使用預訓練的中文 BERT 模型。
    3. 自動生成標注（labels）：根據標記方式自動生成 BIO 標注。
    4. 模型訓練：用 BERT 模型進行訓練或微調以進行斷詞任務。
    5. 預測與后處理：用訓練好的模型進行分詞預測。
1. 准備中文文本數據 假設你有一段中文文本，需要對其進行自動分詞并生成標注：
text = "今天天氣很好，我想去公園散步。"
2. 使用 BERT 模型
可以使用 Hugging Face 的 transformers 庫加載預訓練的中文 BERT 模型，用於斷詞任務。BERT 是一個強大的語言模型，它通過上下文來理解每個字符或詞在句子中的角色。
pip install transformers
from transformers import BertTokenizer, BertForTokenClassification
import torch

# 加載中文 BERT 模型和 tokenizer
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=3)  # 3 類：B, I, O
3. 自動生成 BIO 標注
為了生成訓練數據的標注，我們需要為每個字標注它在詞語中的角色。假設我們有人工分詞結果為：
# 已知分詞結果
segmented_text = ["今天", "天氣", "很好", "，", "我", "想", "去", "公園", "散步", "。"]

# 生成 BIO 標注
def generate_bio_labels(segmented_text):
    labels = []
    for word in segmented_text:
        if len(word) == 1: labels.append("O")
        else:
            labels.append("B")
            labels.extend(["I"] * (len(word) - 1))
    return labels

bio_labels = generate_bio_labels(segmented_text)
print(bio_labels)
# 輸出: ['B', 'I', 'B', 'I', 'O', 'O', 'O', 'B', 'I', 'O']
4. 數據編碼和模型訓練 將中文文本分詞，并根據 BIO 標注生成相應的訓練數據。首先需要將文本轉化為 BERT 可以處理的輸入格式。
inputs = tokenizer(text, return_tensors="pt", is_split_into_words=False, truncation=True, padding=True) # 對原始文本進行編碼

label_map = {"O": 0, "B": 1, "I": 2} # BIO 標簽映射為數值
labels = [label_map[label] for label in bio_labels]

labels_tensor = torch.tensor([labels], dtype=torch.long) # 將標簽轉化為 tensor
然后我們可以進行模型的前向傳播，并計算損失。我們會對每個字符進行分類（B, I, O），因此這個任務是一個 Token-Level Classification（標記級分類） 問題。
# 前向傳播，計算損失
outputs = model(**inputs, labels=labels_tensor)
loss = outputs.loss
logits = outputs.logits

print(f"Loss: {loss.item()}")
5. 訓練與微調模型 在斷詞任務中，模型的訓練過程與其他自然語言處理任務類似。可以使用優化器如 Adam 來更新模型參數，進行多次迭代以提高模型的性能。

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5) # 定義優化器

# 訓練循環
for epoch in range(3):  # 假設訓練 3 個 epoch
    model.train()
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels_tensor) # 前向傳播與計算損失
    loss = outputs.loss
    loss.backward() # 反向傳播與參數更新
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
6. 預測與分詞 在訓練好模型后，可以使用它來對新的中文文本進行分詞預測。
model.eval() # 使用模型進行預測
with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=2) # 預測標簽

predicted_labels = predictions.squeeze().tolist() # 將預測結果映射回 BIO 標簽
inv_label_map = {v: k for k, v in label_map.items()}
predicted_bio_labels = [inv_label_map[label] for label in predicted_labels]

print(f"Predicted BIO Labels: {predicted_bio_labels}") # 打印預測的 BIO 標簽
根據 BIO 標注結果，可以將文本拆分成對應的詞語。
# 將 BIO 標簽映射回詞語
def bio_to_words(text, bio_labels):
    words = []
    current_word = ""
    for char, label in zip(text, bio_labels):
        if label == "B":
            if current_word:
                words.append(current_word)
            current_word = char
        elif label == "I":
            current_word += char
        else:
            if current_word:
                words.append(current_word)
            current_word = ""
            words.append(char)
    
    if current_word:
        words.append(current_word)
    
    return words

words = bio_to_words(text, predicted_bio_labels) # 分詞結果
print(f"Segmented Words: {words}")
總結
    • BIO 標注生成：根據已分詞的文本，自動生成 BIO 標簽。
    • BERT 模型：使用預訓練的 BERT 模型處理中文文本，進行標記級別的分類。
    • 訓練與預測：訓練模型后，可以對新的文本進行斷詞預測，并基於預測結果生成分詞結果。
這種方法可以處理大量中文文本的斷詞任務，特別是基於 BERT 的模型可以捕捉上下文關系，有助於提高分詞的准確性。

********************************************************************************
from transformers import BertTokenizer, BertModel

# 加載BERT中文分詞器和模型
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

#3. 數據準備 準備你的中文文本數據，并進行分詞處理。可以使用分詞器將文本轉換爲模型可以理解的輸入格式。
示例：文本編碼
sentences = ["白日依山盡", "黃河入海流"] # 準備中文句子

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt') # 使用分詞器進行編碼

#4. 模型推理 將編碼後的輸入傳遞給模型，獲取輸出。 示例：獲取模型輸出
with torch.no_grad():  # 不計算梯度
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state # 獲取最後一層隱藏狀態
print(last_hidden_states)

#5. 微調模型（可選） 如果你有特定的任務（如文本分類、命名實體識別等），可以在此基礎上進行微調。你需要準備好標注數據，并使用適當的損失函數和優化器進行訓練。
from transformers import Trainer, TrainingArguments

# 定義訓練參數
training_args = TrainingArguments( output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8, save_steps=10_000, save_total_limit=2)

# 創建Trainer實例并開始訓練
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,  # 你的訓練數據集)

trainer.train()

#6. 保存和加載模型 訓練完成後，可以保存微調後的模型，以便後續使用。 示例：保存和加載模型
model.save_pretrained('./my_model') # 保存模型
tokenizer.save_pretrained('./my_model')

model = BertModel.from_pretrained('./my_model') # 加載模型
tokenizer = BertTokenizer.from_pretrained('./my_model')

#通過以上步驟，你可以在Hugging Face上建立并使用中文語言模型。這些步驟涵蓋了從環境設置到模型推理及微調的全過程。根據具體需求，你可以調整數據集和訓練參數

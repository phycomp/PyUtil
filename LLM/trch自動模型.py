from transformers import AutoTokenizer

# 使用 BERT 作為基礎模型
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, is_split_into_words=True)
    
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # 忽略分詞後的填充詞
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # 對於分詞後的子詞部分設置忽略
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
2.2 分割訓練和測試集
將數據集劃分為訓練集和測試集：
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]
3. 構建 NER 模型
3.1 使用 Hugging Face 的 AutoModelForTokenClassification
我們可以使用 Hugging Face 的 AutoModelForTokenClassification 來訓練一個 NER 模型。這裡我們將基於 BERT 模型進行微調。
from transformers import AutoModelForTokenClassification

# 加載預訓練的 BERT 模型
model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(dataset['train'].features['ner_tags'].feature.names))
3.2 定義優化器和訓練參數
我們將使用 AdamW 作為優化器，並設置訓練參數。
from transformers import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
3.3 訓練模型
使用 PyTorch 的訓練循環來訓練模型。
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
4. 評估模型
4.1 使用 seqeval 進行評估
seqeval 是一個專門用於序列標註任務（如 NER）的評估工具。以下是如何使用 seqeval 來評估模型性能。
pip install seqeval
計算 F1-score
from seqeval.metrics import classification_report
from transformers import BertTokenizerFast

model.eval()

predictions = []
true_labels = []

for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    true_labels.extend(batch["labels"].cpu().numpy())

# 將數值標籤轉換回實體標籤
label_list = dataset['train'].features['ner_tags'].feature.names

predicted_labels = [[label_list[p] for p in prediction] for prediction in predictions]
true_entity_labels = [[label_list[t] for t in true_label] for true_label in true_labels]

# 使用 seqeval 進行評估
print(classification_report(true_entity_labels, predicted_labels))
5. 推理與使用
一旦模型訓練完成，你可以對新的醫療文本進行推理，找出實體。
def predict_ner(text):
    inputs = tokenizer(text.split(), return_tensors="pt", truncation=True, is_split_into_words=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    labels = [label_list[p] for p in predictions[0].cpu().numpy()]
    
    return list(zip(tokens, labels))

# 測試推理
text = "The patient was diagnosed with pneumonia and prescribed aspirin."
print(predict_ner(text))
這樣，你的模型將對給定的醫療文本進行實體識別，並輸出每個詞的實體標籤。
6. 優化與改進
6.1 微調與增量訓練
可以通過調整學習率、批量大小和訓練步數來微調模型。對於小數據集，可以使用 增量訓練，即從預訓練模型開始，然後在醫療文本上進行進一步訓練。
6.2 混合精度訓練
使用 混合精度訓練 可以減少顯存佔用，並加速訓練。PyTorch 提供了 torch.cuda.amp 來實現這一點：
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_dataloader:
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
6.3 使用預訓練的醫療模型
考慮使用專門針對醫療數據預訓練的 BERT 模型，例如 BioBERT 或 ClinicalBERT，這些模型在醫學領域的數據上進行了微調，能夠更好地處理醫療文本。
from transformers import AutoModelForTokenClassification

# 使用預訓練的 BioBERT
model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=num_labels)

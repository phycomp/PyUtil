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

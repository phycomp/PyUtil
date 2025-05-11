# 使用 PyTorch 開發 LLM 模型進行 CSV 欄位比對

在處理結構化數據時，使用大型語言模型(LLM)來比對CSV欄位是一個實用的應用場景。以下是一個基於PyTorch的實現方案：

## 基本架構

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CSVFieldMatcher:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    def match_fields(self, df1, df2, threshold=0.8):
        # 獲取兩個DataFrame的欄位名稱
        fields1 = df1.columns.tolist()
        fields2 = df2.columns.tolist()
        
        # 獲取欄位名稱的嵌入向量
        emb1 = self.get_embeddings(fields1)
        emb2 = self.get_embeddings(fields2)
        
        # 計算相似度矩陣
        sim_matrix = cosine_similarity(emb1, emb2)
        
        # 找出最佳匹配
        matches = []
        for i, field1 in enumerate(fields1):
            best_match_idx = np.argmax(sim_matrix[i])
            best_score = sim_matrix[i][best_match_idx]
            if best_score >= threshold:
                matches.append({
                    'field1': field1,
                    'field2': fields2[best_match_idx],
                    'similarity': best_score
                })
        
        return pd.DataFrame(matches)
```

## 使用範例

```python
# 讀取CSV文件
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')

# 初始化匹配器
matcher = CSVFieldMatcher('bert-base-uncased')  # 也可以使用其他預訓練模型

# 進行欄位比對
matches = matcher.match_fields(df1, df2, threshold=0.85)

# 顯示匹配結果
print(matches.sort_values('similarity', ascending=False))
```

## 進階改進

1. **使用專門的嵌入模型**：
   ```python
   # 使用專門為語義相似度訓練的模型
   matcher = CSVFieldMatcher('sentence-transformers/all-mpnet-base-v2')
   ```

2. **加入欄位內容特徵**：
   ```python
   def enhanced_match_fields(self, df1, df2, threshold=0.8, sample_size=5):
       fields1 = df1.columns.tolist()
       fields2 = df2.columns.tolist()
       
       # 為每個欄位創建描述性文本 (欄位名 + 樣本值)
       def create_descriptions(df):
           desc = []
           for col in df.columns:
               samples = ', '.join(str(x) for x in df[col].dropna().sample(min(sample_size, len(df))[:sample_size]))
               desc.append(f"Column name: {col}. Sample values: {samples}")
           return desc
       
       desc1 = create_descriptions(df1)
       desc2 = create_descriptions(df2)
       
       # 獲取嵌入向量
       emb1 = self.get_embeddings(desc1)
       emb2 = self.get_embeddings(desc2)
       
       # 其餘匹配邏輯相同...
   ```

3. **微調模型**：
   - 收集欄位匹配的標註數據
   - 在特定數據上微調模型以提高準確率

## 注意事項

1. 對於大型CSV文件，可以考慮分批處理嵌入計算
2. 相似度閾值需要根據具體應用場景調整
3. 某些欄位名稱可能非常簡短（如"ID"），這會影響匹配準確性，可以考慮加入欄位內容特徵

這種方法特別適用於需要整合多個來源的結構化數據，且欄位命名不一致的情況，如數據倉庫建設、數據集成等場景。

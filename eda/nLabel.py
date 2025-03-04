import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 讀取數據
df = pd.read_csv('your_data.csv')

# 識別數據中的分類型欄位
categorical_columns = df.select_dtypes(include=['object']).columns

# 檢查有多少獨特值的欄位
for column in categorical_columns:
    print(f"Column '{column}' unique values: {df[column].nunique()}")
2. 動態選擇 n 個最高標籤並檢查偏度
首先找出每個分類欄位中最常見的標籤，並且分析這些標籤的分佈。可以使用**偏度（skewness）**來判斷標籤是否存在不平衡情況。

# 動態找出每個欄位中前 n 個標籤
n = 5
top_n_labels_dict = {}

for column in categorical_columns:
    top_n_labels = df[column].value_counts().nlargest(n)
    top_n_labels_dict[column] = top_n_labels
    
    # 可視化分佈及檢查是否有偏度
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_n_labels.index, y=top_n_labels.values)
    plt.title(f"Top {n} labels for column '{column}'")
    plt.ylabel('Frequency')
    plt.xlabel(column)
    plt.xticks(rotation=45)
    plt.show()
    
    # 打印偏度信息
    skewness = top_n_labels.skew()
    print(f"Skewness of top {n} labels in column '{column}': {skewness}")
3. 多欄位聯合頻次分析（Crosstab）
對於多個分類欄位，可以進行聯合頻次分析來檢查不同欄位的標籤之間是否有關聯性。例如，可以計算兩個欄位的交叉表，以查看特定標籤組合的出現頻率。

# 進行兩個欄位之間的聯合頻次分析（Crosstab）
column_1 = 'column_a'
column_2 = 'column_b'

crosstab_result = pd.crosstab(df[column_1], df[column_2])
print(crosstab_result)

# 視覺化聯合頻次表
plt.figure(figsize=(10, 6))
sns.heatmap(crosstab_result, annot=True, fmt="d", cmap='Blues')
plt.title(f'Crosstab between {column_1} and {column_2}')
plt.show()
這會生成一個交叉表，並將其視覺化為熱圖，展示兩個欄位之間標籤聯動的頻率。這對於多標籤欄位之間的聯合分佈分析很有用。

4. 標籤組合分析
如果數據集中有多個標籤欄位，則可以分析這些標籤欄位之間的組合情況，找出出現頻次最高的標籤組合。

# 假設我們有多個分類欄位，將它們組合在一起進行分析
label_columns = ['label_column_1', 'label_column_2', 'label_column_3']

# 創建一個新欄位表示標籤的組合
df['label_combination'] = df[label_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)

# 找出頻次最高的標籤組合
top_label_combinations = df['label_combination'].value_counts().nlargest(n)

# 可視化標籤組合的頻次
plt.figure(figsize=(10, 6))
sns.barplot(x=top_label_combinations.index, y=top_label_combinations.values)
plt.title('Top Label Combinations')
plt.ylabel('Frequency')
plt.xlabel('Label Combination')
plt.xticks(rotation=90)
plt.show()
這樣你就可以找出多個標籤欄位中哪些組合出現的頻次最高，這在分析多標籤分類問題時特別有用。

5. 分析標籤與數值特徵的關聯性
除了分析標籤之間的關聯性外，還可以分析分類型標籤與數值特徵的關聯性。例如，檢查不同標籤對應的數值特徵的分佈情況。

# 選擇一個標籤欄位和數值欄位進行關聯性分析
label_column = 'label_column_1'
numeric_column = 'numeric_column'

plt.figure(figsize=(10, 6))
sns.boxplot(x=df[label_column], y=df[numeric_column])
plt.title(f'Distribution of {numeric_column} by {label_column}')
plt.show()

output_json_path = '/mnt/data/識別_轉換.json'

# 修正後的程式碼，用於 CSV 轉嵌套 JSON
def csv_to_nested_json(csv_file_path, output_path):
    import pandas as pd
    import json

    # 讀取 CSV
    csv_data = pd.read_csv(csv_file_path)
    paths = csv_data['Id'].dropna().tolist()

    # 構建嵌套 JSON 的函數
    def build_nested_dict(paths):
        nested_dict = {}
        for path in paths:
            keys = path.split('.')
            current_level = nested_dict
            for key in keys[:-1]:  # 遍歷中間層
                if not isinstance(current_level.get(key), dict):
                    current_level[key] = {}
                current_level = current_level[key]
            current_level[keys[-1]] = None  # 最后一層設置為 None
        return nested_dict

    # 轉換為嵌套 JSON 格式
    nested_json = build_nested_dict(paths)

    # 將嵌套 JSON 寫入檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nested_json, f, indent=4, ensure_ascii=False)

    return output_path

# 執行 CSV 到 JSON 的轉換
csv_to_nested_json('/mnt/data/識別.csv', output_json_path)

output_json_path
*****************************
import pandas as pd
import json

# 修正後的程式碼，用於 CSV 轉嵌套 JSON
def csv_to_nested_json(csv_file_path, output_path):
    # 讀取 CSV
    csv_data = pd.read_csv(csv_file_path)
    paths = csv_data['Id'].dropna().tolist()

    # 構建嵌套 JSON 的函數
    def build_nested_dict(paths):
        nested_dict = {}
        for path in paths:
            keys = path.split('.')
            current_level = nested_dict
            for key in keys[:-1]:  # 遍歷中間層
                if not isinstance(current_level.get(key), dict):
                    current_level[key] = {}
                current_level = current_level[key]
            current_level[keys[-1]] = None  # 最后一層設置為 None
        return nested_dict

    # 轉換為嵌套 JSON 格式
    nested_json = build_nested_dict(paths)

    # 將嵌套 JSON 寫入檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nested_json, f, indent=4, ensure_ascii=False)

    print(f"嵌套 JSON 文件已保存至: {output_path}")

# 文件路徑
csv_file_path = '識別.csv'  # 替換為您的 CSV 文件路徑
output_json_path = '識別_轉換.json'  # 替換為輸出的 JSON 文件路徑

# 執行轉換
csv_to_nested_json(csv_file_path, output_json_path)

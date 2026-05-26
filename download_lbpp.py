import json
import os
from datasets import load_dataset

def main():
    print("Đang tải dataset CohereForAI/lbpp từ HuggingFace...")
    # Bỏ qua script cũ của repo bằng cách trỏ thẳng vào file parquet
    ds = load_dataset("parquet", data_files="hf://datasets/CohereForAI/lbpp/python/test.parquet")
    
    # LBPP thường nằm ở split 'test'
    split = 'test' if 'test' in ds else 'train'
    data = ds[split]
    
    out_dir = "data/LBPP"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lbpp.jsonl")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print(f"✅ Đã tải thành công {len(data)} bài toán và lưu vào: {out_path}")

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import pickle

with open("all_data.pkl", "rb") as f:
    all_data = pickle.load(f)
    
with open("emotions.txt", "r", encoding="utf-8") as f:
    emotions = [line.strip() for line in f.readlines()]

emotions = emotions[:28]   # ровно 28 эмоций
num_classes = len(emotions)

print(f"Эмоций загружено: {num_classes}")

# 2. MULTI-HOT функция
def to_multi_hot(value, num_classes):
    vec = np.zeros(num_classes, dtype=int)

    # value может быть числом, а может быть строкой типа "8,20"
    if isinstance(value, str):
        idx_list = value.split(",")
    else:
        idx_list = [value]

    for idx in idx_list:
        idx = int(idx.strip())
        if 0 <= idx < num_classes:
            vec[idx] = 1
        else:
            print(f"⚠️ Предупреждение: индекс {idx} вне диапазона (0..{num_classes-1})")

    return vec

# 3. Применяем multi-hot
print("🔄 Преобразую emotion_id → multi-hot...")
all_data["emotion_onehot"] = all_data["emotion_id"].apply(lambda x: to_multi_hot(x, num_classes))

# 4. Удаляем старый столбец
all_data = all_data.drop(columns=["emotion_id"])

print(all_data.head())

# 5. Сохраняем PKL
with open("all_data.pkl", "wb") as f:
    pickle.dump(all_data, f)

print("✅ Файл обновлён: all_data.pkl")

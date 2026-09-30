from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
import os

model = SentenceTransformer('./encoder')
files = ['train.tsv', 'dev.tsv', 'test.tsv']
output_file = 'all_data.pkl'

#all_data = pd.read_csv('1.tsv', sep = '\t', header = None, usecols=[0,1], names= ['sentence', 'emotion_id'])

'''
all_data = pd.concat([
    pd.read_csv(f, sep='\t', header=None, usecols= [0,1], names=['sentence', 'emotion_id'])
    for f in files
], ignore_index=True)

print(f"Загружено {len(all_data)} строк")


if os.path.exists(output_file):
    print("✅ Найден сохранённый файл. Загружаю...")
    with open(output_file, 'rb') as f:
        all_data = pickle.load(f)
else:
    print("⚙️ Файл не найден. Создаю эмбеддинги...")

    sentences = all_data['sentence'].tolist()
    embeddings = []

total = len(sentences)
for i, sentence in enumerate(sentences, 1):
    if i % 100 == 0 or i == total:
        print(f"→ Обработка {i}/{total}")
    emb = model.encode(sentence)
    embeddings.append(emb)

all_data['embedengs'] = embeddings

all_data = all_data.drop(columns=['sentence'])

with open(output_file, 'wb') as f:
    pickle.dump(all_data, f)
print("✅ Эмбеддинги сохранены в", output_file)

print("Пример данных:")
print(all_data.head())
'''


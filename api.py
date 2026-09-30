from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# ----------------------
# МОДЕЛЬ MLP
# ----------------------
class EmotionMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=28):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# ----------------------
# ЗАПУСК МОДЕЛИ
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionMLP()
model.load_state_dict(torch.load("mlp_multilabel_emotions.pth", map_location=device))
model.to(device)
model.eval()

embedder = SentenceTransformer("./encoder")

# ----------------------
# ЭМОЦИИ
# ----------------------
emotions_en = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

emotions_ru = {
    "admiration": "восхищение",
    "amusement": "развлечение",
    "anger": "гнев",
    "annoyance": "раздражение",
    "approval": "одобрение",
    "caring": "забота",
    "confusion": "замешательство",
    "curiosity": "любопытство",
    "desire": "желание",
    "disappointment": "разочарование",
    "disapproval": "неодобрение",
    "disgust": "отвращение",
    "embarrassment": "смущение",
    "excitement": "волнение",
    "fear": "страх",
    "gratitude": "благодарность",
    "grief": "горе",
    "joy": "радость",
    "love": "любовь",
    "nervousness": "тревога",
    "optimism": "оптимизм",
    "pride": "гордость",
    "realization": "осознание",
    "relief": "облегчение",
    "remorse": "раскаяние",
    "sadness": "печаль",
    "surprise": "удивление",
    "neutral": "нейтрально"
}

# ----------------------
# ФУНКЦИИ ПРЕДСКАЗАНИЯ
# ----------------------
# ----------------------
# ФУНКЦИИ ПРЕДСКАЗАНИЯ С ОКРУГЛЕНИЕМ
# ----------------------
def predict_emotions(texts, threshold=0.5):
    embeddings = embedder.encode(texts, convert_to_tensor=True).to(device)

    with torch.no_grad():
        outputs = model(embeddings)
        probs = torch.sigmoid(outputs).cpu().numpy()

    results = []
    for prob in probs:
        # округляем все вероятности до двух знаков
        all_probs = {emotions_ru[emotions_en[i]]: round(float(prob[i]), 3) for i in range(len(prob))}
        labels = [emo for emo, p in all_probs.items() if p >= threshold]
        results.append({
            "emotions": labels,
            "probabilities": all_probs
        })
    return results

# FASTAPI
# ----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # разрешает всем источникам
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Text(BaseModel):
    text: str

class TextList(BaseModel):
    texts: List[str]


# ----------------------
# ENDPOINTS
# ----------------------

@app.post("/predict")
def predict_one(request: Text):
    result = predict_emotions([request.text])[0]
    return result


@app.post("/predict_many")
def predict_many(request: TextList):
    result = predict_emotions(request.texts)
    return result


def get_page_emotion(all_post_results):
    """
    all_post_results — список результатов predict_emotions для каждого поста
    """
    # Собираем вероятности всех постов в матрицу
    all_probs_matrix = np.array([list(post["probabilities"].values()) for post in all_post_results])

    # Усредняем вероятности по всем постам
    page_probs = np.mean(all_probs_matrix, axis=0)

    # Находим индекс главной эмоции
    main_emotion_idx = np.argmax(page_probs)

    main_emotion = emotions_ru[emotions_en[main_emotion_idx]]
    main_emotion_prob = round(float(page_probs[main_emotion_idx]), 2)

    return {"main_emotion": main_emotion, "probability": main_emotion_prob}


@app.post("/predict_many_with_page")
def predict_many_with_page(request: TextList):
    all_post_results = predict_emotions(request.texts)
    page_result = get_page_emotion(all_post_results)

    return {
        "page_result": page_result,
        "posts_result": all_post_results
    }

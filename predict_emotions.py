import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionMLP(input_dim=768).to(device)
model.load_state_dict(torch.load("mlp_multilabel_emotions.pth", map_location=device))
model.eval()

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

embedder = SentenceTransformer('./encoder')

def predict_emotions(texts, threshold=0.5):
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    embeddings = embeddings.to(device)

    with torch.no_grad():
        outputs = model(embeddings)
        probs = torch.sigmoid(outputs).cpu().numpy()

    results = []
    for prob in probs:
        labels = [emotions_ru[emotions_en[i]] for i, p in enumerate(prob) if p >= threshold]
        all_probs = {emotions_ru[emotions_en[i]]: float(prob[i]) for i in range(len(prob))}
        results.append({"emotions": labels, "probabilities": all_probs})
    return results

if __name__ == "__main__":
    examples = [
        "какая же невероятно красивая была осень. как же красиво. ходишь и отдыхаешь, наслаждаешься этой невероятной красотой. просто сказка. я в шоке"

    ]

    preds = predict_emotions(examples, threshold=0.25)
    for text, result in zip(examples, preds):
        print(f"\nТекст: {text}")
        print(f"Предсказанные эмоции (выше порога): {result['emotions']}")
        print("Все эмоции с вероятностями:")
        for emo, prob in result['probabilities'].items():
            print(f"  {emo}: {prob:.3f}")

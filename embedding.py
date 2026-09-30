import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer

FILES = ["train.tsv", "dev.tsv", "test.tsv"]
OUTPUT_FILE = "all_data.pkl"


def main():
    model = SentenceTransformer("./encoder")

    data = pd.concat(
        [
            pd.read_csv(
                path,
                sep="\t",
                header=None,
                usecols=[0, 1],
                names=["sentence", "emotion_id"],
            )
            for path in FILES
        ],
        ignore_index=True,
    )
    print(f"Загружено строк: {len(data)}")

    embeddings = model.encode(
        data["sentence"].tolist(), batch_size=64, show_progress_bar=True
    )

    # Имя колонки "embedengs" оставлено как есть: его ждёт train_mlp.py.
    data["embedengs"] = list(embeddings)
    data = data.drop(columns=["sentence"])

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"Эмбеддинги сохранены в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

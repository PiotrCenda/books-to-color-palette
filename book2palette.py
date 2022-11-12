import os
from transformers import pipeline

emotions = ["love", "admiration", "joy", "pproval", "caring", "excitement", "amusement", "gratitude", "desire", "anger", "optimism", "disapproval", "grief", "annoyance", "pride", "curiosity", "neutral", "disgust", "disappointment", "realization", "fear", "relief", "confusion", "remorse", "embarrassment", "surprise", "sadness, nervousness]

def load_all_texts(texts_folder_path: str = "./data/texts"):
    txt_paths = [os.path.join(texts_folder_path, file) for file in os.listdir(texts_folder_path)]
    text = []
    
    for path in txt_paths:
        with open(path, "r") as file:
            text.append(file.readlines()[0])

    return text


if __name__ == "__main__":
    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    emotion_labels = emotion("I hate mondays, pls kill me")
    print(emotion_labels)

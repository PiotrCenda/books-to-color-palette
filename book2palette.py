import os
from tqdm import tqdm
from pathlib import Path
from transformers import pipeline

from text_clean import get_only_not_converted


emotions = ["love", "admiration", "joy", "approval", "caring", "excitement", "amusement", 
            "gratitude", "desire", "anger", "optimism", "disapproval", "grief", "annoyance", 
            "pride", "curiosity", "neutral", "disgust", "disappointment", "realization", 
            "fear", "relief", "confusion", "remorse", "embarrassment", "surprise", 
            "sadness", "nervousness"]


def analyze_all_texts(texts_folder_path: str = "./data/texts", output_path: str = "./data/emotions"):
    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa', top_k=3)
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    txt_paths = get_only_not_converted(path_from=texts_folder_path, path_to=output_path, to_prefix="analyzed_")
    filenames = [filepath.split(r"/")[-1] for filepath in txt_paths]
    emotions_paths = [os.path.join(output_path, "analyzed_" + file) for file in filenames]
    
    for txt_path, emotion_path in zip(txt_paths, emotions_paths):
        with open(txt_path, "r") as file:
            print(f"\nAnalyzing {txt_path}\n")
            
            text = file.readlines()[0].split()
            
            with open(emotion_path, "w") as outfile:
                for i in tqdm(range((len(text) // 100) - 1)):
                    text_chunk = " ".join(text[i:i+100])
                    detected_emotions = emotion(text_chunk)
                    outfile.write(str(detected_emotions) + "\n")


if __name__ == "__main__":
    analyze_all_texts()

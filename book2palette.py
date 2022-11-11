import os
import text2emotion as te


def load_all_texts(texts_folder_path: str = "./data/texts"):
    txt_paths = [os.path.join(texts_folder_path, file) for file in os.listdir(texts_folder_path)]
    
    for path in txt_paths:
        with open(path, "r") as file:
            text = file.readlines()
            print(type(text))
        break


if __name__ == "__main__":
    load_all_texts()

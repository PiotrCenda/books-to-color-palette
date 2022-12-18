import os
import re
import string
from tqdm import tqdm
from pathlib import Path
from PyPDF2 import PdfFileReader


def get_only_not_converted(path_from: str, path_to: str, to_prefix: str = ""):
    Path(path_to).mkdir(parents=True, exist_ok=True)
    return [os.path.join(path_from, file) for file in os.listdir(path_from) 
            if (to_prefix + os.path.splitext(file)[0]) not in 
            [os.path.splitext(filename)[0] for filename in os.listdir(path_to)]]


def extract_text_from_pdf(path: str):
    with open(path, 'rb') as file:
        pdf = PdfFileReader(file)
        text = []
        
        for page in range(pdf.numPages):
            pageObject = pdf.getPage(page)
            text.append(pageObject.extract_text())

        txt_path = os.path.join("./data/texts", Path(path).stem + ".txt")
        
        if os.path.exists(txt_path):
            os.remove(txt_path)

        with open(txt_path, 'a', encoding="utf-8") as file2:
            file2.writelines(text)


def convert_all_pdfs_to_txt(pdfs_folder_path: str = "./data/pdfs"):
    pdfs_paths = get_only_not_converted(path_from=pdfs_folder_path, path_to="./data/texts")
    
    for path in tqdm(pdfs_paths, desc="Converting files from pdf to txt"):
        extract_text_from_pdf(path)


def clean_text(text: str):
    text.replace('\n', ' ').replace('\r', '')
    text = re.sub('\s+',' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    return text.lower()


def clean_all_texts(texts_folder_path: str = "./data/texts"):
    txt_paths = [os.path.join(texts_folder_path, file) for file in os.listdir(texts_folder_path)]
    
    for path in tqdm(txt_paths, desc="Cleaning files"):
        text = []
        
        with open(path, 'r') as file:
            lines = file.readlines()
            text = [" " + line.strip() for line in lines]
            text = ''.join(text)[1:]            
            text = clean_text(text)

        with open(path, 'w') as file:
            file.write(text)


if __name__ == "__main__":
    convert_all_pdfs_to_txt()
    clean_all_texts()
    
import os
import re
import string
from pathlib import Path
from PyPDF2 import PdfFileReader


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


def convert_all_pdfs_to_txt():
    pdfs_folder_path = "./data/pdfs"
    
    pdfs_paths = [os.path.join(pdfs_folder_path, file) for file in os.listdir(pdfs_folder_path)]
    
    for path in pdfs_paths:
        extract_text_from_pdf(path)


def clean_text(text: str):
    text.replace('\n', ' ').replace('\r', '')
    text = re.sub('\s+',' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    return text.lower()


def clean_all_texts(texts_folder_path: str = "./data/texts"):
    txt_paths = [os.path.join(texts_folder_path, file) for file in os.listdir(texts_folder_path)]
    
    for path in txt_paths:
        text = []
        
        with open(path, 'r') as file:
            lines = file.readlines()
            text = [" " + line.strip() for line in lines]
            text = ''.join(text)[1:]            
            text = clean_text(text)

        with open(path, 'w') as file:
            file.write(text)


if __name__ == "__main__":
    # convert_all_pdfs_to_txt()
    clean_all_texts()
    
    
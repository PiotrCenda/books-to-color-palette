import os
from pathlib import Path
from PyPDF2 import PdfFileReader


def extract_text_from_pdf(path: str):
    
    with open(path, 'rb') as file:
        pdf = PdfFileReader(file)
        text = []
        
        for page in range(pdf.numPages):
            pageObject = pdf.getPage(page)
            text.append(pageObject.extractText())

        txt_path = os.path.join("./data/texts", Path(path).stem + ".txt")
        
        if os.path.exists(txt_path):
            os.remove(txt_path)

        with open(txt_path, 'a', encoding="utf-8") as file2:
            file2.writelines(text)


def load_book(path: str):
    pass


if __name__ == "__main__":
    pdfs_folder_path = "./data/pdfs"
    texts_folder_path = "./data/texts"
    
    pdfs_paths = [os.path.join(pdfs_folder_path, file, ) for file in os.listdir(pdfs_folder_path)]
    
    for path in pdfs_paths:
        extract_text_from_pdf(path)
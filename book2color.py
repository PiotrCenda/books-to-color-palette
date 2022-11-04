import os
from pathlib import Path
from PyPDF2 import PdfFileReader


def extract_text_from_pdf(path: str):
    print(path)
    
    with open(path, 'r', encoding="utf8", errors="ignore") as file:
        pdf = PdfFileReader(file)
    
        num_of_pages = pdf.numPages
        print(num_of_pages)
        pages = pdf.getPage(num_of_pages) 
        text = pages.extractText()

        txt_path = os.path.join("./data/texts", Path(path).stem + ".txt")

        with open(txt_path, 'w', encoding="utf8") as file2:
            file2.writelines(text)


def load_book(path: str):
    pass


if __name__ == "__main__":
    pdfs_folder_path = "./data/pdfs"
    texts_folder_path = "./data/texts"
    
    pdfs_paths = [os.path.join(pdfs_folder_path, file, ) for file in os.listdir(pdfs_folder_path)]
    
    for path in pdfs_paths:
        extract_text_from_pdf(path)
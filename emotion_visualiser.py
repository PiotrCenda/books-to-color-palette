import numpy as np
import matplotlib.pyplot as plt
import os


COLORS = {"love" : (209, 239, 44), 
        "admiration" : (108, 236, 47), 
        "joy" : (249, 248, 113), 
        "approval" : (227, 251, 191), 
        "caring" : (231, 247, 140), 
        "excitement" : (251, 251, 164), 
        "amusement" : (253, 253, 204), 
        "gratitude" : (203, 248, 137), 
        "desire" : (255, 112, 93), 
        "anger" : (235, 10, 156), 
        "optimism" : (253, 233, 151), 
        "disapproval" : (101, 200, 233), 
        "grief" : (0, 137, 240), 
        "annoyance" : (250, 133, 209), 
        "pride" : (252, 179, 97), 
        "curiosity" : (255, 170, 157), 
        "neutral" : (100, 100, 100), 
        "disgust" : (166, 82, 221), 
        "disappointment" : (128, 198, 236), 
        "realization" : (172, 240, 234), 
        "fear" : (51, 232, 173), 
        "relief" : (247, 251, 196), 
        "confusion" : (125, 232, 208), 
        "remorse" : (141, 137, 228), 
        "embarrassment" : (132, 236, 207),
        "surprise" : (62, 221, 205), 
        "sadness" : (85, 183, 255), 
        "nervousness" : (223, 164, 238)}

def trim_line(text: str):
    for char in ['{', '[' , ']', '}', "'"]:
        if char in text:
            text = text.replace(char, '')
    text = text.split(',')[0]
    text = text.split(': ')[1]
    return text.lower()

def map_colors(txt_path: str = './data/emotions/analyzed_lolita.txt'):
    image = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            emotion = trim_line(line)
            image.append(np.array(COLORS[emotion]))

    image_path = './data/images/'
    book_name = txt_path.split('/')[-1][:-4]
    image_path = image_path + book_name + '.png'
    image = np.expand_dims(np.array(image), axis=0).astype(np.uint8)
    plt.imsave(image_path, image)
    print(image_path, 'saved!')
    
    
def convert_books_to_colors():
    for file in os.listdir('./data/emotions/'):
        map_colors('./data/emotions/' + file)


if __name__ == "__main__":
    convert_books_to_colors()
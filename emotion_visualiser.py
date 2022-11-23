import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


# seed na podstawie nazwy
# bloby na podstawie udziału emocji
# gradient noise
# 3d obraz
# wizualizacja gif/slider


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


STEPS = 1

def trim_line(text: str):
    for char in ['{', '[' , ']', '}', "'"]:
        if char in text:
            text = text.replace(char, '')
    text = text.split(',')[0]
    text = text.split(': ')[1]
    return text.lower()


def load_emotions_from_line(text: str):
    emotions = text.strip().replace("[", "").replace("]", "").split("}, {")
    
    for i in range(len(emotions)):
        if emotions[i][0] != "{":
            emotions[i] = "{" + emotions[i]
        if emotions[i][-1] != "}":
            emotions[i] = emotions[i] + "}"

        emotions[i] = json.loads(emotions[i].replace("'", "\""))
        
    result = list(filter(lambda x: x["label"] != "neutral", emotions))
    
    return result


def normalize(l):
    s = sum(l)
    return [i/s for i in l]


def mix_emotions(emotions: dict):
    img_strip = sorted(list(np.random.choice([em["label"] for em in emotions], 
                                             size=200, 
                                             replace=True, 
                                             p=normalize([em["score"] for em in emotions]))), 
                       key=lambda x: [e["score"] for e in emotions if e["label"] == x])

    return img_strip
    

def map_colors(txt_path: str):
    image = []
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            emotions_dict = load_emotions_from_line(line)
            emotions_strip = mix_emotions(emotions_dict)
            image.append(np.array([COLORS[em] for em in emotions_strip]))
    
    ### Interpolation ###
    #
    # for i in range(100):
    #     changes_idx = {}
    #     prev_color = image[0]
    #     changes_idx[0] = prev_color
        
    #     for i in range(1, len(image)):
    #         next_color = image[i]
    #         if tuple(prev_color) != tuple(next_color):
                
    #             xp = [0, 2 + STEPS]
    #             fp_r = [prev_color[0], next_color[0]]
    #             fp_g = [prev_color[1], next_color[1]]
    #             fp_b = [prev_color[2], next_color[2]]
    #             interp_r = np.interp([i+1 for i in range(STEPS)], xp, fp_r)
    #             interp_g = np.interp([i+1 for i in range(STEPS)], xp, fp_g)
    #             interp_b = np.interp([i+1 for i in range(STEPS)], xp, fp_b)
    #             interp = np.array([np.array([interp_r[i], interp_g[i], interp_b[i]]) for i in range(len(interp_r))])
    #             changes_idx[i] = interp

    #         else:
    #             changes_idx[i] = next_color
                
    #         prev_color = next_color
        
    #     image = []
        
    #     for key in changes_idx.keys():
    #         if len(changes_idx[key].shape) > 1:
    #             for c in changes_idx[key]:
    #                 image.append(np.array(c))
    #         else:
    #             image.append(np.array(changes_idx[key]))
    
    image_path = './data/images/'
    book_name = txt_path.split('/')[-1][:-4]
    image_path = image_path + book_name + '.png'
    image = np.array(image).astype(np.uint8)

    plt.imsave(image_path, image)
    print(image_path, 'saved!')
    
    
def convert_books_to_colors():
    for file in os.listdir('./data/emotions/'):
        map_colors('./data/emotions/' + file)


if __name__ == "__main__":
    convert_books_to_colors()
    
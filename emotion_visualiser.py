import os
import json
import numpy as np
import matplotlib.pyplot as plt


COLORS = {"love" : (255, 247, 228), 
        "admiration" : (135, 168, 137), 
        "joy" : (255, 247, 160), 
        "approval" : (0, 0, 0), 
        "caring" : (0, 0, 0), 
        "excitement" : (0, 0, 0), 
        "amusement" : (0, 0, 0), 
        "gratitude" : (0, 0, 0), 
        "desire" : (254, 170, 228), 
        "anger" : (249, 130, 132), 
        "optimism" : (0, 0, 0), 
        "disapproval" : (172, 204, 228), 
        "grief" : (0, 137, 240), 
        "annoyance" : (0, 0, 0), 
        "pride" : (233, 245, 157), 
        "curiosity" : (255, 170, 157), 
        "neutral" : (0, 0, 0), 
        "disgust" : (108, 86, 113), 
        "disappointment" : (0, 0, 0), 
        "realization" : (172, 204, 228), 
        "fear" : (176, 235, 147), 
        "relief" : (0, 0, 0), 
        "confusion" : (0, 0, 0), 
        "remorse" : (40, 40, 46), 
        "embarrassment" : (217, 200, 191),
        "surprise" : (179, 227, 218), 
        "sadness" : (0, 0, 0), 
        "nervousness" : (255, 195, 132)}


STEPS = 10

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
    
    if len(result) > 4:
        result = result[:4]
    
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
    
import os
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.image as mpimg
from scipy.spatial.distance import cdist

from emotion_visualiser import load_emotions_from_line, COLORS
from text_clean import get_only_not_converted
from perlin import generate_fractal_noise_3d


def remap_emotion(emotion):
    remap_dict = {
        "approval" : 'joy', 
        "caring" : 'love', 
        "excitement" : 'joy', 
        "amusement" : 'joy', 
        "gratitude" : 'admiration', 
        "optimism" : 'joy', 
        "annoyance" : 'anger',
        "disappointment" : 'grief', 
        "relief" : 'joy', 
        "confusion" : 'surprise', 
        "sadness" : 'grief'}

    if emotion in list(remap_dict.keys()):
        return remap_dict[emotion]
    else:
        return emotion


def make_gif(path, filename: str):
    frames_paths = sorted([os.path.join(path, name) for name in os.listdir(path)], key=len)
    frames = []

    for fpath in frames_paths:
        img = Image.open(fpath)
        frames.append(img)
        
    frame_one = frames[0]
    frame_one.save(os.path.join(path, filename + ".gif"), format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)
    
    print("Cleaning temp files...")
    
    for png_file in glob.glob(os.path.join(path, "*.png")):
        os.remove(png_file)
    

def save_gif(img, path, filename: str):
    name = path.split("\\")[-1][:-4]
    save_path = os.path.join('data', 'gifs', name)
    os.makedirs(save_path, exist_ok=True)
    
    for i, gif_slice in tqdm(enumerate(img), desc="Creating gif", total=len(img)):
        img_save_path = os.path.join(save_path, str(i) + '.png')
        mpimg.imsave(img_save_path, normalize(gif_slice[0]))

    make_gif(save_path, filename)
    

def distance_from_center(shape, center=None):
    if center is None:
        center = np.array([[shape[1]//2, shape[2]//2]])

    img = np.zeros((shape[1], shape[2]))
    
    for x in range(shape[1]):
        for y in range(shape[2]):
            point = np.array([[x, y]])
            img[x, y] = cdist(point, center)

    img = img/(np.max(img))
    img = np.array([img for _ in range(shape[0])])
    return img


def normalize(array):
    amin = np.min(array)
    amax = np.max(array)

    return (array - amin) / (amax - amin)


def fit_colors_length(color_smaller, colors_bigger):
    smaller_len = len(color_smaller)
    bigger_len = len(colors_bigger)
    step = bigger_len / smaller_len

    mapped_to_bigger = [[0, 0, 0] for _ in range(bigger_len)]
    sum_step = 0
    
    for i in range(smaller_len):
        mapped_to_bigger[int(sum_step)] = color_smaller[i]
        sum_step += step
    
    smaller_r = [c[0] for c in color_smaller]
    smaller_g = [c[1] for c in color_smaller]
    smaller_b = [c[2] for c in color_smaller]
    known_x = []
    unknown_x = []
    
    for i, mapped in enumerate(mapped_to_bigger):
        if mapped[0] == 0 and mapped[1] == 0 and mapped[2] == 0:
            unknown_x.append(i)
        else:
            known_x.append(i)

    interp_r = np.interp(unknown_x, known_x, smaller_r)
    interp_g = np.interp(unknown_x, known_x, smaller_g)
    interp_b = np.interp(unknown_x, known_x, smaller_b)
    for i, unknown in enumerate(unknown_x):
        mapped_to_bigger[unknown] = (interp_r[i], interp_g[i], interp_b[i])

    return mapped_to_bigger


def interpolate_color_list(colors, steps):
    interpolated_colors = []
    changes_idx = {}
    prev_color = colors[0]
    changes_idx[0] = prev_color
    
    for i in range(len(colors)):
        next_color = colors[i]
        
        if tuple(prev_color) != tuple(next_color):
            xp = [0, 2 + steps]
            fp_r = [prev_color[0], next_color[0]]
            fp_g = [prev_color[1], next_color[1]]
            fp_b = [prev_color[2], next_color[2]]
            interp_r = np.interp([i+1 for i in range(steps)], xp, fp_r)
            interp_g = np.interp([i+1 for i in range(steps)], xp, fp_g)
            interp_b = np.interp([i+1 for i in range(steps)], xp, fp_b)
            interp = np.array([np.array([interp_r[i], interp_g[i], interp_b[i]]) for i in range(len(interp_r))])
            changes_idx[i] = interp

        else:
            changes_idx[i] = next_color
            
        prev_color = next_color

    for key in changes_idx.keys():
            if len(np.array(changes_idx[key]).shape) > 1:
                for c in changes_idx[key]:
                    interpolated_colors.append(np.array(c))
            else:
                interpolated_colors.append(np.array(changes_idx[key]))

    return interpolated_colors




def map_emotions_3d(txt_path: str, filename: str, shape: tuple):
    thr1 = 0.60
    thr2 = 0.35
    STEPS = 10
    seed = int("".join([str(ord(char)) for char in txt_path])[:7])
    np.random.seed(seed=seed)

    with open(txt_path, 'r') as f:
        lines = f.readlines()

        colors1 = []
        colors2 = []
        colors_bg = []

        for i, line in enumerate(lines):
            emotions_dict = load_emotions_from_line(line)
            em_bg = remap_emotion(emotions_dict[0]['label'])
            em1 = remap_emotion(emotions_dict[1]['label'])
            em2 = remap_emotion(emotions_dict[2]['label'])
            colors_bg.append(COLORS[em_bg])
            colors1.append(COLORS[em1])
            colors2.append(COLORS[em2])

        ic1 = interpolate_color_list(colors1, STEPS)
        ic2 = interpolate_color_list(colors2, STEPS)
        ic_bg = interpolate_color_list(colors_bg, STEPS)
        print(len(ic_bg))
        print(len(ic1))
        print(len(ic2))

        colors_len_dict = {len(ic2) : ic2, len(ic1) : ic1, len(ic_bg) : ic_bg}
        colors_len_dict = dict(sorted(colors_len_dict.items()))

        colors_keys = list(colors_len_dict.keys())
        # print(colors_len_dict[colors_keys[0]])

        ic_bg = fit_colors_length(colors_len_dict[colors_keys[0]], colors_len_dict[colors_keys[2]])
        ic1 = fit_colors_length(colors_len_dict[colors_keys[1]], colors_len_dict[colors_keys[2]])
        ic2 = colors_len_dict[colors_keys[2]]

        print(len(ic_bg))
        print(len(ic1))
        print(len(ic2))
        
        if len(ic_bg) % 2 != 0:
            ic_bg = ic_bg[:-1]
            ic1 = ic1[:-1]
            ic2 = ic2[:-1]

        print(len(ic_bg))
        print(len(ic1))
        print(len(ic2))
        
        if len(ic1) < len(ic2):
            ic1 = fit_colors_length(ic1, ic2)
        else:
            ic2 = fit_colors_length(ic2, ic1)

    with open(txt_path, 'r') as f:
        lines = f.readlines()


        W, H = 512, 512
        in_shape = (len(ic1), W, H)
        pic1 = np.zeros(in_shape)
        frequencies = [1, 2]


        for f in tqdm(frequencies, desc="Generating fractal noise"):   
            pic1 += generate_fractal_noise_3d(in_shape, (f, f, f))/len(frequencies)
        
        pic2 = pic1.copy()
        new_pic1 = []
        new_pic2 = []

        for i, (c1, c2, c_bg, slice1, slice2) in tqdm(enumerate(zip(ic1, ic2, ic_bg, pic1, pic2)), desc="Generating vizualization", total=len(ic1)):
            emotions_dict = load_emotions_from_line(line)

            color1 = c1
            color2 = c2
            bg_color = c_bg

            in_shape_slice = (1, W, H)
            rotate_factor = 1/32
            distance = W * 7 / 3
            x1 = np.cos(np.pi + i*np.pi*rotate_factor) * distance
            x2 = np.cos(i*np.pi*rotate_factor) * distance
            y1 = np.sin(np.pi + i*np.pi*rotate_factor) * distance
            y2 = np.sin(i*np.pi*rotate_factor) * distance

            distance_map = distance_from_center(in_shape_slice, np.array([[W//2 + x1, H//2 + y1]]))
            distance_map2 = distance_from_center(in_shape_slice, np.array([[W//2 + x2, H//2 + y2]]))
            
            s = np.random.uniform(0, 1, slice1.shape)

            slice1 -= distance_map[0]
            slice2 -= distance_map2[0]
            slice1 = normalize(slice1) * (1 - (distance_map * s))
            slice2 = normalize(slice2)  * (1 - (distance_map2 * s ))

            slice1 = normalize(np.array(slice1 > thr1).astype(np.int8))
            slice2 = normalize(np.array(slice2 > thr2).astype(np.int8))
            bg_mask = 1 - ((slice1 + slice2) - (slice1 * slice2))

            bg = np.stack((bg_mask,)*3, axis=-1) * bg_color

            slice2 = slice2 - (slice2 * slice1)

            slice1 = np.stack((slice1,)*3, axis=-1) * color1
            slice2 = np.stack((slice2,)*3, axis=-1) * color2

            new_pic1.append(slice1 + bg) 
            new_pic2.append(slice2) 

        new_pic1 = np.array(new_pic1)
        new_pic2 = np.array(new_pic2)

        pic = new_pic1 + new_pic2

        save_gif(pic, txt_path, filename)


if __name__ == "__main__":
    paths = get_only_not_converted(path_from="./data/emotions", path_to="./data/gifs")
# <<<<<<< HEAD
    map_emotions_3d(txt_path='./data/emotions/analyzed_animal_farm.txt', filename="test_256.gif")
# =======
#     shape = (32, 32)
    
#     for path in paths:
#         try:
#             print(f"Creating vizualization for {path.split()[-1]}")
#             map_emotions_3d(txt_path=path, filename=str(shape[0]), shape=shape)
#         except Exception as e:
#             print("Ups: ", e)
# >>>>>>> 8507aad5a0d1a639af885b3f90649f3512fae5ba

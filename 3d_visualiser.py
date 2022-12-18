import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from emotion_visualiser import mix_emotions, load_emotions_from_line, COLORS
from perlin import generate_fractal_noise_3d
from text_clean import get_only_not_converted


def save_pic(img, path):
    p = Path(path)
    print(p.name, p.parent, p.parts[-2])
    print(p.resolve())
    print(p.stem)
    path = path.split('/')[-1][:-4]
    print(path)
    

def distance_from_center(shape, center = None):
    if center is None:
        center = np.array([[shape[1]//2, shape[2]//2]])

    img = np.zeros((shape[1], shape[2]))
    
    for x in range(shape[1]):
        for y in range(shape[2]):
            point = np.array([[x, y]])
            img[x, y] = cdist(point, center)

    img = img/(np.max(img))
    img = np.array([img for _ in range(shape[0])])
    # print(img.shape)
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


def map_emotions_3d(txt_path: str):
    thr1 = 0.75
    thr2 = 0.35
    STEPS = 5
    seed = int("".join([str(ord(char)) for char in txt_path])[:7])
    np.random.seed(seed=seed)

    with open(txt_path, 'r') as f:
        lines = f.readlines()

        colors1 = []
        colors2 = []
        for i, line in enumerate(lines):
            emotions_dict = load_emotions_from_line(line)
            colors1.append(COLORS[emotions_dict[0]['label']])
            colors2.append(COLORS[emotions_dict[1]['label']])

        print(len(colors1))
        print(len(colors2))

        ic1 = interpolate_color_list(colors1, STEPS)
        ic2 = interpolate_color_list(colors2, STEPS)
        print(len(ic1))
        print(len(ic2))

        if len(ic1) < len(ic2):
            ic1 = fit_colors_length(ic1, ic2)
        else:
            ic2 = fit_colors_length(ic2, ic1)

        print(len(ic1))
        print(len(ic2))

    with open(txt_path, 'r') as f:
        lines = f.readlines()

        W, H = 16, 16
        in_shape = (len(ic1), W, H)
        pic1 = np.zeros(in_shape)
        frequencies = [1]

        for f in frequencies:   
            pic1 += generate_fractal_noise_3d(in_shape, (f, f, f))/len(frequencies)
        
        pic2 = pic1.copy()
        new_pic1 = []
        new_pic2 = []

        for i, (c1, c2, slice1, slice2) in tqdm(enumerate(zip(ic1, ic2, pic1, pic2))):
            emotions_dict = load_emotions_from_line(line)

            color1 = c1
            color2 = c2
            bg_color = (0, 0, 0)

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

        save_pic(pic, txt_path)


if __name__ == "__main__":
    paths = get_only_not_converted(path_from="./data/emotions", path_to="./data/gifs")
    map_emotions_3d(txt_path='./data/emotions/analyzed_animal_farm.txt')

import matplotlib.pyplot as plt
import numpy as np
from perlin import generate_fractal_noise_3d
from scipy.spatial.distance import cdist
from tqdm import tqdm

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

def main():
    np.random.seed(seed=42)
    
    thr1 = 0.75
    thr2 = 0.35
    
    in_shape = (32, 256, 256)
    pic1 = np.zeros(in_shape)
    frequencies = [1, 2, 4, 8]

    for f in frequencies:   
        pic1 += generate_fractal_noise_3d(in_shape, (f, f, f))/len(frequencies)
    

    pic2 = pic1.copy()

    # pic1 = normalize(pic)
    # pic2 = normalize(pic2)
    new_pic1 = []
    new_pic2 = []
    new_pic = []
    for i, (slice1, slice2) in tqdm(enumerate(zip(pic1, pic2))):
        # print(i)
        in_shape_slice = (1, 256, 256) 
        rotate_factor = 1/16
        distance = 96
        x1 = np.cos(np.pi + i*np.pi*rotate_factor) * distance
        x2 = np.cos(i*np.pi*rotate_factor) * distance
        y1 = np.sin(np.pi + i*np.pi*rotate_factor) * distance
        y2 = np.sin(i*np.pi*rotate_factor) * distance

        distance_map = distance_from_center(in_shape_slice, np.array([[128 + x1, 128 + y1]]))
        distance_map2 = distance_from_center(in_shape_slice, np.array([[128 + x2, 128 + y2]]))
        
        s = np.random.uniform(0,2,slice1.shape)
        # s = np.stack((s,)*3, axis=-1)
        # plt.figure()
        # plt.imshow(s)
        # plt.show()

        slice1 -= distance_map[0]
        slice2 -= distance_map2[0]
        slice1 = normalize(slice1)
        slice2 = normalize(slice2)
        slice1 = normalize(np.array(slice1 > thr1).astype(np.int8) * (1- normalize((distance_map * s ))))
        slice2 = normalize(np.array(slice2 > thr2).astype(np.int8) * (1 - normalize((distance_map2 * s )))) 
        slice1 = normalize(np.array(slice1 > 0.5).astype(np.int8))
        slice2 = normalize(np.array(slice2 > 0.5).astype(np.int8))

        slice1 = np.stack((slice1,)*3, axis=-1) * (209, 239, 44) 
        slice2 = np.stack((slice2,)*3, axis=-1) * (132, 236, 207) 

        new_pic1.append(slice1) 
        new_pic2.append(slice2) 
 

    new_pic1 = np.array(new_pic1)
    new_pic2 = np.array(new_pic2)
    # pic1 = np.array(pic1 > thr1).astype(np.int8)
    # pic2 = np.array(pic2 > thr2).astype(np.int8)*2
    
    # print(np.unique)
    pic = new_pic1 + new_pic2
    # pic[pic==3] = 1

    print(np.unique(pic))
    for n in range(32):
        print(pic[n][0].shape)
        print(np.unique(pic[n][0]))
        plt.figure()
        plt.imshow(normalize(pic[n][0]))
        plt.show()
    
if __name__ == "__main__":
    main()


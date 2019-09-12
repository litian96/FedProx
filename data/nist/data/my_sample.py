from __future__ import division
import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange

from PIL import Image

NUM_USER = 200
CLASS_PER_USER = 3  # from 10 lowercase characters


def relabel_class(c):
    '''
    maps hexadecimal class value (string) to a decimal number
    returns:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    '''
    if c.isdigit() and int(c) < 40:
        return (int(c) - 30)
    elif int(c, 16) <= 90: # uppercase
        return (int(c, 16) - 55)
    else:
        return (int(c, 16) - 61) # lowercase

def load_image(file_name):
    '''read in a png
    Return: a flatted list representing the image
    '''
    size = (28, 28)
    img = Image.open(file_name)
    gray = img.convert('L')
    gray.thumbnail(size, Image.ANTIALIAS)
    arr = np.asarray(gray).copy()
    vec = arr.flatten()
    vec = vec / 255 # scale all pixel values to between 0 and 1
    vec = vec.tolist()

    return vec


def main():
    file_dir = "raw_data/by_class"

    train_path = "train/mytrain.json"
    test_path = "test/mytest.json"

    X = [[] for _ in range(NUM_USER)]  
    y = [[] for _ in range(NUM_USER)]

    nist_data = {}


    for class_ in os.listdir(file_dir):

        real_class = relabel_class(class_)
        if real_class >= 36 and real_class <= 45:
            full_img_path = file_dir + "/" + class_ + "/train_" + class_
            all_files_this_class = os.listdir(full_img_path)
            random.shuffle(all_files_this_class)
            sampled_files_this_class = all_files_this_class[:4000]
            imgs = []
            for img in sampled_files_this_class:
                imgs.append(load_image(full_img_path + "/" + img))
            class_ = relabel_class(class_)
            print(class_)
            nist_data[class_-36] = imgs  # a list of list, key is (0, 25)
            print(len(imgs))

    num_samples = np.random.lognormal(4, 1, (NUM_USER)) + 5

    idx = np.zeros(10, dtype=np.int64)

    for user in range(NUM_USER):
        num_sample_per_class = int(num_samples[user] / CLASS_PER_USER)
        if num_sample_per_class < 2:
            num_sample_per_class = 2

        for j in range(CLASS_PER_USER):
            class_id = (user + j) % 10
            if idx[class_id] + num_sample_per_class < len(nist_data[class_id]):
                idx[class_id] = 0
            X[user] += nist_data[class_id][idx[class_id]: (idx[class_id] + num_sample_per_class)]
            y[user] += (class_id * np.ones(num_sample_per_class)).tolist()
            idx[class_id] += num_sample_per_class
    
    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    for i in trange(NUM_USER, ncols=120):
        uname = 'f_{0:05d}'.format(i)
        
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()


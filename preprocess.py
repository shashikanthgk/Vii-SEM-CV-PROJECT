import os
import matplotlib.pyplot as plt
import numpy as np


def visualize(**images):

    n_images = len(images)
    plt.figure(figsize = (20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # Get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize = 20)
        plt.imshow(image)
    plt.show()

    
def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis = -1)

    return semantic_map

def reverse_one_hot(image):

    x = np.argmax(image, axis = -1)
    return x

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

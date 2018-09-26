from random import randint
import itertools
import numpy as np
import cv2


def random_mask(height=32, width=32, size=20, channels=1, type='rand', block_size=(6,6)):
    """Generates a random irregular mask with lines, circles and elipses"""    
    img = np.zeros((height, width, channels), np.uint8)

    # Set size scale
    if size<1:
        size = int((width * height) * size)
    else:
        size = size
    
    if width < 16 or height < 16:
        raise Exception("Width and Height of mask must be at least 16!")
    if len(block_size)!=2:
        raise Exception("block size unmatch!")
    if block_size[0]>height or block_size[1]>width:
        raise Exception("block size are too big")
    
    if type=='rand':
        elements = np.random.choice(height*width, size, replace=False)
        coordinate_map = map(lambda x: np.unravel_index(x, (height, width)), elements)
        def f(x, y):
            img[x][y]=1

        list(map(lambda x: f(x[0],x[1]), list(coordinate_map)))
    elif type=='block':
        block_height = block_size[0]
        block_width = block_size[1]
        height = height-block_height
        width = width-block_width

        # element = randint(1, height*width-1)
        # element_coordinate = np.unravel_index(element, (height, width))
        # element_coordinate_x = element_coordinate[0]
        # element_coordinate_y = element_coordinate[1]
        element_coordinate_x = randint(3, height-3)
        element_coordinate_y = randint(3, width-3)

        while block_height > 0:
            while block_width > 0:
                img[element_coordinate_x+block_height-1][element_coordinate_y+block_width-1] = 1
                block_width = block_width -1
            block_width = block_size[1]
            block_height = block_height - 1


    return 1-img


if __name__ == '__main__':
    img = random_mask(32, 32, 20, channels=3, type='block', block_size=(12, 5))[:, :, 0]
    print(img)
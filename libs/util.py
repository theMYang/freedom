from random import randint
import itertools
import numpy as np
import cv2

# smooth_time表示输入矩阵多少维度（层）为模型可以利用的数据。（最上一层为当前待预测数据模型不能利用，否则造成数据泄露。该参数为了方便实验扩展）
# size为待预测（修复）矩阵确实数据数。预测则该参数为整个矩阵采样点个数。该参数为了方便实验扩展
# block_size只在type参数为block时使用
def random_mask(height=32, width=32, size=1024, channels=1, smooth_time=0, type='rand', block_size=(32, 32)):
    """Generates a random irregular mask with lines, circles and elipses
       channels: time segment numbers of road net
       smooth_time: no block time segment numbers
    """ 
    # channels of block time segment   channels-smooth_time需要mask的channels层数
    img = np.zeros((height, width, channels-smooth_time), np.uint8)

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
        # 不放回的在height*width中选取size个元素作为mask的缺失点
        elements = np.random.choice(height*width, size, replace=False)
        coordinate_map = map(lambda x: np.unravel_index(x, (height, width)), elements)
        def f(x, y):
            img[x][y]=1

        list(map(lambda x: f(x[0],x[1]), list(coordinate_map)))
    elif type=='block':
        block_height = block_size[0]
        block_width = block_size[1]
        height = height
        width = width

        # element = randint(1, height*width-1)
        # element_coordinate = np.unravel_index(element, (height, width))
        # element_coordinate_x = element_coordinate[0]
        # element_coordinate_y = element_coordinate[1]
        # 选取blockMask的左上角为基点，防止数组越界
        element_coordinate_x = randint(0, height-block_height)
        element_coordinate_y = randint(0, width-block_width)

        while block_height > 0:
            while block_width > 0:
                img[element_coordinate_x+block_height-1][element_coordinate_y+block_width-1] = 1
                block_width = block_width -1
            block_width = block_size[1]
            block_height = block_height - 1
            
    # smooth_time为参考历史交通数据无缺失数据的时间点数
    img_tmp = np.zeros((height, width, smooth_time), np.uint8)
    img = np.concatenate((img, img_tmp), axis=2)

    return 1-img


if __name__ == '__main__':
    img = random_mask(32, 32, 20, channels=3, type='block', block_size=(12, 5))[:, :, 0]
    print(img)
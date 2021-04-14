import PIL
import PIL.Image
import pathlib
import random
import numpy as np
from matplotlib.pyplot import imshow



fpath = r'/data2/input'
data_dir = pathlib.Path(fpath + '/crops_128')
imgs = list(data_dir.glob('*.png'))
#print(f'There are {str(len(imgs))} cropped image samples available')

# show example sample image (cropped to 128x128)
#r = random.randint(0,len(imgs))
ascii_maps = []

for r in range(0,len(imgs)):
    print(f"filename: {imgs[r]}")
    i = Image(PIL.Image.open(imgs[r]))
    
    imshow(i)    
    
    a = input('is this map based on the original ASCII tileset? (y/n)')
    
    if a == 'y':
        map_num = imgs[r].name.split('_')[1]
        ascii_maps.append(map_num)
    else:
        pass
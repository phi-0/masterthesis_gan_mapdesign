"""

This script creates separate tokens based on the
default dwarf fortress tileset (640x300 or 800x600 version)


input:  default dwarf fortress tileset (curses) as .png in either for 640 x 300 or 800 x 600 pixel resolution
output: 256 12 x 10 glyphs making up the fundamental tokens of the game

"""

from random import randrange
from PIL import Image
from os import path

################ SETTINGS ################


filepath = r"G:/Dev/DataScience/masterthesis_gan_mapdesign/data/standard_tileset_800x600.png"   #640x300: "G:/Dev/DataScience/masterthesis_gan_mapdesign/data/standard_tileset.png"
outpath = r"G:/Dev/DataScience/masterthesis_gan_mapdesign/data/tiles/800x600"                   #640x300: "G:/Dev/DataScience/masterthesis_gan_mapdesign/data/tiles/640x300" 


##########################################


# print(f'Working on image: {file}')
img = Image.open(filepath)
x, y = img.size


# cropping settings
tiles_per_dim = 16
n = tiles_per_dim ** 2          # number of samples to be taken per input image --> tilemap contains 16 x 16 tiles = 256
x_dim = x / tiles_per_dim       # target dimension of a single token x-dimension
y_dim = y / tiles_per_dim       # target dimension of a single token y-dimension


i = 0
for x in range(tiles_per_dim):      # loop through x dimension
    for y in range(tiles_per_dim):  # loop through y dimension
        x1 = x * x_dim
        y1 = y * y_dim

        #print(f'Running crop x: {x1}-{x1 + x_dim}, y: {y1}-{y1 + y_dim}')

        crop = img.crop((x1, y1, x1 + x_dim, y1 + y_dim))
        
        print(f'Saving: {outpath}/tile_{i}.pn')
        crop.save(path.join(outpath, f'tile_{i}.png'), 'PNG')

        i+=1
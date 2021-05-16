from random import randrange
from PIL import Image
from os import walk, path

################ SETTINGS ################
folderpath = r"/data/maps"
outpath = r"/data2/input/ascii_crops_tiles/maps"
map_numbers = r"/home/ubuntu/masterthesis_gan_mapdesign/scripts/image_preprocessing/map_list.txt"

# cropping settings
dim = 12      #target dimension (symmetrical dim x dim matrix)
n = 30         #number of samples to be taken per input image


##########################################


#load list of ASCII based map numbers
with open(map_numbers, "r") as fobj:
    maps = fobj.readlines()

#replace newline characters in each list element
maps = [m.replace('\n','') for m in maps]
print(maps)


#get file list generator with os.walk()
for _, _, files in walk(folderpath):
    f_count = 1

    for file in files:

        #print(f"{maps[1]}-{file.replace('map_','').replace('.png','')}")
        #only continue with PNG files and map numbers contained in map_list.txt ( filtering out .ZIP files)
        file_name = file

        if file[-3:len(file)] == 'png' and file_name.replace('map_','').replace('.png','') in maps:
            print(f'File {str(f_count)} of {str(len(maps))} - filename: {file}', flush=True)
            f_count += 1
            #print(f'Working on image: {file}')
            #loop over images in directory
            img = Image.open(path.join(folderpath, file))
            x, y = img.size

            tiles_per_dim = x // dim
            n = tiles_per_dim ** 2  # number of samples to be taken per input image --> tilemap contains 16 x 16 tiles = 256
            x_dim = x / tiles_per_dim  # target dimension of a single token x-dimension
            y_dim = y / tiles_per_dim  # target dimension of a single token y-dimension
            #name = img.filename.replace('.png','') #get filename and remove filetype extension

            #loop over samples per input image
            samples = []
            i = 0
            for i in range(n):
                for x in range(tiles_per_dim):  # loop through x dimension
                    for y in range(tiles_per_dim):  # loop through y dimension
                        pos = randrange(0,x_dim,1)
                        x1 = (x * pos) * x_dim
                        y1 = (y * pos) * (y_dim - 2) #only move 12-2 = 10 pixels in y-direction since tiles are actually 12x10. We move by 10 pixels but still take a 12x12 crop (which the model will then crop to 12x10)

                        # print(f'Running crop x: {x1}-{x1 + x_dim}, y: {y1}-{y1 + y_dim}')

                        samples.append(img.crop((x1, y1, x1 + x_dim, y1 + y_dim)))

                        i += 1

            # save output images when NOT mostly black
            for i, sample in enumerate(samples):
                #get color range
                colors = sample.getcolors() # this method returns None if the number of colors exceeds the default value of 256.

                #with the following condition we filter out mostly black / unicolor images which don't hold any information
                if colors == None or len(colors) > 2:
                    sample.save(f"{outpath}/{file.replace('.png','')}_crop{str(i)}.png", 'PNG')
                else:
                    pass


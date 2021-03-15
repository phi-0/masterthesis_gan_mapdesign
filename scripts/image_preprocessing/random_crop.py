# refer to https://stackoverflow.com/questions/57221754/how-to-autocrop-randomly-using-pil for the basis of this logic

from random import randrange
from PIL import Image
from os import walk, path

################ SETTINGS ################
folderpath = r"G:\Dev\DataScience\masterthesis_gan_mapdesign\data\Dwarf Fortress Maps\PNG Exports"
outpath = r"G:\Dev\DataScience\masterthesis_gan_mapdesign\data\Dwarf Fortress Maps\random_crops"

# cropping settings
dim = 1024      #target dimension (symmetrical dim x dim matrix)
n = 15          #number of samples to be taken per input image
##########################################


#get file list generator with os.walk()
for _, _, files in walk(folderpath):
    i = 1

    for file in files:
        print(f'File {str(i)} of {str(len(files))}')
        #only continue with PNG files ( filtering out .ZIP files)
        if file[-3:len(file)] == 'png':

            #print(f'Working on image: {file}')
            #loop over images in directory
            img = Image.open(path.join(folderpath, file))
            x, y = img.size
            name = img.filename.replace('.png','') #get filename and remove filetype extension

            #loop over samples per input image
            samples = []  # output list of cropped samples
            for s in range(n):
                x1 = randrange(0, x - dim)
                y1 = randrange(0, y - dim)
                samples.append(img.crop((x1, y1, x1 + dim, y1 + dim)))

            # save output images when NOT mostly black
            for j, sample in enumerate(samples):
                #get color range
                colors = sample.getcolors() # this method returns None if the number of colors exceeds the default value of 256.

                #with the following condition we filter out mostly black / unicolor images which don't hold any information
                if colors == None or len(colors) > 2:
                    sample.save(f"{outpath}/{file.replace('.png','')}_crop{str(j)}.png", 'PNG')
                else:
                    pass

        i += 1


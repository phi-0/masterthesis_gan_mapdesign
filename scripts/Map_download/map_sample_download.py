import pathlib
import urllib
import os

if not pathlib.Path(f'{wd}/tiles').exists():
        print(f'Downloading tiles to {wd}/tiles...')
        os.mkdir(f'{wd}/tiles')
        for ti in range(0,256):
            url = f'https://github.com/phi-0/masterthesis_gan_mapdesign/raw/master/data/tiles/800x600/tile_{ti}.png'
            urllib.request.urlretrieve(url, f'{wd}/tiles/tile_{ti}.png')
        print(f'Finished downloading tileset with last file: {url}')
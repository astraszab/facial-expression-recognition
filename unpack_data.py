import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from PIL import Image


def main():
    zf = zipfile.ZipFile('fer2013.zip')
    df = pd.read_csv(zf.open('fer2013.csv'))

    img_size = int(np.sqrt(len(df.pixels[0].split())))

    for i in df.emotion.unique():
        Path('data/train/{}'.format(i)).mkdir(parents=True, exist_ok=True)
        Path('data/val/{}'.format(i)).mkdir(parents=True, exist_ok=True)
        Path('data/test/{}'.format(i)).mkdir(parents=True, exist_ok=True)

    for i in df.index:
        pixels = np.array([int(val) for val in df.pixels[i].split()],
                          dtype=np.uint8).reshape(img_size, img_size)
        img = Image.fromarray(pixels)
        if df.Usage[i] == 'Training':
            usage = 'train'
        elif df.Usage[i] == 'PublicTest':
            usage = 'val'
        else:
            usage = 'test'
        category = str(df.emotion[i])
        img_name = '{}.png'.format(i)
        img.save(Path.joinpath(Path(), 'data', usage, category, img_name))


if __name__ == '__main__':
    main()

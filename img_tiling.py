from colorthief import ColorThief
import os
from PIL import Image

import pandas as pd
path = 'static/uploads/1_bathroom.jpg'
color_thief = ColorThief(path)

def tile(files_list, name):
    print(files_list[0])
    img_01 = Image.open(files_list[0])

    img_02 = Image.open(files_list[1])
    img_03 = Image.open(files_list[2])
    img_04 = Image.open(files_list[3])
    
    img_01_size = img_01.size
    img_02_size = img_02.size
    img_03_size = img_02.size
    img_02_size = img_02.size
    
    print('img 1 size: ', img_01_size)
    print('img 2 size: ', img_02_size)
    print('img 3 size: ', img_03_size)
    print('img 4 size: ', img_03_size)
    
    new_im = Image.new('RGB', (2*img_01_size[0],2*img_01_size[1]), (250,250,250))
    
    new_im.paste(img_01, (0,0))
    new_im.paste(img_02, (img_01_size[0],0))
    new_im.paste(img_03, (0,img_01_size[1]))
    new_im.paste(img_04, (img_01_size[0],img_01_size[1]))
    
    new_im.save(name, "PNG")

df = pd.read_csv('houses.csv')
path = 'img'

for i in range(1, len(df)+1):
    temp_list = []
    temp_list.append(f'{path}/{i}_bathroom.jpg')
    temp_list.append(f'{path}/{i}_bedroom.jpg')
    temp_list.append(f'{path}/{i}_frontal.jpg')
    temp_list.append(f'{path}/{i}_kitchen.jpg')
    tile(temp_list, f'{path}/{i}.png')


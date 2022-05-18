from sklearn.linear_model import lasso_path
import albumentations as A
from PIL import Image
import numpy as np
import pandas as pd
import random

# Create a pipline with 4 different transformations. 
transform = A.Compose(
    [
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=.5, contrast_limit=.3),
        A.Rotate(),
    ]
)

df = pd.read_csv('house_classes.csv')
intervals = list(df['target'].unqiue())
for interval in intervals:
    # get directory name interval
    dir_name = 'img/' + str(interval)
    dir = df.path()
    df.price = random(interval.left, interval.right)
    





path = 'static/building/534_frontal.jpg'
pillow_image = Image.open(path)
image = np.array(pillow_image)

# Apply transformation
transformed = transform(image=image)

# Access and show transformation
transformed_image = transformed["image"]
img = Image.fromarray(transformed_image)



img.show()
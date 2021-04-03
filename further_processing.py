import glob
from PIL import Image
import os

files = glob.glob("./breast-histopathology-imagesNEW/*/*/*.png")
for file in files:
    im = Image.open(file)
    if(im.size != (50,50)):
        print("Removed: {}".format(file))
        os.remove(file)


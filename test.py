import os
import tensorflow
import cv2
from PIL import Image
import numpy as np 
import skimage
from skimage import io
from PIL import ImageDraw
from PIL import ImageFont


imagePath = './data/JPEGImages/000001.jpg'

img_PIL = Image.open(imagePath)
print(type(img_PIL))
img_PIL = np.array(img_PIL)
print(type(img_PIL))
print("=======img_PIL shape:  ",img_PIL.shape)

img_cv2 = cv2.imread(imagePath)  #cv2.imread 的图片是BGR
print(type(img_cv2))
print("=======img_cv2 shape:  ",img_cv2.shape)

img_skimage = io.imread(imagePath)
print(type(img_skimage))#np.array
print("=======img_skimage shape:  ",img_skimage.shape)

print(np.array_equal(img_PIL, img_cv2))
print(np.array_equal(img_PIL ,img_skimage))
print(np.array_equal(img_cv2 , img_skimage))

#img_PIL = cv2.flip(img_PIL,-1)
img = Image.fromarray(np.uint8(img_PIL))
draw = ImageDraw.Draw(img)
draw.rectangle((100,100,300,300),outline = "red")

text = 'person'
move = 10
#font = ImageFont.truetype(size = 40, encoding="unic")#设置字体
draw.text((100, 100-move), text)


img.show()
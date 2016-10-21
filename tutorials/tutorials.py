# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:02:17 2016

@author: Toshana
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

## OpenCV Tutorial

# load a colour image in greyscale
img = cv2.imread("C:/Users/Toshana/Documents/Udacity/computer_vision/toy_data/img1.jpg")
print img

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save as png
cv2.imwrite("grey_pine.png", img)

# displace image with matplotlib
plt.imshow(img, cmap = "gray")
plt.show()

# accessing pixel value
pine = cv2.imread("C:/Users/Toshana/Documents/Udacity/computer_vision/toy_data/img1.jpg")
px = pine[100, 100]
blue = pine[100, 100, 0]

# accessing channels
b, g, r = cv2.split(pine)

#swap red and blue pixels
pine2 = cv2.merge((r, g, b))
cv2.imwrite("swapped_pixels.png", pine2)
cv2.imwrite("monochrome_green.png", g)
cv2.imwrite("monochrome_red.png", r)

# middle square
h = 512
w = 400
a = (h - 100)/2
b = (w - 100)/2
square = pine[a:(h-a), b:w-b]
r[a:(h-a), b:w-b] = g[a:(h-a), b:w-b]
cv2.imwrite("inserted_square.png", r)

# Subtract the mean from all pixels, 
# then divide by standard deviation, 
# then multiply by 10 (if your image is 0 to 255) or by 0.05 (if your image ranges from 0.0 to 1.0). 
# Now add the mean back in.

avg = np.mean(g)
std = np.std(g)
answer = (((g- avg)/std) * 10) + avg
cv2.imwrite("answer.png", answer)

# shift green to the left by 2 pixels
































import cv2
import random

img = cv2.imread('test.jpg')

[height,width,dim] = img.shape

scale = 0.22
box_height = int(height*scale)
box_wigth = int(width*scale)

x = random.randint(0,width - box_wigth)
y = random.randint(0,height - box_height)

for i in range(x, x+box_wigth):
    for j in range(y, y+box_height):
        img[j,i] = (0,0,0)

cv2.imwrite('output.jpg',img)

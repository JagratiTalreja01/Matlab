from PIL import Image 
import cv2
import numpy as np
from matplotlib import pyplot as plt
#(x,y,width,height)
im = Image.open("Method3/Original_Image/baby_GT.png")  
im1 = im.crop((200,100,400,300)) 
im1.save("Cropped_Image.png")
img1 = cv2.imread("Cropped_Image.png")
img_cvt=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
cv2.imshow("Crop_img",img_cvt)

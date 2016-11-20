#!/usr/bin/python

import cv
import Image
import numpy as np
from numpy.random import randn

#cv.NamedWindow("img",1)

img=cv.LoadImage("/home/deeplearn/Desktop/qin/7.jpg",1)#32x32
arr2=randn(32,32,3)
grayArr=randn(1024,1)#32x32=1024
src=randn(32,32,3)
#arr1=np.array(pix)
#for j in range(arr1.size):
#    print arr1[j]

for j in range(32):
    for i in range(32):
        pix=cv.Get2D(img,j,i)
        arr1=pix
        arr2[i,j,0]=arr1[0]
        arr2[i,j,1]=arr1[1]
        arr2[i,j,2]=arr1[2]
        #print pix

src=cv.CreateImage(cv.GetSize(img),8,3)
grayImg=cv.CreateImage(cv.GetSize(img),8,1)
grayImg1=cv.CreateImage(cv.GetSize(img),8,1)
j=0
i=0
for j in range(32):
    for i in range(32):
        pix1=arr2[i,j]
        cv.Set2D(src,j,i,pix1)

cv.ConvertImage(img,grayImg,cv.CV_BGR2GRAY)
i=0
j=0
for j in range(32):
    for i in range(32):
        pix2=cv.Get2D(grayImg,j,i)
        temp=pix2[0]
        grayArr[i+j*32]=temp
        #print temp
i=0
j=0
for j in range(32):
    for i in range(32):
        temp=grayArr[i+j*32]
        cv.Set2D(grayImg1,j,i,cv.Scalar(temp,0,0))
        
cv.ShowImage("img",grayImg1)
if cv.WaitKey(0)==27:
   cv.DestroyWindow("img")
      



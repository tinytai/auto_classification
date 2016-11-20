#!/usr/bin/python

import cv
import Image
import numpy as np
from numpy.random import randn

#cv.NamedWindow("img",1)

arrTemp=randn(4)
arrPair=randn(46,164,164,1)
arrHorse=randn(46,164,164,1)
arrCat=randn(46,164,164,1)
arrPair=np.asarray(arrPair,dtype=np.float32)
arrHorse=np.asarray(arrHorse,dtype=np.float32)
arrCat=np.asarray(arrCat,dtype=np.float32)
for i in range(46):
    #164x164
    print i
    imgPair=cv.LoadImage("/home/deeplearn/Desktop/qin/pairs_46/"+str(i)+".jpg",0)
    imgHorse=cv.LoadImage("/home/deeplearn/Desktop/qin/horse_46/"+str(i)+".jpg",0)
    imgCat=cv.LoadImage("/home/deeplearn/Desktop/qin/cat_46/"+str(i)+".jpg",0)
    for h in range(164):
        for w in range(164):
            pixPair=cv.Get2D(imgPair,h,w)
            arrTemp=np.array(pixPair)
            arrPair[i,w,h,0]=arrTemp[0]

            pixHorse=cv.Get2D(imgHorse,h,w)
            arrTemp=np.array(pixHorse)
            arrHorse[i,w,h,0]=arrTemp[0]

            pixCat=cv.Get2D(imgCat,h,w)
            arrTemp=np.array(pixCat)
            arrCat[i,w,h,0]=arrTemp[0]

    cv.Zero(imgPair)
    cv.Zero(imgHorse)
    cv.Zero(imgCat)
print "read finished!"            

grayImg=cv.CreateImage(cv.GetSize(imgPair),8,1)
i=0
h=0
w=0
for i in range(46):
    for h in range(164):
        for w in range(164):
            pix=arrPair[i,w,h,0]
            cv.Set2D(grayImg,h,w,cv.Scalar(pix,0,0))
            
    cv.ShowImage("PairImg",grayImg)
    cv.WaitKey(100)

for i in range(46):
    for h in range(164):
        for w in range(164):
            pix=arrHorse[i,w,h,0]
            cv.Set2D(grayImg,h,w,cv.Scalar(pix,0,0))
            
    cv.ShowImage("horseImg",grayImg)
    cv.WaitKey(100)

for i in range(46):
    for h in range(164):
        for w in range(164):
            pix=arrCat[i,w,h,0]
            cv.Set2D(grayImg,h,w,cv.Scalar(pix,0,0))
            
    cv.ShowImage("catImg",grayImg)
    cv.WaitKey(100)    

#for j in range(32):
#    for i in range(32):
#        temp=grayArr[i+j*32]
#        cv.Set2D(grayImg1,j,i,cv.Scalar(temp,0,0))
        
#cv.ShowImage("img",grayImg1)
#if cv.WaitKey(0)==27:
#   cv.DestroyWindow("img")
      



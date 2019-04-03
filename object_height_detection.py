'''
Created on 02-Apr-2019

@author: dhanalaxmi
'''
import cv2
import os
import matplotlib.pyplot as plt
import imutils
import math
import numpy as np
from scipy import ndimage
from operator import itemgetter
import dlib



def height_detection(folder,input,standard):
    standard = cv2.imread(os.path.join(folder,standard))
    resize_standard=cv2.resize(standard,(3500,4000))
    input=cv2.imread(os.path.join(folder,input))
    resize_input=cv2.resize(input,(3500,4000))
    
    gray = cv2.cvtColor(resize_standard, cv2.COLOR_BGRA2GRAY)
    thresh = cv2.threshold(gray, 170,250, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
  
    for c in cnts:
        area=cv2.contourArea(c)
        if area > 1000000  and area<1500000:
            epsilon = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1* epsilon, True)
            if len(approx)== 4 :
                (x,y,w,h)=cv2.boundingRect(approx)
                cv2.circle(resize_standard,(x,y), 3, (255,0,0), 30)
                cv2.circle(resize_standard,(x+w,y), 3, (0,255,0), 30)
                cv2.circle(resize_standard,(x+w,y+h), 3, (0,0,255), 30)
                cv2.circle(resize_standard,(x,y+h), 3, (0,0,0), 30)
             
    STL=(x,y)
    STR=(x+w,y)
    SBR=(x+w,y+h)
    SBL=(x,y+h)
 
    OBJECT_CORNER=(STL,STR,SBR,SBL)
    OBJECT_CORNER=(np.asarray(OBJECT_CORNER, dtype="float32"))
  
    #plt.imshow(cv2.cvtColor(resize_standard,cv2.COLOR_BGR2RGB))
    #plt.show()
     
    im=resize_input.copy()
    img=resize_input.copy()
    gray = cv2.cvtColor(resize_input, cv2.COLOR_BGRA2GRAY)
    thresh = cv2.threshold(gray, 110,220, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
 
    for c in cnts:
        area=cv2.contourArea(c)
        if area > 1000000  and area<1500000:
            epsilon = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1* epsilon, True)
            if len(approx)== 4 :
                (x,y,w,h)=cv2.boundingRect(approx)
                cv2.circle(resize_input,(x,y), 3, (255,0,0), 30)
                cv2.circle(resize_input,(x+w,y), 3, (0,255,0), 30)
                cv2.circle(resize_input,(x+w,y+h), 3, (0,0,255), 30)
                cv2.circle(resize_input,(x,y+h), 3, (0,0,0), 30)
                tl = tuple(c[c[:, :, 0].argmin()][0])
                br = tuple(c[c[:, :, 0].argmax()][0])
                tr = tuple(c[c[:, :, 1].argmin()][0])
                bl = tuple(c[c[:, :, 1].argmax()][0])
                cv2.drawContours(im, [c], -1, (0, 255, 255), 1)
                cv2.circle(im, tl, 8, (255,0,0), 25)
                cv2.circle(im, br, 8, (0, 255, 0), 25)
                cv2.circle(im, tr, 8, (0, 0, 255), 25)
                cv2.circle(im, bl, 8, (0, 0, 0), 25)
 
    TL=(x,y)
    TR=(x+w,y)
    BR=(x+w,y+h)
    BL=(x,y+h)
      
    obj_corner=(tl,tr,br,bl)
    OBJ_CORNER=(TL,TR,BR,BL)
  
    #plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    #plt.show()
     
    image=[]
    Distance=[]
      
    for i in OBJ_CORNER:
        for j in obj_corner:
            image.append((i,j))
            distance=math.sqrt(((i[0]-j[0])**2)+((i[1]-j[1])**2))
            Distance.append((distance))
      
      
    Rect=[]
    list = [Distance[i:i+4] for i in range(0, len(Distance), 4)]
    for j in list:
        for i in Distance:
            for x in image:
                ind=(min(j))
        key=(Distance.index(ind))
        Rect.append(image[key])
      
    tl=Rect[0][1]
    tr=Rect[1][1]
    br=Rect[2][1]
    bl=Rect[3][1]
      
    object_corner=(tl,tr,br,bl)
    object_corner=(np.asarray(object_corner, dtype="float32"))
    cv2.circle(img, tl, 8, (255,0,0), 25)
    cv2.circle(img, br, 8, (0, 255, 0), 25)
    cv2.circle(img, tr, 8, (0, 0, 255), 25)
    cv2.circle(img, bl, 8, (255, 255, 0), 25)
    #plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #plt.show()
      
    h, mask = cv2.findHomography(object_corner,OBJECT_CORNER)
              
    perspective= cv2.warpPerspective(img, h, (3500,4000))
        
    #plt.imshow(cv2.cvtColor(perspective,cv2.COLOR_BGR2RGB))
    #plt.show()
    
    gray = cv2.cvtColor(perspective, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,30,30,apertureSize =3)
#     plt.imshow(edges)
#     plt.show()
    minLineLength=1500
    maxLineGap=500 
     
    lines = cv2.HoughLinesP(image=edges,rho=.7,theta=np.pi/180, threshold=200,lines=np.array([]), minLineLength=minLineLength,maxLineGap=maxLineGap)
    bl_listing=[]
    for i in range(len(lines)):
        if lines[i][0][0]<500:
            if lines[i][0][1]>2800:
                if lines[i][0][2]>2500:
                    if lines[i][0][3]>2800:
                        bl_listing.append(lines[i])
    
    a,b,c = np.shape(bl_listing)
    #print(a,b,c)
    angles = []
    last=(len(bl_listing)-1)
    for x1, y1, x2, y2 in bl_listing[last]:
        cv2.line(perspective, (x1, y1), (x2, y2), (255, 0, 0), 25, cv2.LINE_AA)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    #print(angles[0])
    median_angle=(angles[0])
    # median_angle = np.median(angles)
    img_rotated = ndimage.rotate(perspective, median_angle)
    #plt.imshow(cv2.cvtColor(img_rotated,cv2.COLOR_BGR2RGB))
    #plt.show() 
     
    gray = cv2.cvtColor(img_rotated, cv2.COLOR_BGRA2GRAY)
    thresh = cv2.threshold(gray, 110,220, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    #plt.imshow(thresh)
    #plt.show()
    cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
       
    for c in cnts:
        area=cv2.contourArea(c)
        if area > 1000000 and area < 1400000:
            epsilon = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1* epsilon, True)
            if len(approx)== 4 :
                (x,y,w,h)=cv2.boundingRect(approx)
                cv2.circle(img_rotated,(x,y), 3, (255,0,0), 30)
                cv2.circle(img_rotated,(x,y+h), 3, (0,0,0), 30)
    topleft=(x,y)
    topright=(x+w,y)
    bottomright=(x+w,y+h)
    bottomleft=(x,y+h)
        
    OBJECT_CORNER=(topleft,topright,bottomright,bottomleft)
    OBJECT_CORNER=(np.asarray(OBJECT_CORNER, dtype="float32"))
    RO_Pixel=math.sqrt((topleft[1]-bottomleft[1])**2)  
    print("RO pixel is {}".format(RO_Pixel))
        
    gray = cv2.cvtColor(img_rotated, cv2.COLOR_BGRA2GRAY)
    thresh = cv2.threshold(gray, 100,230, cv2.THRESH_BINARY)[1]
        
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(thresh, None)
         
    # Convert the Key Points into an array of coordinates
    points = []
    for point in kp:
                 
        points.append(point.pt)
    # Determine the Top Most Points of Bottle
    points_ordered = sorted(points, key=itemgetter(1), reverse=False)
    head = [points_ordered[0]]
             
    cv2.circle(img_rotated, (int(head[0][0]), int(head[0][1])), 2, (0, 0, 255), 40)
    edges = cv2.Canny(gray,50,50,apertureSize = 3)
    plt.imshow(edges) 
    plt.show()
    minLineLength=2000
    lines = cv2.HoughLinesP(image=edges,rho=.7,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=300)
    bl_listing=[]
    for i in range(len(lines)):
        if lines[i][0][0] <500:
            if lines[i][0][1] <3200 and lines[i][0][1] >2700:
                if lines[i][0][2] >2200:
                    if lines[i][0][3] <3200 and lines[i][0][3] >2700:
                        bl_listing.append(lines[i][0])
               
    points_ordered = sorted(bl_listing, key=itemgetter(1), reverse=True)
    bottom=points_ordered[-1]
    a,b,c,d=bottom
    midpoint=(c,d)
               
    cv2.line(img_rotated,(a,b),(c,d), (0, 0, 255), 50, cv2.LINE_AA)
    Bottle_Height=(midpoint[1]-head[0][1]) 
    print("Bottle pixel is {}".format(Bottle_Height))  
         
    RO_CM =14.8
    # Calculate the conversion factor
    CM_per_Pixel = (RO_CM/ RO_Pixel)
               
    # Calculate the Kid's Height in cm
    Bottle_Height_CM = (Bottle_Height * CM_per_Pixel)
    print("Bottle Height in cm -  {}".format(Bottle_Height_CM))  
               
    plt.imshow(cv2.cvtColor(img_rotated,cv2.COLOR_BGR2RGB))
    plt.show() 



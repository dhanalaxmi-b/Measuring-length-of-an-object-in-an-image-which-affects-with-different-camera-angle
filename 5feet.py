'''
Created on 18-Mar-2019

@author: dhanalaxmi
'''
import cv2
import imutils
from matplotlib import pyplot as plt
import numpy as np
from imutils import perspective
import math
from scipy import ndimage
from operator import itemgetter



im = cv2.imread("/home/dhanalaxmi/eclipse-workspace/AI/Object_Height/image/5feet/bottomleft.jpg")
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray,60,60,apertureSize =3)


minLineLength=1500
maxLineGap=100    
lines = cv2.HoughLinesP(image=edges,rho=.7,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=maxLineGap)
bl_listing=[]
for i in range(len(lines)):
    if lines[i][0][0]<300:
        if lines[i][0][1]>1000:
            if lines[i][0][2]>1000:
                bl_listing.append(lines[i])
a,b,c = np.shape(bl_listing)
angles = []
last=(len(bl_listing)-1)
for x1, y1, x2, y2 in bl_listing[last]:
    cv2.line(im, (x1, y1), (x2, y2), (255, 0, 0), 5)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)

median_angle=(angles[0])
# median_angle = np.median(angles)
img_rotated = ndimage.rotate(im, median_angle)
              
# plt.imshow(cv2.cvtColor(img_rotated,cv2.COLOR_BGR2RGB))
# plt.show() 
im=img_rotated
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

thresh = cv2.threshold(gray, 110,220, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
 
for c in cnts:
    area=cv2.contourArea(c)
    if area > 500000 and area< 900000:
        epsilon = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1* epsilon, True)
        if len(approx)== 4 :
            box = cv2.minAreaRect(approx)
            box_angle=box[-1]
            box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            for (x, y) in box:
                cv2.circle(im, (int(x), int(y)), 5, (0, 0, 255), 30)
            cv2.drawContours(im, [box.astype("int")], -1, (0, 255, 0), 20)

 
RO_Pixel=math.sqrt((box[3][1]-box[0][1])**2)  
print("RO pixel is {}".format(RO_Pixel))

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
    
cv2.circle(im, (int(head[0][0]), int(head[0][1])), 2, (0, 0, 255), 40)

thresh = cv2.threshold(gray, 70,80, cv2.THRESH_BINARY)[1]

minLineLength=2200
maxLineGap=500  
lines = cv2.HoughLinesP(image=thresh,rho=.7,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=maxLineGap)
bl_listing=[]
for i in range(len(lines)):
    if lines[i][0][0]<500:
        if lines[i][0][1]>1300 and lines[i][0][1]<1950:
            if lines[i][0][2]>2500:
                if lines[i][0][3]>1300 and lines[i][0][3]<1950:
                    bl_listing.append(lines[i][0])
                    
points_ordered = sorted(bl_listing, key=itemgetter(1), reverse=True)
bottom=points_ordered[-1]
a,b,c,d=bottom
midpoint=((a + c) * 0.5), ((b + d) * 0.5)
      
cv2.line(im,(a,b),(c,d), (0, 0, 255), 30, cv2.LINE_AA)

Bottle_Height=(midpoint[1]-head[0][1]) 
print("Bottle pixel is {}".format(Bottle_Height))  
      
RO_CM =14.8
# Calculate the conversion factor
CM_per_Pixel = (RO_CM/ RO_Pixel)
      
# Calculate the Kid's Height in cm
Bottle_Height_CM = (Bottle_Height * CM_per_Pixel)
print("Bottle Height in cm -  {}".format(Bottle_Height_CM))  
      
plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
plt.show()
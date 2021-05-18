import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('Resources/cacaOne.xml')
faceName= "person"
Meenio = "Chair"
water = "mouse"
color = (0,255,0)
colorTwo = (0,0,255)
colorThree = (255,0,0)

chairCascade = cv2.CascadeClassifier('Resources/chair.xml')
bottleCascade = cv2.CascadeClassifier('Resources/mouse.xml')

cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#convert to gray
    faces = faceCascade.detectMultiScale(gray,1.3,5) #depending on size of image
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0),2)
        cv2.putText(img,faceName,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
        roi_color = img[y:y+h,x:x+w]
    glasses  = chairCascade.detectMultiScale(gray,1.3,5)
    for (cx,cy,cw,ch) in glasses:
        cv2.rectangle(img,(cx,cy), (cx+cw,cy+ch), (0,0,255),2)
        cv2.putText(img,Meenio ,(cx,cy-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,colorTwo,2)
        roi_color = img[cy:cy+ch,cx:cx+cw]

    waterBottle = bottleCascade.detectMultiScale(gray,1.3,5)
    for (dx,dy,dw,dh) in waterBottle:
        cv2.rectangle(img,(dx,dy), (dx+dw,dy+dh),(255,0,0),2)
        cv2.putText(img,water ,(dx,dy-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,colorThree,2)
        roi_color = img[dy:dy+dh,dx:dx+dw]


    cv2.imshow("result", img)

    if cv2.waitKey(5) & 0xFF ==ord('q'):#TO BREAK
        break

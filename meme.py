from ast import Store
import cv2
import numpy as np

def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver
def getcontour(contour):
    boxes =[]
    for i in contour:
        area = cv2.contourArea(i)
        if area>50:
            x =cv2.arcLength(i,True)
            approxi =cv2.approxPolyDP(i,x*0.02,True)
            if len(approxi)==4:
                boxes.append(i)
    boxes =sorted(boxes,key =cv2.contourArea,reverse=True)
    return boxes
def expectcontour(img):

    x =cv2.arcLength(img,True)
    approxi =cv2.approxPolyDP(img,x*0.02,True)
    return approxi
    
def reorder(mypoint):
    mypoint =mypoint.reshape((4,2))
    add =mypoint.sum(1)
    newpoint =np.zeros((4,1,2),np.uint32)
    newpoint[0]=mypoint[np.argmin(add)]
    newpoint[3]=mypoint[np.argmax(add)]
    diff =np.diff(mypoint)
    newpoint[2]=mypoint[np.argmax(diff)]
    newpoint[1]=mypoint[np.argmin(diff)]
    return newpoint
   
   
    
   
    # print(newpoint[0])
    # print(mypoint)
    # print(add)
   
def split(img):
    boxes =[]
    x =np.vsplit(img,5)
    for i in x:
        y =np.hsplit(i,5)
        for box in y:
            boxes.append(box)
    return boxes
    
    
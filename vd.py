from string import hexdigits
import meme
import cv2
import numpy as np
import pandas
#du an tìm mark recognize
webcamfedd =True
cap =cv2.VideoCapture(0)
cap.set(10,160)
width =750
height =750
question =5
choice =5
correctanswer =[1,2,1,2,4]
# img =cv2.imread(r"C:\Users\ADMIN\OneDrive\Documents\code\image\1.jpg")
while True:
	if webcamfedd:success,img = cap.read()
	else:
		img =cv2.imread(r"C:\Users\ADMIN\OneDrive\Documents\code\image\1.jpg")
	
	img =cv2.resize(img,(width,height))
	imgRGB =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	imgBLUR =cv2.blur(img,(5,5),1)
	imgcanny =cv2.Canny(imgBLUR,10,50)
	imgcontour = img.copy()
	imgblank =np.zeros_like(img)
	imgwrap =img.copy()
	try:
		imgwrap =img.copy()
		imgcontour = img.copy()

		contours,hierarchy =cv2.findContours(imgcanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		cv2.drawContours(imgcontour,contours,-1,(0,255,0),10)
		t =meme.getcontour(contours)

		biggist =meme.expectcontour(t[0])
		second =meme.expectcontour(t[1])


		cv2.drawContours(imgwrap,biggist,-1,(0,255,0),20)
		cv2.drawContours(imgwrap,second,-1,(255,0,0),20)
		biggist=meme.reorder(biggist)
		second=meme.reorder(second)


		pt1 =np.float32(biggist)
		pt2 =np.float32([[0,0],[width,0],[0,height],[width,height]])
		matrix =cv2.getPerspectiveTransform(pt1,pt2)
		imgbiggist =cv2.warpPerspective(imgwrap,matrix,(width,height))


		pts1 =np.float32(second)
		pts2 =np.float32([[0,0],[325,0],[0,150],[325,150]])
		matrixsecond =cv2.getPerspectiveTransform(pts1,pts2)
		imgsecond =cv2.warpPerspective(imgwrap,matrixsecond,(325,150))
		# cv2.imshow("sencond",imgsecond)
		imgthresold =cv2.cvtColor(imgbiggist,cv2.COLOR_BGR2GRAY)
		IMGTHRESOLD =cv2.threshold(imgthresold,170,300,cv2.THRESH_BINARY_INV)[1]
		boxes =meme.split(IMGTHRESOLD)
		myindex =np.zeros((question,choice))


		countR =0
		countC =0
		for image in boxes:
			

			mytotalpixal =cv2.countNonZero(image)
			myindex[countR][countC]=mytotalpixal
			countC+=1
			if countC==question:
				countR +=1
				countC =0
			
		myanswer =[]	
		for x in range(0,question):
			arrmax =myindex[x]
			maxnonzero =np.where(arrmax==np.amax(arrmax))
			myanswer.append(maxnonzero[0][0])

		grading =[]



		for i in range(0,5):
			
			if myanswer[i] ==correctanswer[i]:
				
				grading.append(1)
			else:
				grading.append(0)
				
			

		sum =sum(grading)
		score =(sum/question)*100
		print("score",score)
		imgsecond=cv2.putText(imgsecond,str(int(score))+"%",(70,100),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),3)

		def findanswer(img,myanswer,grading,correctanswer):
			# secW = int(img.shape[1]/5)
			# secH = int(img.shape[0]/5)

			
			for i in range(0,5):
				x =((myanswer[i]*150)+150//2)
				y =((i*150)+150//2)
				X =((correctanswer[i]*150)+150//2)
				Y =((i*150)+150//2)

				if grading[i]==1:
					cv2.circle(img,(x,y),50,(0,255,0),cv2.FILLED)
				else:
					cv2.circle(img,(x,y),50,(255,0,0),cv2.FILLED)
					cv2.circle(img,(X,Y),50,(0,255,0),cv2.FILLED)


					
			return img
		w =findanswer(imgbiggist,myanswer,grading,correctanswer)
		
		imgcopy =img.copy()

		invmetrix =cv2.getPerspectiveTransform(pt2,pt1)
		invimg =cv2.warpPerspective(w,invmetrix,(width,height))
		imgcopy=cv2.addWeighted(imgcopy,1,invimg,1,0)
		invmetrix2 =cv2.getPerspectiveTransform(pts2,pts1)
		invimg2 =cv2.warpPerspective(imgsecond,invmetrix2,(width,height))
		imgone =cv2.addWeighted(imgcopy,1,invimg2,1,0)
		concat =([img,imgRGB,imgBLUR,imgcanny],
				[imgwrap,imgbiggist,IMGTHRESOLD,imgone])
				
		cv2.imshow("imgone",imgone)
	except:

		concat =([img,imgRGB,imgBLUR,imgcanny],
				[imgblank,imgblank,imgblank,imgblank])
				
	imgstack =meme.stackImages(concat,0.3)

	cv2.imshow("img",imgstack)
	cv2.waitKey(300)
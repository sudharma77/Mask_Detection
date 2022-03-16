

from xlwt import Workbook
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import xlwt



def detect_and_predict_mask(frame, faceNet, maskNet):
	
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	
	faces = []
	locs = []
	preds = []
	

	
	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]

		
		if confidence > 0.5:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			
			
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			
			
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	
	if len(faces) > 0:
		
		
		
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	
	
	return (locs, preds)


prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


maskNet = load_model("mask_detector.model")


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()


lss=0
jj=0
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')
value = True
sum = 0 
a = 1 
b = 1
per = 0 
p = 0 


while True:
	
	
	frame = vs.read()
	frame = imutils.resize(frame, width=500)

	
	
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	
	
	for (box, pred) in zip(locs, preds):
		
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		
		
		label = "MASK" if mask > withoutMask else "NO MASK"
		color = (0, 255, 0) if label == "MASK" else (0, 0, 255)

		
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# sudharma = label
		# if lss==0:
		# 	print(sudharma)
		# 	lss=20
		# 	jj=jj+1
		# 	sheet1.write(jj, 0, sudharma)
		# 	wb.save('xlwt exam.xls')
		# else:
		# 	lss=lss-1
		if p == 20:
			p = 0
			if mask < withoutMask :
				a=a + 1
			
			if mask > withoutMask :
				b=b + 1
				
				

		sum = a + b 
		perPeopleWearMask = (b/sum)*100
		sy = (a/sum)*100

		print(perPeopleWearMask , "% Wearing")
		print(sy , "% Not Wearing")
		p = p + 1

		
		
	
		

	
		
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	
	cv2.imshow("FROYO_FRAME", frame)
	key = cv2.waitKey(1) & 0xFF

	
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
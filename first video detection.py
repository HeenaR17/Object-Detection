import cv2
from cv2.cv2 import destroyAllWindows

#Setting threshold to determine if object detection is valid
threshold = 0.50

#Capturing the live video and setting properties like frame width, height
capture = cv2.VideoCapture(0)
capture.set(3,1280)
capture.set(4,720)
capture.set(10,70)

classNames= []
classFile = 'coco.names'
#Opening and reading the file; the output will give us all the object names as a list
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

process = cv2.dnn_DetectionModel(weightsPath,configPath)
process.setInputSize(320,320)
process.setInputScale(1.0/ 127.5)
process.setInputMean((127.5, 127.5, 127.5))
process.setInputSwapRB(True)

while True:
    success,img = capture.read()
    classIds, confs, boundingbox = process.detect(img,confThreshold=threshold)
    print(classIds,boundingbox)

    if len(classIds) != 0:
        #Using zip function for parallel iteration
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),boundingbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Output",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        cv2.destroyAllWindows()
        break 
	
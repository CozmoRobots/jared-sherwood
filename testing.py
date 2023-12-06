import numpy
from ultralytics import YOLO
import cv2
import cvzone
import math
from src.sort import *

def Main():
    # constrain values for choices later
    #if needed you can make a mask with any eiting software canva for example
    #Region of counter
    #Assign a tracking id that stays consistent
    #Within consecutive frames find where item is moving to
    #use sort tracker for videos

    #Tracking
    tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)


#The below function contains the ability to create bounding boxes, track items, and count/
#items when they pass a threshold which I can use to create a choice based system, I.e put on
#detection on the left and one on the right.
def webCamBoxes():

    cap = cv2.VideoCapture(1)  # Front Webcam
    cap2 = cv2.VideoCapture(1)  # Rear Webcam
    cap.set(3, 640)
    cap.set(4, 480)
    model = YOLO('../YoloWeights/yolov8n.pt')
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    limits = [0, 297, 673, 297]
    totalCount=[]


    while True:
        success, img = cap.read()
        results = model(img, show=False)

        detections = np.empty((0,5))

        for r in results:
            boxes = r.boxes
            counter = 0
            for box in boxes:
                counter = counter + 1
                # Opencv version bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # CVZONE version bounding box
                w, h = x2 - x1, y2 - y1
                bbox = int(x1), int(y1), int(w), int(h)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=5)

                # Confidence Rectangle
                conf = math.ceil((box.conf[0] * 100)) / 100
                print(conf)
                cvzone.putTextRect(img, f"{conf}", (max(0, x1), max(35, y1)))
                # Class Name
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)),scale=0.8,thickness=1,offset=5)
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections = numpy.vstack((detections,currentArray))

        resultsTracker = tracker.update(detections)
        cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

        for result in resultsTracker:
            x1,y1,x2,y2,Id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=2,colorR=(255,0,0))
            cvzone.putTextRect(img, f"{int(Id)}", (max(0, x1), max(35, y1)), scale=0.8, thickness=1,
                               offset=5)
            cx,cy = x1+w//2,y1+h//2
            cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

            if limits[0] <cx< limits[2] and limits[1]-20<cy<limits[1]+2:
                if totalCount.count(Id) == 0:
                    totalCount.append(Id)

        cvzone.putTextRect(img, f"Count: {len(totalCount)}", (50, 50))

        cv2.imshow("video", img)
        cv2.waitKey(1)


def readImageYolo():
    img = cv2.imread("Images/ScoobyDoo_Front.jpg")
    imgSmall = cv2.resize(img, (1000, 1100))
    model = YOLO('../YoloWeights/yolov8l.pt')
    results = model(imgSmall, show=True)
    cv2.waitKey(0)

    pass


#Main()
webCamBoxes()


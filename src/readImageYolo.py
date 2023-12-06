import numpy
from ultralytics import YOLO
import cv2
import cvzone
import math
from src.sort import *


def readImageYolo():
    cap = cv2.VideoCapture(1)
    cap.set(3, 640)
    cap.set(4, 480)
    # all characters to be classified
    classNames = ["ScoobyDoo", "MickeyMouse", "Bert", "Ernie", "Rosita", "Perdita", "BBear", "BFox"]
    # load the trained model, disney or disneynano. disneynano runs at 100ms, disney runs at 1000ms
    model = YOLO('../YoloWeights/disneynano.pt')
    # Initialize the starting variables for the object tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    # Create a list of characters this run
    characters = []

    # Infinite loop to display
    while True:
        # capture video
        success, img = cap.read()
        # run model on single frame to detect object classes
        results = model(img, stream=True)
        # int array to track detections later
        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Values for boxes locations
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # CVZONE version bounding box
                w, h = x2 - x1, y2 - y1
                bbox = int(x1), int(y1), int(w), int(h)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=6)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if cls not in characters:
                    characters.append(cls)
                cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=2, thickness=2,
                                   offset=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = numpy.vstack((detections, currentArray))
        resultsTracker = tracker.update(detections)

        # The below shapes will form the detection area for left and right
        # left
        limitsL = [0,0,200,1080]
        limitsR = [1720,0,1920,1080]
        cv2.rectangle(img, (limitsL[0], limitsL[1]), (limitsL[2], limitsL[3]), (0, 0, 255), cv2.FILLED)
        # right
        cv2.rectangle(img, (limitsR[0], limitsR[1]), (limitsR[2], limitsR[3]), (0, 255, 0), cv2.FILLED)

        for result in resultsTracker:
            x1, y1, x2, y2, Id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            print(result)
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
            cvzone.putTextRect(img, f"{int(Id)}", (max(0, x1), max(35, y1)), scale=0.8, thickness=1,
                               offset=5)
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 2:
            #     if totalCount.count(Id) == 0:
            #         totalCount.append(Id)

        # cvzone.putTextRect(img, f"Count: {len(totalCount)}", (50, 50))

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 800, 600)
        cv2.imshow("image", img)
        cv2.waitKey(1)


readImageYolo()

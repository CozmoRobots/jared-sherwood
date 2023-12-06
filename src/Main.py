# NOTES
# For some reason it is not running normally before you submit "c" for character
# list creation. Fix it, it is running otherwise though!!

import numpy
from ultralytics import YOLO
import cv2
import cvzone
import math
from src.sort import *
import os
from openai import OpenAI
import cozmo
from cozmo.util import degrees

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), )
toRead = ''
assistant = ("You will narrate an interactive adventure story featuring user-specified characters and two options for "
             "user decisions - left or right. You should follow the format: User: sends characters You: reply with the "
             "first part of story and choice to be made User: replies with choice You: generate next part of story "
             "and choice")

def readImageYolo():
    # Create the video capture obj
    cap = cv2.VideoCapture(1)
    cap.set(3, 640)
    cap.set(4, 480)
    # all characters to be classified
    classNames = ["ScoobyDoo", "MickeyMouse", "Bert", "Ernie", "Rosita", "Perdita", "BrerBear", "BrerFox"]
    # load the trained model, disney or disneynano. disneynano runs at 100ms, disney runs at 1000ms
    model = YOLO('../YoloWeights/disneynano.pt')
    # Initialize the starting variables for the object tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    # Create a list of characters this run
    characters = []
    print("Please enter the title of your story:")
    title = str(input())
    leftright = []
    global toRead

    # Infinite loop to display
    while True:
        # capture video
        success, img = cap.read()
        # run model on single frame to detect object classes
        results = model(img, stream=True)
        # int array to track detections later
        detections = np.empty((0, 5))

        c=[]
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Values for boxes locations
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # CVZONE version bounding box
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=6)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                # save the characters seen this run
                if classNames[cls] not in c:
                    c.append(classNames[cls])
                cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=2, thickness=2,
                                   offset=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = numpy.vstack((detections, currentArray))
        characters=c
        print("These are the characters:", characters)
        resultsTracker = tracker.update(detections)

        # The below shapes will form the detection area for left and right
        # left
        limitsL = [0, 0, 200, 1080]
        cv2.rectangle(img, (limitsL[0], limitsL[1]), (limitsL[2], limitsL[3]), (0, 0, 255), cv2.FILLED)
        # right
        limitsR = [1720, 0, 1920, 1080]
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
            # Choose Left
            if limitsL[0] < cx < limitsL[2] and limitsL[1] < cy < limitsL[1]:
                leftright.append(0)
                print("adding weight to left")
            # Choose Right
            if limitsR[0] < cx < limitsR[2] and limitsR[1] < cy < limitsR[1]:
                leftright.append(1)
                print("adding weight to right")

        # cvzone.putTextRect(img, f"Count: {len(totalCount)}", (50, 50))

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 800, 600)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            print("contacting story teller first time")
            toRead = contactStoryTeller(str(characters))
            print(toRead)
            cozmo.run_program(cozmo_program, use_viewer=False, force_viewer_on_top=False)
            writeFile(title, toRead)
        if key == ord("n"):
            print("contacting story teller with choice+story")
            choice = max(leftright.count(0), leftright.count(1))
            if choice == 0:
                choice = "left"
            if choice == 1:
                choice = "right"
            toRead = contactStoryTeller(str(characters), readFile(title), choice)
            print(toRead)
            cozmo.run_program(cozmo_program, use_viewer=False, force_viewer_on_top=False)
            writeFile(title, toRead)
            leftright = []
        cv2.imshow("image", img)


# Saves story so far
def writeFile(title, story):
    # This will be used to update the story so far to keep track for the assistant.
    path = f'C:\\Users\\sherw\\PycharmProjects\\RoboticsFinal\\Stories\\{title}'
    file = open(path, "a+")
    file.seek(0)
    data = file.read(100)
    if len(data) > 0:
        file.write("\n")
    file.write(story)
    file.close()


# String of the written file
def readFile(title):
    path = f'C:\\Users\\sherw\\PycharmProjects\\RoboticsFinal\\Stories\\{title}'
    file = open(path, "r")
    content = file.read()
    return content


# Send Characters/Story so far/Choices to chat to continue story
def contactStoryTeller(characters=None, story=None, choice=None):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    if story is None:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": f"{assistant}"},
                {"role": "user", "content": f"{characters}"}
            ]
        )
    else:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": f"{assistant}"},
                {"role": "user", "content": f"{story}\n{choice}"}
            ]
        )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def cozmo_program(robot: cozmo.robot.Robot):
    robot.set_head_angle(degrees(0)).wait_for_completed()
    n=20
    words = toRead.split()
    words = [' '.join(words[i:i + n]) for i in range(0, len(words), n)]
    print("this is what words becomes",words)
    for w in words:
        robot.say_text(w,duration_scalar=0.6).wait_for_completed()
    return


readImageYolo()

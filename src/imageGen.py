import os
import cv2
directory = r"C:\Users\sherw\PycharmProjects\RoboticsFinal\Images"
#make count whatever value you left off on otherwise it will only overwrite files
count = [88]
def Main():
    cap = cv2.VideoCapture(1)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        success, img = cap.read()
        cv2.imshow("Webcam",img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            os.chdir(directory)
            end = sum(count)
            while sum(count) < end + 1:
                filename = f"{sum(count)}.jpg"
                cv2.imwrite(filename, img)
                print(f'Successfully saved picture {sum(count)}')
                count.append(1)
                if key == 27:
                    break
                print("Finished writing pictures")
            key = 0





Main()

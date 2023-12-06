import cozmo
import cv2
import numpy as np
from pyzbar.pyzbar import decode

def on_camera_image(evt, **kwargs):
    # Convert the Cozmo camera image to a format that OpenCV can work with
    image = np.asarray(evt.image.raw_image)
    image_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Perform QR code detection using pyzbar
    decoded_objects = decode(image_cv2)

    # Display the Cozmo camera feed with detected QR codes
    for obj in decoded_objects:
        s = obj.data.decode()
        print("QR Code Data:", s)

        # Draw a bounding box around the QR code
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            points = hull
        num_of_points = len(points)
        for j in range(num_of_points):
            cv2.line(image_cv2, tuple(points[j]), tuple(points[(j+1)%num_of_points]), (0, 255, 0), 3)

    # Display the processed image
    cv2.imshow("Cozmo's Camera", image_cv2)
    cv2.waitKey(1)

def cozmo_camera_qr_code():
    robot = cozmo.Robot()
    
    try:
        robot.camera.image_stream_enabled = True
        robot.start()

        print("Press 'q' to exit...")

        while True:
            latest_image = robot.world.latest_image

            if latest_image is not None:
                on_camera_image(latest_image)  # Call the image handler

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    finally:
        robot.camera.image_stream_enabled = False
        cv2.destroyAllWindows()
        robot.stop()

if __name__ == '__main__':
    cozmo_camera_qr_code()
import cozmo
import cv2
import numpy as np
from pyzbar.pyzbar import decode

def on_camera_image(evt, **kwargs):
    # Convert the Cozmo camera image to a format that OpenCV can work with
    image = np.asarray(evt.image.raw_image)
    image_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Perform QR code detection using pyzbar
    decoded_objects = decode(image_cv2)

    # Display the Cozmo camera feed with detected QR codes
    for obj in decoded_objects:
        s = obj.data.decode()
        print("QR Code Data:", s)

        # Draw a bounding box around the QR code
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            points = hull
        num_of_points = len(points)
        for j in range(num_of_points):
            cv2.line(image_cv2, tuple(points[j]), tuple(points[(j+1)%num_of_points]), (0, 255, 0), 3)

    # Display the processed image
    cv2.imshow("Cozmo's Camera", image_cv2)
    cv2.waitKey(1)

def cozmo_camera_qr_code():
    robot = cozmo.Robot()
    
    try:
        robot.camera.image_stream_enabled = True
        robot.start()

        print("Press 'q' to exit...")

        while True:
            latest_image = robot.world.latest_image

            if latest_image is not None:
                on_camera_image(latest_image)  # Call the image handler

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    finally:
        robot.camera.image_stream_enabled = False
        cv2.destroyAllWindows()
        robot.stop()

if __name__ == '__main__':
    cozmo_camera_qr_code()

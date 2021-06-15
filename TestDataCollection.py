import torch
from PIL import Image
import time
import cv2
import sys
from pathlib import Path
import shutil
from datetime import datetime

try:
    cascPath = "webcam.xml"
    name = sys.argv[1]
    set = sys.argv[2]
    side = "none"
    if set != "negative":
        side = sys.argv[3]

    if (set != "droop" and set != "negative"):
        print("Invalid set. Enter \"droop\" or \"negative\"")
        raise Exception("Invalid set")

    if (side != "right" and side != "left" and set != "negative"):
        print("Invalid side. Enter \"right\" or \"left\"")
        raise Exception("Invalid set")

    if set == "negative":
        dir_path = "test_data\\" + name + "_" +  set + "_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S") + "\\"
        file_path = name + "_" + set
    else:
        dir_path = "test_data\\" + name + "_" + side + "_" + "_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S") + "\\"
        file_path = name + "_" + side + "_" + set

    try:
        shutil.rmtree(dir_path)
    except:
        pass
    Path(dir_path).mkdir(parents=True, exist_ok=True)
except:
    print("Usage:  DataCollection.py {name} {droop/negative} {left/right/}")
    exit(-1)

faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

for i in range(0,40):
    ret, frame = video_capture.read()
    cv2.rectangle(frame, (10, 20), (20, 40), (0, 0, 255), 20)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
for i in range(0,20):
    ret, frame = video_capture.read()
    cv2.rectangle(frame, (10, 20), (20, 40), (0, 255, 255), 20)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
for i in range(0,15):
    ret, frame = video_capture.read()
    cv2.rectangle(frame, (10, 20), (20, 40), (0, 255, 0), 20)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for i in range(0, 100):
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
    )

    # Draw a rectangle around the faces
    if len(faces) == 1:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 0)
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_img = Image.fromarray(RGB_frame[y:y+h, x:x+w])
            face_img.save(dir_path + file_path + "_" + str(i) + ".jpg")

    # RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # face_img = Image.fromarray(RGB_frame)
    # face_img.save(dir_path + file_path + "_" + str(i) + ".jpg")

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

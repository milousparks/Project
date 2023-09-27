import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import math
import mediapipe as mp
maxLeng= 200
volBar = 400
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion BGR to RGB
    image.flags.writeable = False  # Image not longer writable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color conversion RGB to BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=4))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
    while cap.isOpened():

        # Read Frame
        ret, frame = cap.read()
        # make detections
        image, results = mediapipe_detection(frame, holistic)
        if results.right_hand_landmarks:
            myhand = results.right_hand_landmarks.landmark
            # print(myhand[4], myhand[8])
            x1, y1 = myhand[4].x, myhand[4].y
            # print(x1, y1)
            x2, y2 = myhand[8].x, myhand[8].y
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
            x1c, y1c = int(x1 * width), int(y1 * height)
            x2c, y2c = int(x2 * width), int(y2 * height)
            cv2.circle(image, (x1c, y1c), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2c, y2c), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(image, (x1c, y1c), (x2c, y2c), (255, 0, 255), 3)
            length = math.hypot(x2c - x1c, y2c - y1c)
            y = (length / maxLeng) * 100
            vol = str(y) + "%"
            from subprocess import call

            call(["amixer", "-D", "pulse", "sset", "Master", vol])
            # print(length)
            volBar = np.interp(length, [50, maxLeng], [400, 150])
            print(int(volBar))
            cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
            # Draw Landmarks
        draw_landmarks(image, results)
        cv2.rectangle(image, (50, 150), (85, 400), (0, 255, 0), 3)

        # Show to Screen
        cv2.imshow('OpenCV Feed', image)

        # Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

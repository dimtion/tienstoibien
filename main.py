#!/usr/bin/env python3
import os
import sys
import time
import sqlite3

import cv2
from sklearn import svm

# Waiting time between capturing and processing two frames (in ms)
UPDATE_DELAY = 1000

# Waiting time between capturing and processing two frames during a training
# phase (in ms)
UPDATE_DELAY_TRAIN = 10

# Time during which the system remains in training mode after pressing a key
# (in s)
TRAIN_DUR = 10

# Key to press to set a frame as a good one
TRAIN_GOOD_KEY = "y"
# Key to press to set a frame as a bad one
TRAIN_BAD_KEY = "t"
# Key to press to quit
QUIT_KEY = "q"

# Minimal time betweeen two notifications (in s)
NOTIFICATION_DELAY = 20

# Minimal number of predicted as wrong position before sending a notification
BAD_COUNT = 3

# Id of the video capure device, if you have only one webcam keep the default
# value
VIDEO_DEVICE_ID = 0

DATASET_DB = "dataset.sqlite3"

COLORS = {
    "good": (0, 255, 0),
    "bad": (0, 0, 255),
    "neutral": (255, 255, 255),
}


class Model:
    DATASET_SCHEMA = """
    CREATE TABLE IF NOT EXISTS dataset (
        class INTEGER,
        x1 INTEGER,
        y1 INTEGER,
        h1 INTEGER,
        w1 INTEGER,
        x2 INTEGER,
        y2 INTEGER,
        h2 INTEGER,
        w2 INTEGER
    )"""

    def __init__(self, dataset_db):
        self.conn = sqlite3.connect(dataset_db)
        self.model = svm.SVC(gamma=0.001)
        self.fitted = False

        self.dataset_x = []
        self.dataset_y = []

    def load(self):
        c = self.conn.cursor()
        c.execute(self.DATASET_SCHEMA)
        self.conn.commit()

        for row in c.execute("SELECT * FROM dataset"):
            self.dataset_y.append(row[0])
            self.dataset_x.append(row[1:])
        c.close()
        self.fit()

    def append(self, x, y):
        if not x:
            return

        self.dataset_x.append(x)
        self.dataset_y.append(y)

        c = self.conn.cursor()
        x = [int(x) for x in x]
        print((y, *x))
        c.execute("INSERT INTO dataset VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", [y, *x])
        self.conn.commit()
        c.close()

    def fit(self):
        if len(self.dataset_x) > 2 and len(set(self.dataset_y)) > 1:
            self.model.fit(self.dataset_x, self.dataset_y)
            self.fitted = True

    def predict_single(self, x):
        if self.fitted:
            return self.model.predict([x])[0]
        else:
            return None


def notify(title, text):
    os.system(
        f'osascript -e \'display notification "{title}" with title "{text}" sound name "Glass"\''
    )


def create_pt(eyes):
    # only save the two first detected eyes
    x1 = y1 = x2 = y2 = w1 = w2 = h1 = h2 = 0
    if len(eyes) == 0:
        return None
    if len(eyes) >= 1:
        x1, y1, h1, w1 = eyes[0]
    if len(eyes) >= 2:
        x2, y2, h2, w2 = eyes[1]
    return [x1, y1, w1, h1, x2, y2, w2, h2]


def main():
    # Init
    model = Model(DATASET_DB)
    model.load()

    eyesCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    )

    video_capture = cv2.VideoCapture(VIDEO_DEVICE_ID)

    last_notification = 0
    last_train = 0
    bad_count = 0

    while video_capture.isOpened():
        # Capture video
        ret, frame = video_capture.read()
        if not ret:
            print("Error capturing video output")
            notify("Ton dos !", "Error capturing video output")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eyesCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Predict position
        text = "neutral"
        pt = create_pt(eyes)
        if pt:
            prediction = model.predict_single(pt)
            if prediction == 0:
                text = "bad"
                bad_count += 1
            elif prediction == 1:
                text = "good"
                bad_count = 0
        color = COLORS[text]

        if (
            text == "bad"
            and time.time() > last_notification + NOTIFICATION_DELAY
            and bad_count > BAD_COUNT
        ):
            last_notification = time.time()
            notify("Ton dos !", "Tiens-toi bien !")

        # display
        cv2.putText(
            frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["neutral"], 2,
        )
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.imshow("Tiens toi bien !", frame)

        # Training
        if time.time() < last_train + TRAIN_DUR:
            wait_dur = UPDATE_DELAY_TRAIN
        else:
            wait_dur = UPDATE_DELAY

        pressed_key = cv2.waitKey(wait_dur) & 0xFF
        if pressed_key == ord(QUIT_KEY):
            break
        elif pressed_key == ord(TRAIN_GOOD_KEY):
            model.append(create_pt(eyes), 1)  # 1 == good
            model.fit()
            last_train = time.time()
        elif pressed_key == ord(TRAIN_BAD_KEY):
            model.append(create_pt(eyes), 0)  # 0 == bad
            model.fit()
            last_train = time.time()

    # Release capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

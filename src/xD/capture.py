#!/usr/bin/env python3


import cv2
import os.path
import signal


from datetime import datetime, timedelta


def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()


def capture(cap):

    org = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2


    checkpoint = datetime.now()

    i = 0

    while True:
        ret, frame = cap.read()

        delta = datetime.now() - checkpoint


        if delta > timedelta(seconds=1):
            ts = int(datetime.now().timestamp())
            fname = os.path.join(
                "imgs",
                f"img-{ts}-{i}.jpg",
            )

            print("SNAP! Took a photo here:", fname)

            if not cv2.imwrite(fname, frame):
                print("Writing failed!")

            i += 1
            checkpoint = datetime.now()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.putText(
            frame,
            f"countdown...{delta.seconds}.{delta.microseconds/1000:.02f}",
            org,
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        cv2.imshow('frame', frame)


        if cv2.waitKey(1) == ord('q'):
            break


def main():
    cap = cv2.VideoCapture(0)
    capture(cap)
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup(cap))
    cleanup(cap)

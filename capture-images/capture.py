#!/usr/bin/env python3


import cv2
import os.path
import signal


from datetime import datetime, timedelta


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def cleanup():
    cap.release()
    cv2.destroyAllWindows()


def capture():

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
            cv2.imwrite(fname, frame)
            i += 1
            checkpoint = datetime.now()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detects faces of different sizes in the input image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            # To draw a rectangle in a face 

            mw = 0
            mh = 0
            x = max(x - mw, 0)
            y = max(y - 2*mh, 0)
            w = w + 2*mw
            h = h + 2*mh

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (255,255,0),
                2,
            ) 
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]


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


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)


    capture()


    signal.signal(signal.SIGINT, cleanup)
    cleanup()

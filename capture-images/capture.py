#!/usr/bin/env python3


import cv2
import signal


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def cleanup():
    cap.release()
    cv2.destroyAllWindows()


def capture():
    while True:
        ret, frame = cap.read()

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


        cv2.imshow('frame', frame)


        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)


    capture()


    signal.signal(signal.SIGINT, cleanup)
    cleanup()

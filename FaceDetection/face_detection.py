import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# while cv.waitKey(33) < 0:
while(True):
    ret, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y : y +h, x : x + w]
        roi_color = frame[y : y +h, x : x + w]
        img_item = "my-image.png"
        cv.imwrite(img_item, roi_gray)

        color = (255, 0, 0) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    #display the frame
    cv.imshow("VideoFrame", frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

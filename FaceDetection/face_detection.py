import cv2 as cv
import numpy as np
import pickle

face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name" : 1}
with open("labels.pickle", 'rb') as f :
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

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

        id_, conf = recognizer.predict(roi_gray)
        if conf >=45: #and cof <= 85:
            print(id_)
            print(labels[id_])
            font = cv.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)

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

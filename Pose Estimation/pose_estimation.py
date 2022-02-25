import pafy
from cv2 import cv2 

alma_cam_url = "https://youtu.be/h6hzVOwaN_4"
alma_video = pafy.new(alma_cam_url)

quad_cam_url = "https://youtu.be/Yq4WLVuX0bU"
quad_video = pafy.new(quad_cam_url)

best_alma = alma_video.getbest(prefype = 'mp4')
best_quad = quad_video.getbest(prefype = 'mp4')

capture_alma = cv2.VideoCapture(best_alma)
capture_quad = cv2.VideoCapture(best_alma)

while True:
    grabbed, frame = capture_alma.read()
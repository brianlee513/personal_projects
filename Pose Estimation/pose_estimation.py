
import cv2
import pafy

alma_cam_url = "https://youtu.be/h6hzVOwaN_4"
alma_video = pafy.new(alma_cam_url)

quad_cam_url = "https://youtu.be/Yq4WLVuX0bU"
quad_video = pafy.new(quad_cam_url)

best_alma = alma_video.getbest()
best_quad = quad_video.getbest()

capture_alma = cv2.VideoCapture(best_alma.url)
capture_quad = cv2.VideoCapture(best_quad.url)


while cv2.waitKey(33) < 0:
    cv2.namedWindow("VideoFrame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Quad", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VideoFrame", 1920, 1080)
    cv2.resizeWindow("Quad", 1920, 1080)
    quad_ret, quad_frame = capture_quad.read()

    #cv2.resizeWindow("VideoFrame", (1920, 1080))
    alma_ret, alma_frame = capture_alma.read()
    #cv2.resizeWindow(alma_frame, 1920, 1080)

    cv2.imshow("VideoFrame", alma_frame)

    cv2.imshow("Quad", quad_frame)

capture_alma.release()
cv2.destroyAllWindows()
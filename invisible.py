import cv2
import numpy as np
import time

capture_video = cv2.VideoCapture(0)

time.sleep(1)
count = 0
background = None

for i in range(5):
    return_val, bg_frame = capture_video.read()
    if not return_val:
        print("Failed to capture background frame")
        break
    if background is None:
        background = np.zeros_like(bg_frame, dtype=np.float32)
    background += bg_frame.astype(np.float32)

if background is not None:
    background /= 5 
    background = np.flip(background, axis=1) 
else:
    print("Failed to capture background frames. Exiting.")
    exit()

lower_dark_red = np.array([0, 40, 40])
upper_dark_red = np.array([10, 255, 255])

while capture_video.isOpened():
    return_val, img = capture_video.read()
    if not return_val:
        break
    count = count + 1
    img = np.flip(img, axis=1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_dark_red, upper_dark_red)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    mask_inv = cv2.bitwise_not(mask)

    res1 = cv2.bitwise_and(background, background, mask=mask).astype(np.uint8)
    res2 = cv2.bitwise_and(img, img, mask=mask_inv).astype(np.uint8)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.namedWindow("INVISIBLE MAN", cv2.WINDOW_NORMAL)

    cv2.imshow("INVISIBLE MAN", final_output)
    k = cv2.waitKey(10)
    if k == 27:  
        break

capture_video.release()
cv2.destroyAllWindows()

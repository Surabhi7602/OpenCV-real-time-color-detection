from collections import deque
import numpy as np
import argparse
import imutils
import cv as cv
import urllib 

smaller = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80), 'brown':(1,42,170), 'pink':(85, 24, 94), 'purple':(94, 24, 72), 'white': (181, 173, 184), 'black': (0,0,0)} 
higher = {'red':(180,255,255), 'green':(85,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255), 'brown':(230,255,255), 'pink':(168, 134, 227), 'purple':(255, 0, 162), 'white': (255, 255, 255), 'black': (83, 78, 84)}
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255), 'brown':(13, 55, 113), 'pink':(255, 138, 255), 'purple':(154, 91, 222), 'white': (255, 255, 255), 'black': (0,0,0)}
 
arparser = argparse.ArgumentParser()
arparser.add_argument("-v", "--video", help="path to the (optional) video file")
arparser.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(arparser.parse_args())

if not args.get("video", False):
    camera = cv.VideoCapture(0)

else:
    camera = cv.VideoCapture(args["video"])

while True:
    (grabbed, window) = camera.read()
    if args.get("video") and not grabbed:
        break

    window = imutils.resize(window, width=1500)
    hsv = cv.cvtColor(window, cv.COLOR_BGR2HSV)

    for key, value in higher.items():
    
        kernel = np.ones((5,5),np.uint8)
        mask = cv.inRange(hsv, smaller[key], higher[key])
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
               
        findcnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)[-2]
        center = None

        if len(findcnts) > 0:
            c = max(findcnts, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(c)
            M = cv.moments(c)
           # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = (cX, cY)

            if radius > 0.3:
                cv.circle(window, (int(x), int(y)), int(radius), colors[key], 5)
                cv.putText(window, "color = " + key , (int(x-radius),int(y-radius)), cv.FONT_HERSHEY_PLAIN, 0.9, colors[key], 2)
 
     
    cv.imshow("Display", window)

    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break
 
camera.release()
cv.destroyAllWindows()

#python webcamcolordetector.py

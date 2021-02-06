import cv2
import numpy as np
from tqdm import tqdm
import os
import json


window = 'Image'
low_h = 'lowH'
low_s = 'lowS'
low_v = 'lowV'
high_h = 'highH'
high_s = 'hishS'
high_v = 'highV'

DATA = 'task-2'
HSV_JSON = 'data.json'

HSVLOW = np.zeros(3)
HSVHIGH = np.zeros(3)

img = None
hsv = None
res = None
area = []

def callback(v):
    global hsv
    global img

    if np.any(hsv) != None:
        H_low = cv2.getTrackbarPos(low_h, window)
        H_high = cv2.getTrackbarPos(high_h, window)
        S_low = cv2.getTrackbarPos(low_s, window)
        S_high = cv2.getTrackbarPos(high_s, window)
        V_low = cv2.getTrackbarPos(low_v, window)
        V_high = cv2.getTrackbarPos(high_v, window)

        HSVLOW = np.array([H_low, S_low, V_low])
        HSVHIGH = np.array([H_high, S_high, V_high])

        #making mask for hsv range
        mask = cv2.inRange(hsv, tuple(HSVLOW), tuple(HSVHIGH))

        #masking HSV value selected color becomes black
        res = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, 1, 2)


        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            a = cv2.contourArea(cnt)

            if a > 300:
                area.append(a)
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow(window, res)
        cv2.imshow('Original Image', img)
    else:
        print(hsv)


def draw_contours(res):
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, 1, 2)


    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        a = cv2.contourArea(cnt)

        if a > 600:
            area.append(a)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)


def set_trackbar_values(data): 
    cv2.setTrackbarPos(low_h, window, data['h_low'])
    cv2.setTrackbarPos(low_s, window, data['s_low'])
    cv2.setTrackbarPos(low_v, window, data['v_low'])
    
    cv2.setTrackbarPos(high_h, window, data['h_high'])
    cv2.setTrackbarPos(high_s, window, data['s_high'])
    cv2.setTrackbarPos(high_v, window, data['v_high'])

if __name__ == '__main__':
    files = sorted(os.listdir(DATA))
    cv2.namedWindow(window)

    #create trackbars for high,low H,S,V 
    cv2.createTrackbar(low_h, window, 0, 179, callback)
    cv2.createTrackbar(high_h, window, 0, 179, callback)
    
    cv2.createTrackbar(low_s, window, 0, 255, callback)
    cv2.createTrackbar(high_s, window, 0, 255, callback)
    
    cv2.createTrackbar(low_v, window, 0, 255, callback)
    cv2.createTrackbar(high_v, window, 0, 255, callback)

    with open(HSV_JSON) as f:
        data = json.load(f)[-1]
        HSVLOW[0] = data['h_low']
        HSVLOW[1] = data['s_low']
        HSVLOW[2] = data['v_low']

        HSVHIGH[0] = data['h_high']
        HSVHIGH[1] = data['s_high']
        HSVHIGH[2] = data['v_high']

        # set_trackbar_values(data)

    for i in files:
        img = cv2.imread(os.path.join(DATA, i))
        img = cv2.resize(img, (720, 480))
        img = cv2.GaussianBlur(img, (5, 5), 0)

        #convert sourece image to HSC color mode
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #making mask for hsv range
        mask = np.array(cv2.inRange(hsv, HSVLOW, HSVHIGH))

        #masking HSV value selected color becomes black
        res = cv2.bitwise_and(img, img, mask=mask)
        # draw_contours(res)

        # cv2.imshow('Mask', mask)
        cv2.imshow(window, res)
        cv2.imshow('Original Image', img)

        #waitfor the user to press escape and break the while loop 
        k = cv2.waitKey()
        if k == 27:
            break

    #destroys all window
    cv2.destroyAllWindows()
    print(area)
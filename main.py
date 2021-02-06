import os
import cv2
import numpy as np
import imutils
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

DATA = 'task-2'
BASE_FRAME = 'frame_0.jpg'
HSVLOW = tuple([17, 34, 146])
HSVHIGH = tuple([51, 217, 226])
area = []
table_centers = []
table_rectangles = []
counter_dict = {}

def do_intersect(table_moment, ppl_moment, img) -> bool:
    '''
    This method calculates the center of a table and a person and then calculates the distance between the table center and ther person center.
    If the distance is between a certain threshold value, then the person is sitting on the table.
    '''
    tcx = None
    tcy = None 
    pcx = None
    pcy = None
    if table_moment['m00'] != 0:
        tcx = int(table_moment["m10"] / table_moment["m00"])
        tcy = int(table_moment["m01"] / table_moment["m00"])

    if ppl_moment['m00'] != 0:
        pcx = int(ppl_moment["m10"] / ppl_moment["m00"])
        pcy = int(ppl_moment["m01"] / ppl_moment["m00"])

        cv2.circle(img, (pcx, pcy), 3, (0, 255, 0), -1)

    if tcx and tcy and pcx and pcy:
        Dx = abs(tcx - pcx)
        Dy = abs(tcy - pcy)

        if Dx < 25 and Dy < 25:
            return True

    return False




def draw_contours_and_track(frame_name, img, base_frame, res) -> None:
    '''
    This method will perform motion tracking as well as try to segment the tables from rest of the scene.
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find the delta between the base frame and the current frame
    delta = cv2.absdiff(base_frame, blurred)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # If there is a significant delta then a motion has occured (probably a person?)
    ppl_cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ppl_cnts = imutils.grab_contours(ppl_cnts)

    # Find table contours
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    table_cnts, _ = cv2.findContours(gray, 1, 2)

    for table_cnt in table_cnts:
        for ppl_cnt in ppl_cnts:
            tx, ty, tw, th = cv2.boundingRect(table_cnt)
            table_area = cv2.contourArea(table_cnt)
            table_moment = cv2.moments(table_cnt)

            # if the contour area  is more than 550 then mostly it's a table
            if table_area > 550:
                cv2.rectangle(res, (tx,ty), (tx+tw, ty+th), (0,255,0), 2)

                px, py, pw, ph = cv2.boundingRect(ppl_cnt)
                ppl_area = cv2.contourArea(ppl_cnt)

                # if the contour area  is more than 1000 then mostly it's a person
                if ppl_area > 1000:
                    cv2.rectangle(img, (px, py), (px + pw, py + ph), (255, 0, 255), 2)
                    ppl_moment = cv2.moments(ppl_cnt)

                    if do_intersect(table_moment, ppl_moment, img):
                        # print('{} has person present'.format(frame_name))
                        counter_dict[frame_name] += 1


if __name__ == '__main__':
    base_frame = cv2.resize(cv2.imread(os.path.join(DATA, BASE_FRAME)), (720, 480))
    base_frame = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    base_frame = cv2.GaussianBlur(base_frame, (5, 5), 0)

    files = sorted(os.listdir(DATA))
    for f in tqdm(files):
        counter_dict[f] = 0
        img = cv2.imread(os.path.join(DATA, f))
        img = cv2.resize(img, (720, 480))
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        # convert sourece image to HSV color mode
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # making mask for hsv range (the HSV range is calibrated to extract the table)
        mask = np.array(cv2.inRange(hsv, HSVLOW, HSVHIGH))

        # masking HSV value selected color becomes black
        res = cv2.bitwise_and(img, img, mask=mask)

        draw_contours_and_track(f, img, base_frame, res)

        cv2.imshow('Output', res)
        cv2.imshow('Original Image', img)

        # wait for the user to press escape and break the loop 
        k = cv2.waitKey(1)
        if k == 27:
            break


    x = []
    y = []
    for i in tqdm(range(0, len(os.listdir(DATA)))):
        x.append(i)
        y.append(counter_dict['frame_{}.jpg'.format(i)])
    
    df = pd.DataFrame(data={'Frame No/Time': x, 'People Count': y})
    sns.barplot(x='Frame No/Time', y='People Count', data=df)

    plt.show()
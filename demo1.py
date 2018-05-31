from pypylon import pylon
from pypylon import genicam

import sys
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def simplifyImg(frame, width_size = 500.0):
    ''' Simplify the image to make possible movement detection'''
    # Resize image
    r = width_size / frame.shape[1]
    dim = (int(width_size), int(frame.shape[0] * r))
    frame_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    # Convert it to grayscale, and blur it
    # gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray = frame_resized
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    return gray, r

def singleMovDetect(img, simple_img_ref, min_area = 500):
    ''' Compare the image and the reference image to detect changes'''
    # Get dimensions of the img
    dim = img.shape[0:2]

    # Simplify the image to make it comparable
    simple_img, ratio = simplifyImg(img)

    # --- Compare both the images ---
    # compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(simple_img_ref, simple_img)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    image, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create list of bbox
    bboxes = []
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        # compute the bounding box for the contour, draw it on the frame
        bboxes.append(cv2.boundingRect(c))

    # Merge the bboxes incase
    if len(bboxes) > 0:
        if len(bboxes) > 1:
            dim_array = np.zeros((len(bboxes), 4))

            for i, bbox in enumerate(bboxes):
                (x, y, w, h) = bbox
                dim_array[i, :] = np.array((x, y, (x + w), (y + h)))

            bbox = (np.min(dim_array[:,0]),
                    np.min(dim_array[:,1]),
                    np.max(dim_array[:,2]) - np.min(dim_array[:,0]),
                    np.max(dim_array[:,3]) - np.min(dim_array[:,1]))

        else:
            bbox = bboxes[0]

        # Scale up the bbox
        bbox = [int(dim / ratio) for dim in bbox]

    else:
        bbox = []

    # Scale up the mask
    mask = cv2.resize(thresh, dim, interpolation = cv2.INTER_AREA)

    return bbox, mask

def detect_defects(param):
    img_ref = None
    found_img_ref = False
    img_counter = 0

    try:
        # Create an instant camera object with the camera device found first.
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # Print the model name of the camera.
        print("Using device ", camera.GetDeviceInfo().GetModelName())

        # The parameter MaxNumBuffer can be used to control the count of buffers
        # allocated for grabbing. The default value of this parameter is 10.
        camera.MaxNumBuffer = 5

        # Start the grabbing of c_countOfImagesToGrab images.
        # The camera device is parameterized with a default configuration which
        # sets up free-running continuous acquisition.
        # countOfImagesToGrab = 100
        # camera.StartGrabbingMax(countOfImagesToGrab)

        camera.StartGrabbing()

        # Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        # when c_countOfImagesToGrab images have been retrieved.
        while camera.IsGrabbing():
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                # Access the image data.
                # print("SizeX: ", grabResult.Width)
                # print("SizeY: ", grabResult.Height)
                # print("Gray value of first pixel: ", img[0, 0])

                img = grabResult.Array

                if not found_img_ref:
                    img_ref, _ = simplifyImg(img)
                    found_img_ref = True
                else:
                    bbox, mask = singleMovDetect(img, img_ref)

                    if bbox:
                        # Draw bbox
                        if param['draw_bbox']:
                            (x, y, w, h) = bbox
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Save image
                        cv2.imwrite(os.path.join(param['path2save'], str(img_counter).zfill(4) + param['img_ext']), img)
                        img_counter += 1
            else:
                print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
            grabResult.Release()

    except genicam.GenericException as e:
        # Error handling.
        print("An exception occurred.")
        print(e.GetDescription())
        exitCode = 1

    sys.exit(exitCode)

if __name__ == '__main__':
    param = {'path2save': '/home/deepeye/Desktop',
             'img_ext': '.png',
             'draw_bbox': True}

    detect_defects(param)

    # Test functions
    # frame = cv2.imread('/home/deepeye/Documents/Test_images/01.bmp', -1)
    #
    # gray, r = simplifyImg(frame)
    #
    # frame_2 = cv2.imread('/home/deepeye/Documents/Test_images/24.bmp', -1)
    #
    # bbox, thresh = singleMovDetect(frame_2, gray)
    #
    # (x, y, w, h) = bbox
    # cv2.rectangle(frame_2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # cv2.imwrite('/home/deepeye/Documents/test.png', frame_2)
    # print(r)

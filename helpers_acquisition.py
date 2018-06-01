import cv2
import numpy as np

def simplifyImg(frame, width_size = 500.0):
    ''' Simplify the image to make possible movement detection'''
    # Resize image
    r = width_size / frame.shape[1]
    dim = (int(width_size), int(frame.shape[0] * r))
    frame_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    # Convert it to grayscale, and blur it
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    # gray = frame_resized
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

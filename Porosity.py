
'''
# basic img functions
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
clone = np.copy(gray)
binImg = cv2.threshold(clone, int(100), 255, cv2.THRESH_BINARY)[1]

# applying masks
res = np.multiply(gray, mask)
if area == 'hatch':
    cont_res = np.multiply(gray, cont_mask)
'''

import cv2
import numpy as np


def findSectionMask(binImg, area):
    # Takes a binary image and str.
    # Will find the mask of the sample inside the picture.
    # intended for high used with high constrast micrograph images
    # to find a mask of the whole sample or hatching/contour areas

    # IN:
    # img - a numpy array image (should be a highly polished sample section in contrasting substrate)

    # area - a str defining the area chosen (hatch, section, whole)
    # 	hatch - return two images: the hatching area & contour area
    #   section - the whole sample section
    #   whole image

    # OUT:
    # list - list of masks, either 1 long, or two when hatch is called and contour is returned

    contours, hierarchy = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    masks = []

    if area == 'hatch':
        sectionCont = max(contours, key=lambda x: cv2.contourArea(x))
        sectionIndex = list(map(lambda x: (np.array_equal(x, sectionCont)) & (cv2.contourArea(x) < binImg.size), contours)).index(True)

        mask = np.zeros(binImg.shape[0:2], np.uint8)  # hatch mask
        cv2.drawContours(mask, contours, sectionIndex, (255, 0, 0), thickness=cv2.FILLED)  # make hatch mask = section mask
        cont_mask = np.copy(mask)

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        mask = cv2.threshold(dist, 0.1125, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
        cont_mask = (cv2.subtract(cont_mask, mask * 255) / 255).astype('uint8')
        masks = [mask, cont_mask]

    elif area == 'section':
        sectionCont = max(contours, key=lambda x: cv2.contourArea(x))
        sectionIndex = list(map(lambda x: (np.array_equal(x, sectionCont)) & (cv2.contourArea(x) < binImg.size), contours)).index(True)

        mask = np.zeros(binImg.shape[0:2], np.uint8)
        cv2.drawContours(mask, contours, sectionIndex, (1, 1, 1), thickness=cv2.FILLED)
        masks = [mask]

    elif area == 'whole':
        mask = np.ones(binImg.shape[0:2])
        mask = masks

    return masks


def porosityBasic(img, mask=None, start=0, end=255):
    # estimates porosity from an image for a range of porosities on a mask
    # average run time: 10.53s (very large image 8kx8k)

    # IN:
    # img - a np array image
    # mask - the mask to be used to select a specifc area of the image, a np array with vals between 0-1
    # start - lowest threshold to be applied
    # end - the highest threshold to be applied

    # OUT:
    # porosity - a list of porosity values for each threshold val
    # thresholds - a list of threshold correcposonding to each porosity val

    pts = np.where(mask == 1)
    lst_intensities = np.zeros_like(pts)
    lst_intensities = res[pts[0], pts[1]]

    porosity = np.zeros(end - start)
    for thr in range(start, end):
        porosity[thr - start] = len(lst_intensities[lst_intensities > thr]) / len(lst_intensities)

    return porosity, list(range(start, end))

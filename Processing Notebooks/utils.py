import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os

import random

def get_contours(img):
    
    image = img.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    inv_gray = cv2.bitwise_not(gray)
    
    ret, thresh = cv2.threshold(inv_gray, 200, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    image_copy = img.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    return gray, inv_gray, thresh, image_copy, contours
     
    
def clean_contours(contours):
    
    # Remove any contours that is not closed
    contours = [x for x in contours if cv2.arcLength(x, True) > 200 and cv2.arcLength(x, True) < 2000]
    
    # Remove any contours whose area is too small or too big
    contours = [x for x in contours if cv2.contourArea(x) > np.pi*150*150 and cv2.contourArea(x) < 380*380]
    
    return contours


def get_centroids(img, contours):
    
    # Finding center of cell
    image_copy = img.copy()

    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(image_copy, [i], -1, (0, 255, 0), 2)
            cv2.circle(image_copy, (cx, cy), 7, (0, 0, 255), -1)
            
    return image_copy


def get_final_wall(img, contours):
    
    # Including outer ring, extract the outermost ring
    biggest_area = 0
    ring = None

    for i in contours:
        area = cv2.contourArea(i)

        if biggest_area < cv2.contourArea(i):
            biggest_area = area
            ring = i
            
    image_copy = img.copy()

    M = cv2.moments(ring)

    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.drawContours(image_copy, [ring], -1, (0, 255, 0), 2)
        
    return image_copy, ring


def extract_cell(gray, ring):
    
    rect = cv2.boundingRect(ring)
    x,y,w,h = rect
    
    ring = ring.reshape((1, ring.shape[0], -1))
    
    mask = np.zeros(gray.shape).astype(gray.dtype)
    cv2.fillPoly(mask, ring, (255,255,255))
    masked = cv2.bitwise_and(gray, mask)
    
    extracted_image = np.zeros(gray.shape)
    extracted_image[:h, :w] = masked[y:y+h, x:x+w]
    extracted_image = extracted_image[:h, :w]
    
    return extracted_image


def show_processed_image(img, show_plot=True):
    
    gray, inv_gray, thresh, contoured, contours = get_contours(img)

    contours = clean_contours(contours)
    
    centroids = get_centroids(img, contours)

    final_image, ring = get_final_wall(img, contours)
    
    extracted_image = extract_cell(gray, ring)
    
    if show_plot:
        fig, ax = plt.subplots(1, 8, figsize=(24, 8))
        
        ax[0].imshow(img)
        ax[0].set_title('Original Image')

        ax[1].imshow(gray, cmap='gray')
        ax[1].set_title('Grayscale Image')

        ax[2].imshow(inv_gray, cmap='gray')
        ax[2].set_title('Inverted Grayscale Image')

        ax[3].imshow(thresh)
        ax[3].set_title('Extracted Area Image')

        ax[4].imshow(contoured)
        ax[4].set_title('Contours of Image')

        ax[5].imshow(centroids)
        ax[5].set_title('Centroids of Contours')

        ax[6].imshow(final_image)
        ax[6].set_title('Final Cell Image')

        ax[7].imshow(extracted_image, cmap='gray')
        ax[7].set_title('Extracted Wall')
    
    return extracted_image



def display_random_images(images, k=16, figsize=(25, 25)):
    
    side = int(np.sqrt(k))

    sampled_images = random.sample(images, k=k)

    fig, ax = plt.subplots(side, side, figsize=figsize)

    for i in range(k):

        row, col = i//(side), i%(side)

        ax[row][col].imshow(sampled_images[i][1], cmap='gray')
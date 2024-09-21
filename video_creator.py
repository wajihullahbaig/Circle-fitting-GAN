# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 20:37:59 2024

@author: Acer
"""

import cv2
import os

# create video from frames
def create_video_from_images(folder):
    video_filename = 'fit_2_features.mp4'
    valid_images = [i for i in os.listdir(folder) if i.endswith((".jpg", ".jpeg", ".png"))]
    # sort files on creation data
    valid_images.sort(key=lambda f: os.path.getctime(os.path.join(folder, f)))

    first_image = cv2.imread(os.path.join(folder, valid_images[0]))
    h, w, _ = first_image.shape

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(video_filename, codec, 30, (w, h))

    for img in valid_images:
        print(img)        
        loaded_img = cv2.imread(os.path.join(folder, img))        
        for i in range(15):
            vid_writer.write(loaded_img)

    vid_writer.release()

# Create video from resized images
img_dir = 'images_fit_2_features/'
create_video_from_images(img_dir)

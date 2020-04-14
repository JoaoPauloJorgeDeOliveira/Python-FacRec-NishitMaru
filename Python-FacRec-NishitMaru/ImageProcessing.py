import os
import cv2

# Paths and file names:
image_file_name = 'bill gates.jpg'
current_directory = os.path.dirname(__file__)
images_folder_path = os.path.join(current_directory, 'faces')
image_path = os.path.join(images_folder_path, image_file_name)

# Reading image:
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Displaying image:
cv2.imshow('Image', image)
cv2.waitKey()               # Ref: https://stackoverflow.com/questions/21810452/cv2-imshow-command-doesnt-work-properly-in-opencv-python
# Gray scale:
cv2.imshow('Image_gray', image_gray)
cv2.waitKey()
# HSV: hue, saturation, value:
cv2.imshow('Image_hsv', image_hsv)
cv2.waitKey()

# Saving image:
image_path_2 = os.path.join(images_folder_path, 'bill gates_2.jpg')
cv2.imwrite(image_path_2, image)

# Resizing:
image_resized = cv2.resize(image, (200,200))
cv2.imshow('Window name', image_resized)
cv2.waitKey()

# Rotating:
rows, cols = image.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1) # Parameters: center, angle (degrees), scale
image_rotated = cv2.warpAffine(image, M, (cols, rows))
cv2.imshow('Window name', image_rotated)
cv2.waitKey()

# Moving:
import numpy
M = numpy.float32([[1, 0, -100], [0, 1, -100]])         # Creating matrix.
print(M)
image_moved = cv2.warpAffine(image, M, (cols, rows))
cv2.imshow('Window name', image_moved)
cv2.waitKey()

# Detecting edges:
image_edges = cv2.Canny(image, 100,200)
cv2.imshow('Window name', image_edges)
cv2.waitKey()
import os
import face_recognition     # Ref: https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py
import cv2
import numpy
import datetime

time_start = datetime.datetime.now()

# Paths and file names:
current_directory = os.path.dirname(__file__)
known_images_folder_path = os.path.join(current_directory, 'faces')
unknown_images_folder_path = os.path.join(current_directory, 'test')


# Getting known encodings:
database_size = len(os.listdir(known_images_folder_path))
i = 0
known_encodings = [0]*database_size
name = [0]*database_size
for known_image in os.listdir(known_images_folder_path):                            # Ref: https://stackoverflow.com/questions/1120707/using-python-to-execute-a-command-on-every-file-in-a-folder
    full_path_known = os.path.join(known_images_folder_path, known_image)           # Getting image full path.
    image_fac_rec_known = face_recognition.load_image_file(full_path_known)         # Loading known image on face recognition module.
    known_encodings[i] = face_recognition.face_encodings(image_fac_rec_known)[0]    # Getting first encoding (method returns one for each face in image).
    name[i], _ = os.path.splitext(known_image)                                      # Getting file (person) name. Ref: https://openwritings.net/pg/python/python-how-get-filename-without-extension
    i += 1

print('Got encodings for known people.')


# For each test image:
for unknown_image in os.listdir(unknown_images_folder_path):                        # Ref: https://stackoverflow.com/questions/1120707/using-python-to-execute-a-command-on-every-file-in-a-folder

    full_path_unknown = os.path.join(unknown_images_folder_path, unknown_image)     # Getting image full path.

    # To make face recognition (face_recognition module):
    image_fac_rec_unknown = face_recognition.load_image_file(full_path_unknown)     # Loading unknown image on face recognition module.
    loc = face_recognition.face_locations(image_fac_rec_unknown)                    # Detecting faces locations.
    unknown_encodings = face_recognition.face_encodings(image_fac_rec_unknown, loc) # Getting list with encodings of faces found.

    # To display image (OpenCV module):
    image_cv2 = cv2.imread(full_path_unknown)                                       # Loading unknown image on OpenCV.



    # Face counter:
    i = 0

    # For each face (encoding) found in test image:
    for unknown_enc in unknown_encodings:

        distances = face_recognition.face_distance(known_encodings, unknown_enc)        # Getting distances between faces.
        tolerance = 0.6

        index = numpy.argmin(distances)                                                 # Getting index for most similar person (lower distance). Ref: https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
        title = name[index] if distances[index] < tolerance else 'unknown'              # If found person (distance < tolerance), getting him/her name.

        # Drawing rectangles on faces:
        cv2.rectangle(image_cv2,                                            # Image.                                            
                        (loc[i][3], loc[i][0]), (loc[i][1], loc[i][2]),     # Corners
                        (255,0,0),                                          # Color
                        2,                                                  # Thickness
                        )
        # Putting name label - Border:
        cv2.putText(image_cv2,                                              # Image.
                    f'{i}: {title}',                                        # Text.
                    (loc[i][3]-10, loc[i][0]-10),                           # Location.
                    cv2.FONT_HERSHEY_SIMPLEX,                               # Font.
                    0.5,                                                    # Font scale.
                    (0,0,0),                                                # Color.
                    4,                                                      # Thickness.
                    )
        # Putting name label:
        cv2.putText(image_cv2,                                              # Image.
                    f'{i}: {title}',                                        # Text.
                    (loc[i][3]-10, loc[i][0]-10),                           # Location.
                    cv2.FONT_HERSHEY_SIMPLEX,                               # Font.
                    0.5,                                                    # Font scale.
                    (255,255,255),                                          # Color.
                    1,                                                      # Thickness.
                    )
        
        i += 1


    # Displaying image:
    cv2.imshow('Detected faces', image_cv2)
    cv2.waitKey()

print('Elapsed '+ str(datetime.datetime.now() - time_start))
import os
import face_recognition
import cv2
import pprint

# Paths and file names:
current_directory = os.path.dirname(__file__)
known_images_folder_path = os.path.join(current_directory, 'faces')
unknown_images_folder_path = os.path.join(current_directory, 'test')


# Getting known encodings:
for unknown_image in os.listdir(unknown_images_folder_path):                 # Ref: https://stackoverflow.com/questions/1120707/using-python-to-execute-a-command-on-every-file-in-a-folder

    # Opening unknown image:
    full_path_unknown = os.path.join(unknown_images_folder_path, unknown_image) # Getting image full path.
    image_fac_rec_unknown = face_recognition.load_image_file(full_path_unknown) # Loading unknown image on face recognition module.
    image_cv2 = cv2.imread(full_path_unknown)                                   # Loading unknown image on OpenCV.
    
    # Getting face encodings (?):
    unknown_encodings = face_recognition.face_encodings(image_fac_rec_unknown)

    # Detecting faces locations:
    loc = face_recognition.face_locations(image_fac_rec_unknown) 

    # Face counter:
    i = 0

    # For each encoding found:
    for unknown_enc in unknown_encodings:

        face_found = False
        end_of_directory = False
        distance_1, distance_2 = 1

        while not face_found or not end_of_directory:

            for known_image in os.listdir(known_images_folder_path):                                # For each image on known folder:
                full_path_known = os.path.join(known_images_folder_path, known_image)               # Getting image full path.
                image_fac_rec_known = face_recognition.load_image_file(full_path_known)             # Loading known image on face recognition module.
                known_encoding = face_recognition.face_encodings(image_fac_rec_known)[0]            # Getting encoding.
                comparison = face_recognition.compare_faces([known_encoding], unknown_enc)[0]       # Comparing faces.

                if comparison == True :                                                             # If found face.
                    face_found = True
                    distance_1 = face_recognition.face_distance([known_encoding], unknown_enc)[0]   # Getting distance.
                    break                                                                           # Breaking for loop.

            end_of_directory =  True                                                                # If reached end of directory.

        if face_found :
            file, ext = os.path.splitext(known_image)   # Getting file name. Ref: https://openwritings.net/pg/python/python-how-get-filename-without-extension
            name = file
        else:
            name = 'Unknown'

        # Drawing rectangles on faces:
        if True:
            cv2.rectangle(image_cv2,                                        # Image.                                            
                         (loc[i][3], loc[i][0]), (loc[i][1], loc[i][2]),    # Corners
                         (255,0,0),                                         # Color
                         2,                                                 # Thickness
                         )
            # Putting name label - Border:
            cv2.putText(image_cv2,                                          # Image.
                        f'{i}: {name}',                                     # Text.
                        (loc[i][3]-10, loc[i][0]-10),                       # Location.
                        cv2.FONT_HERSHEY_SIMPLEX,                           # Font.
                        0.5,                                                # Font scale.
                        (0,0,0),                                            # Color.
                        4,                                                  # Thickness.
                        )
            # Putting name label:
            cv2.putText(image_cv2,                                          # Image.
                        f'{i}: {name}',                                     # Text.
                        (loc[i][3]-10, loc[i][0]-10),                       # Location.
                        cv2.FONT_HERSHEY_SIMPLEX,                           # Font.
                        0.5,                                                # Font scale.
                        (255,255,255),                                      # Color.
                        1,                                                  # Thickness.
                        )
        
        i += 1


    # Displaying image:
    cv2.imshow('Detected faces', image_cv2)
    cv2.waitKey()
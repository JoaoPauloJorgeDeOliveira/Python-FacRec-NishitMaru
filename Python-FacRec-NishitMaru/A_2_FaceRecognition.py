import os
import face_recognition
import cv2
import pprint

# Paths and file names:
person_name = 'Obama'
current_directory = os.path.dirname(__file__)
images_folder_path = os.path.join(current_directory, 'faces')
image_path_known = os.path.join(images_folder_path, 'obama.jpg')
image_path_unknown = os.path.join(images_folder_path, 'obama_test.png')

# Opening image:
image_cv2 = cv2.imread(image_path_known)                                        # On OpenCV.
image_fac_rec = face_recognition.load_image_file(image_path_known)              # On face recognition module, known face.
image_fac_rec_unknown = face_recognition.load_image_file(image_path_unknown)    # On face recognition module, unknown face.

# Detecting face location:
loc = face_recognition.face_locations(image_fac_rec)[0]
# Drawing rectangle on image.
cv2.rectangle(image_cv2,                                        # Image.                                            
              (loc[3], loc[0]), (loc[1], loc[2]),               # Corners
              (255,0,0),                                        # Color
              2,                                                # Thickness
              )
# Putting name label - Border:
cv2.putText(image_cv2,                                          # Image.
            person_name,                                        # Text.
            (loc[3]-10, loc[0]-10),                             # Location.
            cv2.FONT_HERSHEY_SIMPLEX,                           # Font.
            0.5,                                                # Font scale.
            (0,0,0),                                            # Color.
            4,                                                  # Thickness.
            )
# Putting name label:
cv2.putText(image_cv2,                                          # Image.
            person_name,                                        # Text.
            (loc[3]-10, loc[0]-10),                             # Location.
            cv2.FONT_HERSHEY_SIMPLEX,                           # Font.
            0.5,                                                # Font scale.
            (255,255,255),                                      # Color.
            1,                                                  # Thickness.
            )

# Displaying image:
cv2.imshow('Image', image_cv2)
cv2.waitKey()


# Getting face features:
features = face_recognition.face_landmarks(image_fac_rec)
pprint.pprint(features)

# ?
known_encodings = face_recognition.face_encodings(image_fac_rec)[0]
unknown_encodings = face_recognition.face_encodings(image_fac_rec_unknown)[0]

# Comparing faces:
comparison = face_recognition.compare_faces([known_encodings], unknown_encodings)[0]
print('Comparison: ' + str(comparison))

# Distances:
distance =  face_recognition.face_distance([known_encodings], unknown_encodings)[0]
print('Distance: ' + str(distance))
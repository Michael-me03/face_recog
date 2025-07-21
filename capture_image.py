import face_recognition
import cv2
import numpy as np
import os

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Use relative paths or check if files exist
script_dir = os.path.dirname(os.path.abspath(__file__))
captured_images_dir = os.path.join(script_dir, "captured_images")

# Check if the directory exists
if not os.path.exists(captured_images_dir):
    print(f"Error: Directory '{captured_images_dir}' not found!")
    print("Please create the 'captured_images' directory and add your image files.")
    exit()

# Load a sample picture and learn how to recognize it.
obama_path = os.path.join(captured_images_dir, "/Users/michaelmeier/Coding/face_recog/captured_images/img1.jpg")
biden_path = os.path.join(captured_images_dir, "/Users/michaelmeier/Coding/face_recog/captured_images/Biden.jpg")

# Check if files exist before loading
if not os.path.exists(obama_path):
    print(f"Error: File '{obama_path}' not found!")
    print("Please add img1.jpg to the captured_images directory.")
    exit()

if not os.path.exists(biden_path):
    print(f"Error: File '{biden_path}' not found!")
    print("Please add img2.jpg to the captured_images directory.")
    exit()

# Load the images with error handling
try:
    obama_image = face_recognition.load_image_file(obama_path)
    obama_face_encodings = face_recognition.face_encodings(obama_image)
    if len(obama_face_encodings) == 0:
        print("No face found in obama.jpg")
        exit()
    obama_face_encoding = obama_face_encodings[0]
    print("✓ Obama image loaded successfully")
except Exception as e:
    print(f"Error loading Obama image: {e}")
    exit()

try:
    biden_image = face_recognition.load_image_file(biden_path)
    biden_face_encodings = face_recognition.face_encodings(biden_image)
    if len(biden_face_encodings) == 0:
        print("No face found in Biden.jpg")
        exit()
    biden_face_encoding = biden_face_encodings[0]
    print("✓ Biden image loaded successfully")
except Exception as e:
    print(f"Error loading Biden image: {e}")
    exit()

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Michi",
    "Joe Biden"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

print("Starting face recognition... Press 'q' to quit.")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces and face encodings in the current frame of video
        try:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            
            # Only try to get encodings if we found faces
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
            else:
                face_encodings = []
                
        except Exception as e:
            print(f"Error in face detection: {e}")
            face_locations = []
            face_encodings = []

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            try:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]
                    
            except Exception as e:
                print(f"Error comparing faces: {e}")
                name = "Unknown"

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
print("Face recognition stopped.")
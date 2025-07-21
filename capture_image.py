import face_recognition
import cv2
import numpy as np
import os
import stripe

stripe.api_key = "API_KEY_HERE"
amount_eur = 10.0

def charge_recognized_user(amount_eur):
    try:
        amount_cents = int(float(amount_eur) * 100)
        payment_intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency='eur',
            payment_method_types=['card'],
        )
        print("💳 Stripe PaymentIntent created:", payment_intent['id'])
        return True, payment_intent['id']
    except Exception as e:
        print("❌ Stripe error:", e)
        return False, str(e)

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
martin_path = os.path.join(captured_images_dir, "/Users/michaelmeier/Coding/face_recog/captured_images/Martin.jpg")
linus_path = os.path.join(captured_images_dir, "/Users/michaelmeier/Coding/face_recog/captured_images/Linus.jpg")
pascal_path = os.path.join(captured_images_dir, "/Users/michaelmeier/Coding/face_recog/captured_images/Pascal.jpg")

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

try:
    martin_image = face_recognition.load_image_file(martin_path)
    martin_face_encodings = face_recognition.face_encodings(martin_image)
    if len(martin_face_encodings) == 0:
        print("No face found in Martin.jpg")
        exit()
    martin_face_encoding = martin_face_encodings[0]
    print("✓ Martin image loaded successfully")
except Exception as e:
    print(f"Error loading Martin image: {e}")
    exit()

try:
    linus_image = face_recognition.load_image_file(linus_path)
    linus_face_encodings = face_recognition.face_encodings(linus_image)
    if len(linus_face_encodings) == 0:
        print("No face found in Linus.jpg")
        exit()
    linus_face_encoding = linus_face_encodings[0]
    print("✓ Linus image loaded successfully")
except Exception as e:
    print(f"Error loading Linus image: {e}")
    exit()

try:
    pascal_image = face_recognition.load_image_file(pascal_path)
    pascal_face_encodings = face_recognition.face_encodings(pascal_image)
    if len(pascal_face_encodings) == 0:
        print("No face found in Pascal.jpg")
        exit()
    pascal_face_encoding = pascal_face_encodings[0]
    print("✓ Pascal image loaded successfully")
except Exception as e:
    print(f"Error loading Pascal image: {e}")
    exit()

# Mindest Abstand von Personen für die Gesichtserkennung
# Drei/eine sekunden warten, um sicherzustellen, dass es die richtige Person ist
# Möglichkeit in einer Datenbank neue einträge zu erstellen aber auch zu löschen
# Clusterung der Gesichter, um zu erkennen, ob es sich um eine Person handelt, die schon mal da war
# Clustering von mehrren Gesichtern eine Person zuordnen für kontinuieerliches Lernen

known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    martin_face_encoding,
    pascal_face_encoding,
    linus_face_encoding
]
known_face_names = [
    "Michi",
    "Joe Biden",
    "Prof. Martin",
    "Pascal",
    "Linus"
]

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
        # Resize frame of video to 25% size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.10, fy=0.10)

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
        top *= 10
        right *= 10
        bottom *= 10
        left *= 10

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
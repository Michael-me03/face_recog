import face_recognition
import dlib 
from PIL import Image, ImageDraw, ImageFont

# === PART 1: Load and analyze PNG_1.jpg ===

# Load target image
image = face_recognition.load_image_file("./50_cent.jpg")
face_locations = face_recognition.face_locations(image)
face_landmarks_list = face_recognition.face_landmarks(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

# Print face location and landmarks
print("Face locations in PNG_1.jpg:")
print(face_locations)

print("\nFace landmarks in PNG_1.jpg:")
for i, landmarks in enumerate(face_landmarks_list):
    print(f"Face {i + 1}:")
    for feature, points in landmarks.items():
        print(f"  {feature}: {points}")

# Load known face
known_image = face_recognition.load_image_file("./obama.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Prepare image for drawing
pil_image = Image.fromarray(image)
draw = ImageDraw.Draw(pil_image)

# Optional: Load a font
try:
    font = ImageFont.truetype("arial.ttf", 18)
except IOError:
    font = ImageFont.load_default()

# === PART 2: Draw boxes, landmarks, and label matches ===

for (top, right, bottom, left), encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
    # Compare with known face (Obama)
    match = face_recognition.compare_faces([known_encoding], encoding)[0]

    # Draw red box
    draw.rectangle([left, top, right, bottom], outline="red", width=4)

    # Draw facial landmarks
    for feature_points in landmarks.values():
        draw.line(feature_points, fill="blue", width=2)

    if match:
        name = "Obama"
        bbox = font.getbbox(name)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle([left, bottom, left + text_width + 10, bottom + text_height + 10], fill="red")
        draw.text((left + 5, bottom + 5), name, fill="white", font=font)

# Show result
pil_image.show()
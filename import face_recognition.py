import capture_image

import stripe
stripe.api_key = "API_KEY_HERE"  # Ersetze durch deinen Stripe API-Schlüssel

def charge_recognized_user():
    try:
        payment_intent = stripe.PaymentIntent.create(
            amount=100,  # 1 Dollar = 100 Cent
            currency='eur',
            payment_method_types=['card'], # Nur Kartenzahlung
        )
        print("💳 Stripe PaymentIntent erzeugt:", payment_intent['id'])
    except Exception as e:
        print("❌ Stripe-Fehler:", e)

import face_recognition
import dlib 
from PIL import Image, ImageDraw, ImageFont
import pygame  # Für MP3-Wiedergabe

# Analyse des gespeicherten Bildes ("img2.jpg")
image = face_recognition.load_image_file("./captured_images/img2.jpg")
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

known_image = face_recognition.load_image_file("./captured_images/img1.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

pil_image = Image.fromarray(image)
draw = ImageDraw.Draw(pil_image)

try:
    font = ImageFont.truetype("arial.ttf", 36)
except IOError:
    font = ImageFont.load_default()

# Part 2

for (top, right, bottom, left), encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
    match = face_recognition.compare_faces([known_encoding], encoding)[0]

    draw.rectangle([left, top, right, bottom], outline="red", width=4)

    for feature_points in landmarks.values():
        draw.line(feature_points, fill="blue", width=2)

    if match:
        name = "Linus"
        bbox = font.getbbox(name)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle([left, bottom, left + text_width + 20, bottom + text_height + 20], fill="red")
        draw.text((left + 10, bottom + 10), name, fill="white", font=font)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load("success.mp3")
        pygame.mixer.music.play()

        charge_recognized_user()

pil_image.show()
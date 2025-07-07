import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk
import face_recognition
import pygame
import stripe
import os

# Stripe API-Key Setup
stripe.api_key = "API_KEY_HERE"  # Replace with your actual Stripe API key

def charge_recognized_user(amount_eur):
    try:
        amount_cents = int(float(amount_eur) * 100)
        payment_intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency='eur',
            payment_method_types=['card'],  # Only card payment
        )
        print("💳 Stripe PaymentIntent created:", payment_intent['id'])
        messagebox.showinfo("Success", f"PaymentIntent created: {payment_intent['id']}")
    except Exception as e:
        print("❌ Stripe error:", e)
        messagebox.showerror("Stripe Error", str(e))


def recognize_and_charge():
    euro_amount = euro_entry.get()
    if not euro_amount:
        messagebox.showwarning("Missing Input", "Please enter a Euro amount.")
        return

    try:
        # Load images
        known_image = face_recognition.load_image_file("./captured_images/img1.jpg")
        known_encoding = face_recognition.face_encodings(known_image)[0]

        image = face_recognition.load_image_file("./captured_images/img2.jpg")
        face_locations = face_recognition.face_locations(image)
        face_landmarks_list = face_recognition.face_landmarks(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except IOError:
            font = ImageFont.load_default()

        matched = False
        for (top, right, bottom, left), encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
            match = face_recognition.compare_faces([known_encoding], encoding)[0]

            draw.rectangle([left, top, right, bottom], outline="red", width=4)
            for feature_points in landmarks.values():
                draw.line(feature_points, fill="blue", width=2)

            if match:
                matched = True
                name = "Linus"
                bbox = font.getbbox(name)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.rectangle([left, bottom, left + text_width + 20, bottom + text_height + 20], fill="red")
                draw.text((left + 10, bottom + 10), name, fill="white", font=font)

                # Play sound
                pygame.init()
                pygame.mixer.init()
                pygame.mixer.music.load("success.mp3")
                pygame.mixer.music.play()

                charge_recognized_user(euro_amount)

        if not matched:
            messagebox.showinfo("Result", "No recognized face found.")

        # Show result in GUI
        img_tk = ImageTk.PhotoImage(pil_image.resize((400, 300)))
        image_label.config(image=img_tk)
        image_label.image = img_tk

    except Exception as e:
        print("⚠️ Error:", e)
        messagebox.showerror("Error", str(e))


# GUI Setup
root = tk.Tk()
root.title("Face Recognition & Stripe Charge")
root.geometry("500x500")

tk.Label(root, text="Enter amount in EUR:").pack(pady=10)
euro_entry = tk.Entry(root)
euro_entry.pack()

tk.Button(root, text="Start Face Recognition and Charge", command=recognize_and_charge).pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

root.mainloop()
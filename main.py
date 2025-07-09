from PIL import Image, ImageDraw, ImageFont
import face_recognition
import pygame
import stripe
import os
import sys

# Stripe API-Key Setup
stripe.api_key = "API_KEY_HERE"  # Replace with your actual Stripe API key

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

def recognize_and_charge(euro_amount):
    try:
        print("🔍 Starting face recognition...")
        
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
                
                print("✅ Face recognized as:", name)
                
                # Play sound
                try:
                    pygame.init()
                    pygame.mixer.init()
                    pygame.mixer.music.load("Money.mp3")
                    pygame.mixer.music.play()
                    print("🔊 Success sound played")
                except Exception as sound_error:
                    print(f"⚠️ Could not play sound: {sound_error}")
                
                # Process payment
                success, result = charge_recognized_user(euro_amount)
                if success:
                    print(f"💰 Payment successful! Payment ID: {result}")
                else:
                    print(f"❌ Payment failed: {result}")
        
        if not matched:
            print("❌ No recognized face found.")
        
        # Save result image
        result_path = "./captured_images/result.jpg"
        pil_image.save(result_path)
        print(f"📸 Result image saved to: {result_path}")
        
        return matched
        
    except Exception as e:
        print("⚠️ Error:", e)
        return False

def capture_image():
    print("📸 Capturing image...")
    os.system('python capture_image.py')
    print("✅ Image capture completed")

def main():
    print("🎭 Face Recognition & Stripe Charge System")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Capture new image")
        print("2. Process face recognition and charge")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            capture_image()
        
        elif choice == '2':
            # Check if required images exist
            if not os.path.exists("./captured_images/img1.jpg"):
                print("❌ Reference image (img1.jpg) not found!")
                continue
            
            if not os.path.exists("./captured_images/img2.jpg"):
                print("❌ Test image (img2.jpg) not found! Please capture an image first.")
                continue
            
            try:
                euro_amount = float(input("Enter amount in EUR: "))
                if euro_amount <= 0:
                    print("❌ Please enter a valid positive amount.")
                    continue
                
                recognize_and_charge(euro_amount)
                
            except ValueError:
                print("❌ Please enter a valid number.")
        
        elif choice == '3':
            print("👋 Goodbye!")
            sys.exit(0)
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
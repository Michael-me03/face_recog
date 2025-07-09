import cv2
import os
import time

# Ordner vorbereiten
output_dir = "captured_images"
os.makedirs(output_dir, exist_ok=True)

# Kamera öffnen
vc = cv2.VideoCapture(0)

if not vc.isOpened():
    print("❌ Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

print("🎥 Live-Stream läuft. Drücke [s], um ein Bild zu speichern, oder [ESC], um zu beenden.")

while True:
    time.sleep(1)  # Kurze Pause, um die CPU-Auslastung zu reduzieren
    ret, frame = vc.read()
    if not ret:
        print("❌ Fehler beim Lesen des Kamerabilds.")
        break

    # Helligkeit erhöhen
    brighter = cv2.convertScaleAbs(frame, alpha=1.0, beta=1)

    # Zeige das helle Bild live an
    cv2.imshow("Live Preview – Drücke [s] zum Speichern", brighter)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        img_path = os.path.join(output_dir, "img2.jpg")
        cv2.imwrite(img_path, brighter)
        print(f"✅ Bild gespeichert: {img_path}")
        break
    elif key == 27:  # ESC
        print("⏹️ Abgebrochen.")
        break

vc.release()
cv2.destroyAllWindows()
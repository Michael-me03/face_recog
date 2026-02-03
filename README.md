# 🌙 Moon Technologies | Face Pay

**Seamless biometric payments. Part of the Moon Technologies autonomous retail ecosystem.**

## 👁️ About
**Face Pay** is a Computer Vision payment terminal that authenticates users via facial recognition and processes transactions instantly through Stripe. It eliminates the need for cards or phones, offering a truly frictionless checkout experience.

## 🚀 Features
*   **Biometric Auth**: Real-time face detection & verification.
*   **Instant Checkout**: Integration with Stripe for automated billing.
*   **Interactive UI**: Visual bounding boxes and audio feedback.

##  Installation
```bash
git clone https://github.com/Michael-me03/face_recog.git
cd face_recog
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
*(Requires CMake for `face_recognition`: `brew install cmake` or `apt-get install cmake`)*

## ⚙️ Setup
1.  **Stripe**: detailed in `main.py` (`stripe.api_key`).
2.  **Faces**: Add reference images to `./captured_images/` (e.g., `user.jpg`).

## 🖥 Usage
```bash
python main.py
```

---
*Powered by OpenCV, Face_Recognition, and Stripe.*

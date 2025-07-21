# face_recog
This project is being implemented as a student project to explore the use of facial recognition in contactless payments. 

For this purpose, an application is being developed that will automatically process contactless payments using facial recognition. A transaction will then be automatically created and later posted via Stripe. 

For facial recognition, another GitHub repository will be accessed, which is used under the following license.

# face_recog


# Face Recognition Payment System

Ein Raspberry Pi basiertes Bezahlsystem mit Gesichtserkennung und Stripe Integration.

## Setup

### 1. Repository klonen
git clone https://github.com/Michael-me03/face_recog
cd face_recog

### 2. Python Virtual Environment erstellen
python3 -m venv face_payment_env
source face_payment_env/bin/activate  # Linux/Mac
# oder
face_payment_env\Scripts\activate     # Windows

### 3. Dependencies installieren
# Erst System-Dependencies 
# (Ubuntu/Raspberry Pi)
sudo apt-get update
sudo apt-get install cmake libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev

# Mac
brew install cmake

# Python-Pakete aus requirements.txt installieren
pip install --upgrade pip
pip install -r requirements.txt

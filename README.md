# Face Recognition Payment System (CV Project)

A Computer Vision based payment system that utilizes face recognition to identify users and process payments via Stripe. This project integrates OpenCV, face_recognition, and the Stripe API to create a seamless "Face Pay" experience.

## 🚀 Features

*   **Real-time Face Recognition**: Detects and identifies faces from a live webcam feed.
*   **Stripe Integration**: Automatically processes payments for recognized users.
*   **Audio/Visual Feedback**: Draws bounding boxes around faces and plays success sounds upon transaction completion.
*   **CLI Menu**: Simple command-line interface to control the flow (Capture, Charge, Exit).

## 🛠 Prerequisites

*   Python 3.8+
*   A Stripe Account (for API keys)
*   Webcam

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Michael-me03/face_recog.git
    cd face_recog
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # Mac/Linux
    source venv/bin/activate
    # Windows
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install CMake and other system dependencies for `face_recognition` / `dlib`.*
    *   **Mac**: `brew install cmake`
    *   **Linux**: `sudo apt-get install cmake libopenblas-dev liblapack-dev`

## ⚙️ Configuration

1.  **Stripe API Key**:
    Open `main.py` and `capture_image.py` and replace the placeholder API key with your actual Stripe Secret Key:
    ```python
    stripe.api_key = "sk_test_..."
    ```

2.  **Reference Images**:
    Create a folder named `captured_images` in the root directory and add images of the known users. You will need to update the file paths in `capture_image.py` to point to your specific reference images (e.g., `img1.jpg`, `Linus.jpg`).

    Structure:
    ```
    face_recog/
    ├── captured_images/
    │   ├── img1.jpg      # Reference for main.py
    │   ├── img2.jpg      # Test/Captured image for main.py
    │   ├── Linus.jpg     # User 1
    │   └── Pascal.jpg    # User 2
    ├── main.py
    ├── capture_image.py
    └── ...
    ```

## 🖥 Usage

Run the main application:

```bash
python main.py
```

You will be presented with a menu:

1.  **Capture new image**: Launches the live webcam face recognition module (`capture_image.py`). Press `q` to quit the webcam view.
2.  **Process face recognition and charge**: Performs a static check comparing `./captured_images/img1.jpg` (known) vs `img2.jpg` (captured) and processes a Stripe charge if they match.
3.  **Exit**: Closes the application.

## 📁 File Structure

*   `main.py`: The main entry point and CLI menu. Handles specific file-based comparison and payment logic.
*   `capture_image.py`: Real-time webcam script using OpenCV to detect and label faces.
*   `requirements.txt`: Python dependencies.

## 🔗 Acknowledgements

*   Built with [face_recognition](https://github.com/ageitgey/face_recognition) and [Stripe](https://stripe.com).

# 🐾 Food Calorie Estimator

This app predicts the type of food in an uploaded image using a YOLO object detection model, then estimates its calorie content using Gemini (Google Generative AI). Madu, the pixel-perfect cat mascot, brings nutritional insights with cuteness and charm 🐱✨

## 🔍 Features
- 🍔 YOLO-based food detection from image uploads
- 🧠 Gemini-powered calorie estimation and food chatbot
- 📦 Flask web interface with multiple routes
- 📁 Secure image upload and intelligent calorie calculation
- 😺 Chat interface styled around a virtual nutritionist kitten

## 🧠 How It Works
1. YOLO detects food in the uploaded image
2. Gemini estimates calories based on food label and weight
3. Results displayed with playful and informative messaging

## 🛠️ Tech Stack
- Python
- Flask
- Ultralytics YOLO
- Gemini AI via `google-generativeai`
- OpenCV (optional for image processing)
- Dotenv for secure environment management

## 🚀 Setup Instructions

### 1. Clone the repository
    git clone https://github.com/L3tzG0/food_classification_app.git
    cd food_classification_app


### 2. Install dependencies
    pip install -r requirements.txt

### 3. Configure environment variables
    Create a .env file with your Gemini or OPENAI API key:
    API_KEY=your_api_key_here

### 4. Run the app
    python app.py

## 📁 File Structure
    ├── app.py
    ├── CNN_Model.h5
    ├── classes.npy
    ├── food_detection_model.pt
    ├── static/
    │   └── uploads/
    │   └── images/
    ├── templates/
    │   └── index.html
    │   └── estimate.html
    │   └── chatbot.html
    │   └── aboutpage.html
    │   └── estimatedescription.html
    │   └── estimateshowcalorie.html
    ├── requirements.txt
    ├── .env (not tracked)
    └── README.md

## 📝 Notes
- Ensure model files and data paths are correctly placed
- Create an empty "uploads" folder inside the static folder to store images/uploads by user
- Project was made using python 3.10.11

## 👥 Credits
This project was developed by a group of four students as part of a collaborative effort.
Special thanks to all team members who contributed across model development, frontend design, and logic integration.
Flask integration, backend architecture, and CNN model creation and training were led by Dhruv.

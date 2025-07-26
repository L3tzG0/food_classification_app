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
```bash
git clone https://github.com/L3tzG0/food_classification_app.git
cd food_classification_app

### 2. Install dependencies
```bash
pip install -r requirements.txt

### 3. Configure environment variables
Create a .env file with your Gemini API key:
GEMINI_API_KEY=your_api_key_here

### 4. Run the app
```bash
python app.py
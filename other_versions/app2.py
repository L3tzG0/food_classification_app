# Main App
# Food Classification Model = CNN
# Chatbot Model = GPT3.5

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import openai
import os
import re
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("CNN_Model.h5")
class_names = np.load("classes.npy", allow_pickle=True).item()
idx_to_class = {v: k for k, v in class_names.items()}

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Check the file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/estimate')
def estimate():
    return render_template('estimate.html')

@app.route('/aboutpage')
def aboutpage():
    return render_template('aboutpage.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot2.html')

# Upload the images
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        flash("No file part in the request.")
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        flash("No file selected.")
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('estimatedescription', image_filename=filename))

    flash("Invalid file type. Only PNG, JPG, and JPEG files are allowed.")
    return redirect(url_for('index'))

# Show page description with images
@app.route('/estimatedescription')
def estimatedescription():
    image_filename = request.args.get('image_filename')
    return render_template('estimatedescription.html', image_filename=image_filename)

# Calculate the calorie estimation
@app.route('/calculate_calorie', methods=['POST'])
def calculate_calorie():
    image_filename = request.form['image_filename']
    weight = request.form.get('weight')
    weight_value = int(weight) if weight and weight.isdigit() else 0

    # Prepare image for CNN model
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    img = preprocess_image(image_path)  # should return shape (1, H, W, C)
    preds = model.predict(img)
    class_idx = np.argmax(preds)
    label = idx_to_class[class_idx]  # food item name

    detections = [label]  # keep this for template

    # Generate prompt based on weight
    if weight_value == 0:
        prompt = (
            f"You are a certified nutritionist. Estimate the kilocalories for the food item '{label}' "
            f"based on the following:\n"
            f"- Calorie estimate for 1 piece\n"
            f"- Calorie per 100 grams\n\n"
            f"Output ONLY in this exact format:\n"
            f"{label} = [kcal_piece] kcal per piece\n"
            f"{label} = [kcal_100g] kcal per 100g\n"
        )
    else:
        prompt = (
            f"You are a certified nutritionist. Estimate the kilocalories for the food item '{label}' "
            f"weighing {weight_value} grams.\n"
            f"Output ONLY in this exact format:\n"
            f"{label} = [kcal_total] kcal for {weight_value} grams\n"
        )
    # OpenAI request
    client = openai.OpenAI(api_key=os.getenv("API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a nutritionist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    # Parse response
    answer = response.choices[0].message.content
    numbers = re.findall(r'\d+', answer)
    calorie_answer1 = numbers[0] if len(numbers) > 0 else "N/A"
    calorie_answer2 = numbers[1] if len(numbers) > 1 else None

    # Formatting for template
    if weight_value == 0:
        calorie_estimate2 = calorie_answer2
        unit1 = "per piece"
        unit2 = "per 100 grams"
    else:
        calorie_estimate2 = None
        unit1 = f"Cal for {weight_value} grams"
        unit2 = None

    return render_template(
        'estimateshowcalorie.html',
        image_filename=image_filename,
        detections=detections,
        calorie_estimate1=calorie_answer1,
        unit1=unit1,
        calorie_estimate2=calorie_estimate2,
        unit2=unit2
    )

@app.route('/chatbot_api', methods=['POST'])
def chatbot_api():
    user_input = request.json.get('user_input')
    message_history = request.json.get('message_history', [])

    client = openai.OpenAI(api_key=os.getenv("API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message_history + [{"role": "user", "content": user_input}],
        temperature=0.7
    )

    return {'reply': response.choices[0].message.content.strip()}


# Run the apps
if __name__ == '__main__':
    app.run(debug=True)

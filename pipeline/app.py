from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import random


# adding monitoring imports
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_metric import (
    PREDICTION_COUNT,
    CORRECT_PREDICTION_COUNT,
    DATA_DRIFT_COUNT,
    PREDICTION_LATENCY,
    ACTIVE_USERS
)
import time

# Initialize Flask app
app = Flask(__name__)

# Load trained EMNIST model
# changed the path for docker 
# model_path = "//mnt//c//Users//aimte//OneDrive//Desktop//uni work//mlops//project_mlops//notebooks//models//emnist_federated_model.h5"
model_path = "models/emnist_federated_model.h5"

model = load_model(model_path,compile=False)
print(f"Loaded model from {model_path}")

# EMNIST stats for drift detection
EMNIST_MEAN = 0.1307
EMNIST_STD = 0.3081
DRIFT_THRESHOLD = 0.2  # pixel distribution drift threshold

# ===============================
# Preprocessing function
# ===============================
def preprocess_canvas_image(image):
    """
    Convert canvas image to 28x28 EMNIST-like input
    """
    # Convert to grayscale
    image = image.convert('L')
    # Invert to match EMNIST (white digit on black)
    # image = ImageOps.invert(image)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Crop bounding box of non-black pixels
    coords = np.column_stack(np.where(img_array > 10))  # threshold to remove black background
    if coords.size == 0:  # empty drawing
        img_array = np.zeros((28,28))
    else:
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)
        img_array = img_array[x0:x1+1, y0:y1+1]

        # Resize while keeping aspect ratio
        h, w = img_array.shape
        if h > w:
            new_h = 20
            new_w = int(round((w / h) * 20))
        else:
            new_w = 20
            new_h = int(round((h / w) * 20))
        img_array = Image.fromarray(img_array).resize((new_w,new_h), Image.LANCZOS)

        # Pad to 28x28 to center the digit
        new_img = np.zeros((28,28), dtype=np.uint8)
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        new_img[top:top+new_h, left:left+new_w] = np.array(img_array)
        img_array = new_img

    # Normalize to 0-1
    img_array = img_array.astype("float32") / 255.0

    # Flatten to match old model input
    return img_array.reshape(1, 784)

# ===============================
# Home page
# ===============================

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/")
def home():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Kids Digit Learning App</title>
<style>
    body { 
        font-family: 'Comic Sans MS', cursive, sans-serif; 
        background: linear-gradient(to bottom, #a8edea, #fed6e3); 
        text-align: center; 
        padding-top: 50px; 
    }
    .container { 
        width: 600px; 
        margin: auto; 
        background: #fff3b0; 
        padding: 30px; 
        border-radius: 20px; 
        box-shadow: 0px 10px 25px rgba(0,0,0,0.2);
        border: 4px dashed #ff6f61;
    }
    h1 { 
        color: #ff6f61; 
        font-size: 36px;
        margin-bottom: 20px;
    }
    h2 {
        color: #ff6f61;
        font-size: 24px;
        margin-bottom: 20px;
    }
    canvas { 
        border: 3px solid #ff6f61; 
        background-color: black; 
        border-radius: 15px; 
        cursor: crosshair; 
        box-shadow: 0px 5px 15px rgba(0,0,0,0.3);
    }
    button { 
        padding: 15px 25px; 
        margin: 15px; 
        background: #ffcc5c; 
        color: #fff; 
        font-weight: bold;
        font-size: 18px;
        border: none; 
        border-radius: 12px; 
        cursor: pointer; 
        box-shadow: 0px 5px 10px rgba(0,0,0,0.2);
        transition: transform 0.1s ease-in-out;
    }
    button:hover { 
        background: #ff6f61; 
        transform: scale(1.1);
    }
    #result { 
        font-size: 26px; 
        margin-top: 20px; 
        color: #ff6f61; 
        font-weight: bold;
    }
    .footer {
        margin-top: 25px;
        font-size: 16px;
        color: #555;
    }
</style>
</head>
<body>
<div class="container">
    <h1>Kids Digit Learning App</h1>
    <h2>Draw the digit: <span id="targetDigit"></span></h2>
    <canvas id="canvas" width="280" height="280"></canvas><br>
    <button onclick="clearCanvas()">🧹 Clear Canvas</button>
    <button onclick="predictDigit()">🔮 Check </button>
    <h2 id="result"></h2>
    <div class="footer">Learn to draw numbers !</div>
</div>

<script>
var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
var targetDigit = Math.floor(Math.random() * 10);
document.getElementById("targetDigit").innerText = targetDigit;

// Black background
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Draw white ink
var drawing = false;
canvas.addEventListener("mousedown", e => { drawing = true; ctx.beginPath(); ctx.moveTo(e.offsetX, e.offsetY); });
canvas.addEventListener("mousemove", e => {
    if(drawing){
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.strokeStyle = "white";
        ctx.lineWidth = 20;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.stroke();
    }
});
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseleave", () => drawing = false);

// Clear canvas
function clearCanvas(){
    ctx.fillStyle = "black";
    ctx.fillRect(0,0,canvas.width,canvas.height);
    document.getElementById("result").innerText = "";
}

// Send to backend
function predictDigit(){
    var dataURL = canvas.toDataURL("image/png");
    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/x-www-form-urlencoded"},
        body: "image=" + encodeURIComponent(dataURL) + "&target_digit=" + targetDigit
    }).then(r => r.json())
      .then(d => {
          if(d.error){
              document.getElementById("result").innerText = "Error: " + d.error;
          } else {
              document.getElementById("result").innerText = d.message;
              // Generate new target digit
              targetDigit = Math.floor(Math.random() * 10);
              document.getElementById("targetDigit").innerText = targetDigit;
          }
      });
}
</script>
</body>
</html>
""")

# ===============================
# Predict endpoint
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    PREDICTION_COUNT.inc()
    ACTIVE_USERS.inc() 
    try:
        data_url = request.form['image']
        target_digit = int(request.form['target_digit'])
        _, encoded = data_url.split(",", 1)

        # Load image
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))

        # Preprocess to EMNIST format
        img_flat = preprocess_canvas_image(image)

        # Compute drift
        mean_diff = abs(img_flat.mean() - EMNIST_MEAN)
        std_diff = abs(img_flat.std() - EMNIST_STD)
        drift_detected = (mean_diff > DRIFT_THRESHOLD) or (std_diff > DRIFT_THRESHOLD)
        if drift_detected:
            DATA_DRIFT_COUNT.inc()
        # Predict
        raw_pred = int(np.argmax(model.predict(img_flat), axis=1)[0])
        predicted_digit = raw_pred

        if predicted_digit == target_digit:
            message = "Well done!"
            CORRECT_PREDICTION_COUNT.inc()   # correct prediction
        else:
            message = "Try again!"

        # Record latency
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        ACTIVE_USERS.dec()

        return jsonify({
            "message": message,
            "prediction": int(predicted_digit),
            "target": int(target_digit),
            "data_drift": bool(drift_detected),
            "mean_diff": float(mean_diff),
            "std_diff": float(std_diff),
            "latency":latency
        })

    except Exception as e:
        # Record latency even for errors
        PREDICTION_LATENCY.observe(time.time() - start_time)
        PREDICTION_LATENCY.observe(latency)
        ACTIVE_USERS.dec()
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, render_template, request, jsonify
import openai
import cv2
import numpy as np
import math
from ultralytics import YOLO
import signal
import sys

app = Flask(__name__)
app.detected_classes = []

# Set up OpenAI API credentials
openai.api_key = 'SecretKEY'

# Create a flag to stop webcam capture
stop_webcam_capture = False

# Define a signal handler to stop the webcam capture when the application is terminated
def signal_handler(sig, frame):
    global stop_webcam_capture
    stop_webcam_capture = True
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Define the default route to return the index.html file
@app.route("/")
def index():
    return render_template("index.html")

# Define a function to continuously detect and display classes
def detect_objects_and_return_list():
    class_names_list = []
    frame_counter = 0
    print("in loop")

    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Model
    model = YOLO("yolo-Weights/yolov8n.pt")

    # Object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        frame_class_names = []

        # Coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Class name
                cls = int(box.cls[0])
                class_name = classNames[cls]
                frame_class_names.append(class_name)

        app.detected_classes.append(frame_class_names)
        print(len(app.detected_classes))

        # Check if the capture should be stopped
        if stop_webcam_capture:
            break

    cap.release()
    cv2.destroyAllWindows()

    return app.detected_classes

# Define a function to generate a response using ChatGPT with a prompt
def generate_response_with_classes(message):
    # Check if there are detected classes
    if app.detected_classes:
        # Extract the user's input message
        user_input = message.strip()

        # Use the detected classes to enhance the user's input
        classes_str = ", ".join(app.detected_classes[-1])
        enhanced_prompt = f"{user_input} based on classes: {classes_str}"

        # Send the enhanced prompt to ChatGPT
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": enhanced_prompt}
            ]
        )

        if completion.choices[0].message != None:
            response = completion.choices[0].message
        else:
            response = 'Failed to generate response!'
    else:
        response = 'No classes detected yet.'

    return response
@app.route("/opencam", methods=['POST'])
def opencam():
    # This route will return detected class names to the client
    return jsonify(detect_objects_and_return_list())

# Define the /api route to handle POST requests
@app.route("/api", methods=["POST"])
def api():
    # Get the message from the POST request
    data = request.get_json()
    message = data.get("message")

    # Generate a response using ChatGPT with the prompt and detected classes
    response = generate_response_with_classes(message)

    # Return the response as JSON
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run()

import cv2
import threading
import time
import requests
from flask import Flask, request, jsonify
from ultralytics import YOLO

# Global variables to manage threads
threads = []
stop_event = threading.Event()

# Load YOLO model (YOLOv8 is used here; modify the model path if using another version)
model = YOLO('../runs/best/weights/best.pt')  # Replace with your YOLO model path

app = Flask(__name__)

# Function to handle RTSP feed and YOLO predictions
def process_rtsp_feed(rtsp_url, server_url):
    cap = cv2.VideoCapture(rtsp_url)
    frame_rate = 20
    prev = 0
    while not stop_event.is_set():
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if frame is None:
            print(f"end of source {rtsp_url}")
            break
        if time_elapsed < 1./frame_rate:
            continue
        prev = time.time()
        # Do something with your image here.
        # process_image()
        if not ret:
            print(f"Failed to grab frame from {rtsp_url}")
            break

        # YOLO prediction
        results = model.predict(source=frame, verbose=False)  # YOLO model prediction

        # Check if any object is detected
        if results and len(results[0].boxes) > 0:
            # Send only when something is detected
            detected_objects = []
            for box in results[0].boxes:
                cls_name = model.names[int(box.cls)]  # Get class name
                # confidence = box.conf   Confidence score
                # x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
                
                # Append detection info to list
                # detected_objects.append({
                #     "class": cls_name,
                #     "confidence": float(confidence),
                #     "bbox": [float(x1), float(y1), float(x2), float(y2)]
                # })
                detected_objects.append({
                    "detected": cls_name,})

            # Send detected objects to the server
            requests.post(server_url, json={
                "deviceId": rtsp_url,
                "content": str(detected_objects)
            })


        # Optional: Display the frame (for testing purposes)
        # cv2.imshow(f"RTSP feed: {rtsp_url}", frame)

        # Press 'q' to quit the window (optional testing functionality)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to start RTSP streams with YOLO and send predictions to server
def start_rtsp_streams(rtsp_urls, server_url):
    global threads, stop_event
    stop_event.clear()
    threads = []

    for rtsp_url in rtsp_urls:
        thread = threading.Thread(target=process_rtsp_feed, args=(rtsp_url, server_url))
        thread.start()
        threads.append(thread)

    return "RTSP feeds started with YOLO and sending detections to server"

# Function to stop RTSP streams
def stop_rtsp_streams():
    global threads, stop_event
    stop_event.set()  # Trigger stop event

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    return "RTSP feeds stopped"

# Route to start RTSP streams and send YOLO detections to server
@app.route('/start', methods=['POST'])
def start():
    data = request.json
    rtsp_urls = data.get('rtsp_urls', [])
    server_url = data.get('server_url', "")
    if not rtsp_urls or not server_url:
        return jsonify({"error": "No RTSP URLs or server URL provided"}), 400

    # Start the RTSP feeds and YOLO detections
    return jsonify({"message": start_rtsp_streams(rtsp_urls, server_url)}), 200

# Route to stop RTSP streams
@app.route('/stop', methods=['POST'])
def stop():
    return jsonify({"message": stop_rtsp_streams()}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

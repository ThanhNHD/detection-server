import cv2
import threading
import time
import requests
from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np

# Global variables to manage threads
threads = []
stop_event = threading.Event()

# Load YOLO model (YOLOv8 is used here; modify the model path if using another version)
model = YOLO('../runs/best/weights/best.pt')  # Replace with your YOLO model path

app = Flask(__name__)

# Auto-configure CLAHE parameters based on the image
def enhance_contrast_auto(gray_image):
    # Get image dimensions
    height, width = gray_image.shape

    # Dynamically set the CLAHE parameters
    # Clip limit adapts based on image intensity distribution
    clip_limit = max(
        2.0, min(4.0, gray_image.std() / 20)
    )  # Adjusted based on image contrast
    # Tile grid size adapts based on image size
    grid_size = (
        (8, 8) if max(height, width) <= 1000 else (16, 16)
    )  # Larger for high-res

    # Create CLAHE with dynamic parameters
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # Apply CLAHE to enhance contrast
    enhanced_image = clahe.apply(gray_image)
    return enhanced_image


def reduce_noise_auto(image):
    """
    Applies a Bilateral Filter with adaptive parameters based on the input grayscale image.

    :param image: Grayscale image (numpy array).
    :return: Filtered image.
    """
    # Calculate noise level (standard deviation)
    noise_std = np.std(image)

    # Image dimensions
    height, width = image.shape
    image_size = max(height, width)

    # Adaptive parameters
    d = max(5, image_size // 100)  # Neighborhood diameter, scaled by image size
    sigma_color = noise_std * 10  # Adjust intensity influence based on noise level
    sigma_space = d * 1.5  # Spatial influence, linked to diameter

    # Apply Bilateral Filter
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    return filtered_image


def auto_edge_enhance(gray_image):
    # Calculate the image contrast (standard deviation of pixel values)
    contrast = gray_image.std()

    # Set a base clip limit and tile grid size
    clip_limit = 2.0  # default
    tile_grid_size = (8, 8)  # default size

    # Adjust clip limit based on image contrast (higher contrast = higher clip limit)
    if contrast > 50:
        clip_limit = 3.0
    elif contrast > 30:
        clip_limit = 2.5
    else:
        clip_limit = 2.0

    # Adjust tile grid size based on image resolution (smaller images = smaller tiles)
    height, width = gray_image.shape
    if height < 500 or width < 500:
        tile_grid_size = (4, 4)
    elif height < 1000 or width < 1000:
        tile_grid_size = (8, 8)
    else:
        tile_grid_size = (16, 16)

    # Apply CLAHE with the auto-configured parameters
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(gray_image)

    return clahe_img


def dehaze_auto(gray_image):
    # Calculate dynamic CLAHE parameters based on image properties
    mean_brightness = gray_image.mean()
    contrast_std = gray_image.std()

    # Dynamically configure CLAHE
    clip_limit = max(
        2.0, min(4.0, contrast_std / 10)
    )  # Adjust clip limit based on contrast
    grid_size = (8, 8) if gray_image.shape[0] <= 1000 else (16, 16)  # Adjust grid size

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced_image = clahe.apply(gray_image)

    # Normalize intensity if brightness is too low
    if mean_brightness < 100:  # If the image is very dark
        normalized_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized_image
    else:
        return enhanced_image


def adaptive_unsharp_masking(gray_image):
    """
    Automatically sharpens a grayscale image using unsharp masking with dynamic adjustments
    based on image size, brightness, and contrast.

    Args:
        gray_image (numpy.ndarray): Input grayscale image (loaded with cv2).

    Returns:
        sharpened_image (numpy.ndarray): The sharpened grayscale image.
    """
    # Get image dimensions
    height, width = gray_image.shape

    # Dynamically adjust blur kernel size based on image dimensions
    # Larger images get larger kernels
    kernel_size = (max(3, width // 500), max(3, height // 500))

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, kernel_size, 0)

    # Calculate the mean and standard deviation of the image
    mean_intensity = np.mean(gray_image)
    std_intensity = np.std(gray_image)

    # Dynamically set alpha and beta
    # Higher contrast (std_intensity) reduces alpha (to avoid over-sharpening)
    # Lower brightness increases beta (to enhance edges more aggressively)
    alpha = 1.0 + (std_intensity / 128)  # Base sharpness control
    beta = -0.5 - (mean_intensity / 255)  # Edge control based on brightness

    # Combine the original image and the blurred image for sharpening
    sharpened_image = cv2.addWeighted(gray_image, alpha, blurred_image, beta, 0)

    return sharpened_image
# Function to handle RTSP feed and YOLO predictions
def process_rtsp_feed(source_place, server_url):
    cap = cv2.VideoCapture(source_place, cv2.IMREAD_GRAYSCALE)
    frame_rate = 20
    prev = 0
    while not stop_event.is_set():
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if not ret:
            print(f"end of source {source_place}")
            break
        if time_elapsed < 1./frame_rate:
            continue
        prev = time.time()
        # Do something with your image here.
        # process_image()
        if not ret:
            print(f"Failed to grab frame from {source_place}")
            break

        # YOLO prediction
        resized_image = cv2.resize(frame, (640, 640))
        denoised_image = reduce_noise_auto(resized_image)
        enhance_contract_image = enhance_contrast_auto(denoised_image)
        unsharp_masking_image = adaptive_unsharp_masking(enhance_contract_image)

        input_image = cv2.merge([unsharp_masking_image, unsharp_masking_image, unsharp_masking_image])

        results = model.predict(source=input_image, verbose=False)  # YOLO model prediction

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
                "sourcePlace": source_place,
                "content": str(detected_objects)
            })


        # Optional: Display the frame (for testing purposes)
        # cv2.imshow(f"RTSP feed: {rtsp_url}", frame)

        # Press 'q' to quit the window (optional testing functionality)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

# Function to start RTSP streams with YOLO and send predictions to server
def start_rtsp_streams(source_place, server_url):
    global threads, stop_event
    stop_event.clear()
    threads = []

    for source_place in source_place:
        thread = threading.Thread(target=process_rtsp_feed, args=(source_place, server_url))
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
    source_place = data.get('sourcePlace', [])
    server_url = data.get('serverMessageUrl', "")
    if not source_place or not server_url:
        return jsonify({"error": "No RTSP URLs or server URL provided"}), 400

    # Start the RTSP feeds and YOLO detections
    return jsonify({"message": start_rtsp_streams(source_place, server_url)}), 200

# Route to stop RTSP streams
@app.route('/stop', methods=['POST'])
def stop():
    return jsonify({"message": stop_rtsp_streams()}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

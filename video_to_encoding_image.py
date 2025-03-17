import face_recognition
import numpy as np
import dlib
import cv2
import os
import shutil
import tempfile

# Paths to pre-trained models
SHAPE_PREDICTOR_PATH = "statics/shape_predictor_68_face_landmarks.dat"
HAAR_CASCADE_PATH = "statics/haarcascade_frontalface_default.xml"


def extract_frames_from_video(input_video_file):
    """
    Extract frames from a video file and save them as images in a temporary directory.

    Args:
        input_video_file (str): Path to the video file.

    Returns:
        str: Path to the temporary directory containing extracted frames.
    """
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory for frames

    cap = cv2.VideoCapture(input_video_file)

    frame_rate = 1  # Extract one frame per second
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    frame_interval = fps * frame_rate

    frame_count = 0  
    saved_count = 0  

    while True:
        ret, frame = cap.read()
        if not ret:
            break  

        if frame_count % frame_interval == 0:  
            frame_filename = os.path.join(temp_dir, f"extracted_frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    return temp_dir


def process_user_images(input_path):
    """
    Process input images to detect faces and save them in a temporary directory.

    Args:
        input_path (str): Path to the input image or directory containing images.

    Returns:
        str: Path to the temporary directory containing processed faces.
    """
    output_dir = tempfile.mkdtemp()  # Create a temporary directory for processed faces

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Load dlib predictor for facial landmarks
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    def process_image(img_path):
        """
        Process an individual image to detect and extract face, and save it.

        Args:
            img_path (str): Path to the input image.
        """
        img = cv2.imread(img_path)
        if img is None:
            return

        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Process the first detected face
            x, y, w, h = faces[0]

            face = img[y:y+h, x:x+w]

            detected_landmarks = predictor(img, dlib.rectangle(x, y, x+w, y+h))

            target_size = (224, 224)
            resized_face = cv2.resize(face, target_size)

            normalized_face = resized_face / 255.0

            face_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_face.jpg")
            cv2.imwrite(face_filename, normalized_face * 255)  

    if os.path.isfile(input_path):
        process_image(input_path)
    else:
        for img_file in os.listdir(input_path):
            process_image(os.path.join(input_path, img_file))

    return output_dir


def encode_images_to_vector(input_path):
    """
    Encode faces in images from the given path into facial feature vectors.

    Args:
        input_path (str): Path to the input image or directory containing images.

    Returns:
        np.array: The average face encoding vector of all the processed faces.
    """
    if os.path.isdir(input_path):
        image_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith(('.jpg', '.png', '.jpeg'))]
    else:
        image_files = [input_path]

    image_encodings = []  # List to store face encodings

    for image_path in image_files:
        image = face_recognition.load_image_file(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(image, model="cnn")
        encodings = face_recognition.face_encodings(image, face_locations, model="cnn")

        if len(encodings) > 0:  # Add the first encoding if a face is found
            image_encodings.append(encodings[0])

    if len(image_encodings) == 0:
        print("No face encodings found in the input.")

    # Calculate the mean encoding
    image_mean_encoding = np.mean(image_encodings, axis=0)

    return image_mean_encoding


def video_to_encoding_image(input_video_file):
    """
    Convert a video file to a single face encoding by processing the frames.

    Args:
        input_video_file (str): Path to the input video file.

    Returns:
        np.array: The face encoding vector derived from the video.
    """
    frames_dir = None
    faces_dir = None

    try:
        frames_dir = extract_frames_from_video(input_video_file)
        faces_dir = process_user_images(frames_dir)
        output = encode_images_to_vector(faces_dir)
        return output
    
    finally:
        if frames_dir:
            shutil.rmtree(frames_dir, ignore_errors=True)
        if faces_dir:
            shutil.rmtree(faces_dir, ignore_errors=True)

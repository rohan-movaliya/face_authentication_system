from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from video_to_encoding_image import video_to_encoding_image
from verify_face_from_image import verify_face_from_image

app = FastAPI()

@app.post("/video-to-encoding")
async def register_video(video: UploadFile = File(...)):  
    """
    Endpoint to extract face encodings from a video file. The video file is uploaded and processed to
    extract face encodings. If no face encodings are found, a message is returned. Temporary files 
    are cleaned up after processing.

    Args:
    - video: The video file to be processed (max size: 50 MB).

    Returns:
    - A JSON response containing the extracted face encodings.
    - A message if no face encodings are found.
    """
    try:
        upload_dir = "uploads"
        
        os.makedirs(upload_dir, exist_ok=True)
        
        video_path = f"{upload_dir}/{video.filename}"
        with open(video_path, "wb") as f:
            f.write(video.file.read())

        if not os.path.isfile(video_path):
            raise HTTPException(status_code=404, detail="File not found.")

        face_encodings = video_to_encoding_image(video_path)
        if face_encodings is None or face_encodings.size == 0:
            return {"detail": "No face encodings found in the input."}
        
        os.remove(video_path)

        remaining_files = os.listdir(upload_dir)
        if not remaining_files:
            os.rmdir(upload_dir)
        return {"face_encodings": face_encodings.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify-face")
async def verify_face_endpoint(
    encodings_file: UploadFile = File(...),
    auth_image: UploadFile = File(...),
):
    """
    Endpoint to verify if a face in an authentication image matches any face encodings provided in 
    a file. Both the encodings file and authentication image are uploaded. If a match is found, 
    it returns a confirmation; otherwise, it returns a mismatch result. Temporary files are cleaned up 
    after processing.

    Args:
    - encodings_file: The file containing previously extracted face encodings.
    - auth_image: The authentication image for face verification.

    Returns:
    - A JSON response indicating whether the authentication image matches any face encodings.
    """
    try:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        pickle_file_path = f"{upload_dir}/{encodings_file.filename}"

        image_path = f"{upload_dir}/{auth_image.filename}"
        with open(image_path, "wb") as f:
            f.write(auth_image.file.read())

        if not os.path.isfile(image_path):
            raise HTTPException(status_code=404, detail="Authentication image file not found.")
        
        with open(pickle_file_path, "wb") as f:
            f.write(encodings_file.file.read())

        if not os.path.isfile(pickle_file_path):
            raise HTTPException(status_code=404, detail="Encodings file not found.")

        is_match = verify_face_from_image(pickle_file_path, image_path)
        
        os.remove(pickle_file_path)
        os.remove(image_path)

        remaining_files = os.listdir(upload_dir)
        if not remaining_files:
            os.rmdir(upload_dir)

        return {"is_match": is_match}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import face_recognition
import numpy as np
import pickle
from video_to_encoding_image import process_user_images, encode_images_to_vector


def load_database_encoding(pickle_file_path):
    """
    Load the facial encoding data from a pickle file.

    Args:
        pickle_file_path (str): The path to the pickle file containing the facial encoding data.

    Returns:
        np.array: The facial encodings loaded from the pickle file.
    """
    with open(pickle_file_path, "rb") as pickle_file:
        encoding = pickle.load(pickle_file)

    return np.array(encoding)


def process_and_encode_auth_image(image_path):
    """
    Process and encode the authentication image to extract facial encodings.

    Args:
        image_path (str): The path to the authentication image.

    Returns:
        np.array: A NumPy array containing the facial encodings extracted from the image.

    Raises:
        ValueError: If no faces are detected in the authentication image.
    """
    processed_faces_dir = process_user_images(image_path)
    output_encodings = encode_images_to_vector(processed_faces_dir)
    encodings = np.array(output_encodings)

    if len(encodings) == 0:
        raise ValueError("No face found in the authentication image.")

    return encodings


def verify_face(database_encoding, auth_encoding, tolerance=0.45):
    """
    Verify if the authentication image encoding matches the database encoding.

    Args:
        database_encoding (np.array): The encoding of the database image.
        auth_encoding (np.array): The encoding of the authentication image.
        tolerance (float): The tolerance level for the comparison (default is 0.45).

    Returns:
        bool: True if the faces match, False otherwise.
    """
    matches = face_recognition.compare_faces([database_encoding], auth_encoding, tolerance=tolerance)
    return matches[0]


def verify_face_from_image(pickle_file_path, image_path):
    """
    Verify if the face in the authentication image matches the database encoding.

    Args:
        pickle_file_path (str): The path to the pickle file containing the database facial encoding.
        image_path (str): The path to the authentication image.

    Returns:
        bool: True if the faces match, False otherwise.
    """
    database_encoding = load_database_encoding(pickle_file_path)
    auth_encoding = process_and_encode_auth_image(image_path)

    is_match = verify_face(database_encoding, auth_encoding)

    if is_match:
        return True
    else:
        return False

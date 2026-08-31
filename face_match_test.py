import cv2
import dlib
import numpy as np


# --------------------------------------------------
# 1. Load dlib models
# --------------------------------------------------

face_detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

face_recognizer = dlib.face_recognition_model_v1(
    "dlib_face_recognition_resnet_model_v1.dat"
)


# --------------------------------------------------
# 2. Convert an image into a face embedding
# --------------------------------------------------

def get_face_embedding(image):
    """
    Finds a face in an image and converts it
    into a 128-dimensional face embedding.
    """

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = face_detector(rgb_image, 1)

    if len(faces) == 0:
        return None

    # For now, use the first detected face
    face = faces[0]

    landmarks = shape_predictor(rgb_image, face)

    embedding = face_recognizer.compute_face_descriptor(
        rgb_image,
        landmarks
    )

    return np.array(embedding)


# --------------------------------------------------
# 3. Extract one frame from the KYC video
# --------------------------------------------------

video_path = r"video_kyc_test1.mp4"

video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print("Could not open video.")
    exit()

total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

middle_frame_number = total_frames // 2

video_capture.set(
    cv2.CAP_PROP_POS_FRAMES,
    middle_frame_number
)

success, frame = video_capture.read()

video_capture.release()

if not success:
    print("Could not read video frame.")
    exit()


# --------------------------------------------------
# 4. Read the ID image
# --------------------------------------------------

id_image = cv2.imread("dummy_id.png")

if id_image is None:
    print("Could not open dummy_id.png.")
    exit()


# --------------------------------------------------
# 5. Generate embeddings
# --------------------------------------------------

id_embedding = get_face_embedding(id_image)

video_embedding = get_face_embedding(frame)


if id_embedding is None:
    print("No face detected in ID image.")
    exit()

if video_embedding is None:
    print("No face detected in video frame.")
    exit()


# --------------------------------------------------
# 6. Calculate face distance
# --------------------------------------------------

distance = np.linalg.norm(
    id_embedding - video_embedding
)


# --------------------------------------------------
# 7. Decide match
# --------------------------------------------------

MATCH_THRESHOLD = 0.60

match = distance < MATCH_THRESHOLD


# --------------------------------------------------
# 8. Display result
# --------------------------------------------------

print("\n========== FACE MATCH RESULT ==========")

print(f"Face distance: {distance:.4f}")
print(f"Threshold: {MATCH_THRESHOLD}")
print(f"Match: {match}")

print("======================================")
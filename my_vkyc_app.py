import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import dlib
import cv2
from scipy.spatial import distance as dist
import tempfile
import os
import io
import numpy as np
import easyocr
import re
from pypdf import PdfReader 
from openai import OpenAI
from PIL import Image
import time
import logging

# ============================================================
# SET UP THE LOGGER
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================
# CACHE OCR MODEL
# ============================================================

@st.cache_resource
def load_ocr_model():

    return easyocr.Reader(
        ["en"],
        gpu=False
    )



# ============================================================
# CACHE DLIB MODELS
# ============================================================

@st.cache_resource
def load_dlib_models():

    face_detector = dlib.get_frontal_face_detector()

    shape_predictor = dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat"
    )

    face_recognizer = dlib.face_recognition_model_v1(
        "dlib_face_recognition_resnet_model_v1.dat"
    )

    return (
        face_detector,
        shape_predictor,
        face_recognizer
    )


# ============================================================
# LOAD CACHED MODELS
# ============================================================

ocr_reader = load_ocr_model()

(
    face_detector,
    shape_predictor,
    face_recognizer
) = load_dlib_models()


# ============================================================
# AWS CONFIGURATION AND OPENAi client
# ============================================================

aws_bucket_name = "video-kyc-ai"
aws_region = "ap-south-1"

openai_client = OpenAI()

# ============================================================
# UPLOAD FILE TO S3
# ============================================================

def file_upload_to_s3(file, bucket_name, file_name):

    s3_client = boto3.client(
        "s3",
        region_name=aws_region
    )

    try:

        s3_client.upload_fileobj(
            file,
            bucket_name,
            file_name
        )

        return file_name

    except FileNotFoundError:

        st.error(
            f"File {file_name} not found."
        )

    except NoCredentialsError:

        st.error(
            "AWS credentials not available."
        )

    except PartialCredentialsError:

        st.error(
            "AWS credentials are incomplete."
        )

    except Exception as e:

        st.error(
            f"S3 upload error: {e}"
        )

    return None


# ============================================================
# GET FACE EMBEDDING
# ============================================================

def get_face_embedding(image):

    """
    Detect exactly one face and return
    its 128-dimensional dlib face embedding.
    """

    if image is None:
        return None

    # OpenCV uses BGR.
    # Convert to RGB for dlib.
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Detect faces.
    faces = face_detector(
        rgb_image,
        1
    )

    # No face.
    if len(faces) == 0:
        return None

    # More than one face.
    if len(faces) > 1:
        return None

    # Exactly one face.
    face = faces[0]

    # Find 68 facial landmarks.
    landmarks = shape_predictor(
        rgb_image,
        face
    )

    # Generate 128-dimensional embedding.
    embedding = face_recognizer.compute_face_descriptor(
        rgb_image,
        landmarks
    )

    return np.array(embedding)


# ============================================================
# OCR - EXTRACT ID INFORMATION
# ============================================================

def extract_id_information(image_path):

    # Run EasyOCR.
    text_lines = ocr_reader.readtext(
        image_path,
        detail=0
    )

    # Convert all lines into one text block.
    text = "\n".join(text_lines)

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    name_match = re.search(
        r"Name\s*:\s*\n?\s*([A-Za-z ]+)",
        text,
        re.IGNORECASE
    )

    if name_match:

        name = name_match.group(1).strip()

    else:

        name = "Not detected"

    # --------------------------------------------------------
    # DOB
    # --------------------------------------------------------

    dob_match = re.search(
        r"DOB\s*:\s*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})",
        text,
        re.IGNORECASE
    )

    if dob_match:

        dob = dob_match.group(1).strip()

    else:

        dob = "Not detected"

    # --------------------------------------------------------
    # ID NUMBER
    # --------------------------------------------------------

    id_match = re.search(
        r"ID\s*:\s*([A-Za-z0-9]+)",
        text,
        re.IGNORECASE
    )

    if id_match:

        id_number = id_match.group(1).strip()

    else:

        id_number = "Not detected"

    # --------------------------------------------------------
    # RETURN STRUCTURED INFORMATION
    # --------------------------------------------------------

    return {
        "name": name,
        "dob": dob,
        "id_number": id_number
    }


# ============================================================
# COMPARE FACE EMBEDDINGS
# ============================================================

def compare_faces(
    embedding1,
    embedding2
):

    # We cannot compare missing embeddings.
    if embedding1 is None or embedding2 is None:

        return None, None

    # Calculate Euclidean distance.
    face_distance = np.linalg.norm(
        embedding1 - embedding2
    )

    # Initial dlib reference threshold.
    threshold = 0.50

    # Determine whether it is below the threshold.
    face_match = face_distance < threshold

    return face_distance, face_match


# ============================================================
# GET A FACE EMBEDDING FROM VIDEO
# ============================================================

def get_video_face_embedding(video_path):

    video_capture = cv2.VideoCapture(
        video_path
    )

    if not video_capture.isOpened():

        return None

    while True:

        ret, frame = video_capture.read()

        if not ret:

            break

        # Try to find a usable single face.
        embedding = get_face_embedding(
            frame
        )

        if embedding is not None:

            video_capture.release()

            return embedding

    video_capture.release()

    return None


# ============================================================
# LIVENESS DETECTION --> by dlib
# ============================================================

def detect_blinks_and_lip_movement(video_path):

    # --------------------------------------------------------
    # EAR calculation
    # --------------------------------------------------------

    def calculate_ear(eye):

        A = dist.euclidean(
            eye[1],
            eye[5]
        )

        B = dist.euclidean(
            eye[2],
            eye[4]
        )

        C = dist.euclidean(
            eye[0],
            eye[3]
        )

        if C == 0:

            return 0

        return (A + B) / (2.0 * C)

    # --------------------------------------------------------
    # Landmark indexes
    # --------------------------------------------------------

    LEFT_EYE_IDX = slice(36, 42)

    RIGHT_EYE_IDX = slice(42, 48)

    LIPS_IDX = slice(48, 68)

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    EAR_THRESHOLD = 0.25

    BLINK_FRAMES = 3

    LIP_MOVEMENT_THRESHOLD = 0.20

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    blink_count = 0

    consecutive_blink_frames = 0

    lip_movement_count = 0

    previous_lip_movement_ratio = None

    face_detected = False

    multiple_faces_detected = False

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    video_capture = cv2.VideoCapture(
        video_path
    )

    if not video_capture.isOpened():

        return {
            "blink_count": 0,
            "lip_movement_count": 0,
            "face_detected": False,
            "multiple_faces_detected": False,
            "liveness_status": "REVIEW"
        }

    # --------------------------------------------------------
    # Process video frame by frame
    # --------------------------------------------------------

    while True:

        ret, frame = video_capture.read()

        if not ret:

            break

        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector(
            gray_frame
        )

        # No face in this frame.
        if len(faces) == 0:

            continue

        face_detected = True

        # Multiple faces are an important KYC condition.
        if len(faces) > 1:

            multiple_faces_detected = True

            continue

        face = faces[0]

        # ----------------------------------------------------
        # Get 68 landmarks
        # ----------------------------------------------------

        landmarks = shape_predictor(
            gray_frame,
            face
        )

        landmarks = [
            (point.x, point.y)
            for point in landmarks.parts()
        ]

        left_eye = landmarks[
            LEFT_EYE_IDX
        ]

        right_eye = landmarks[
            RIGHT_EYE_IDX
        ]

        lips = landmarks[
            LIPS_IDX
        ]

        # ----------------------------------------------------
        # BLINK DETECTION
        # ----------------------------------------------------

        left_ear = calculate_ear(
            left_eye
        )

        right_ear = calculate_ear(
            right_eye
        )

        if (
            left_ear < EAR_THRESHOLD
            and right_ear < EAR_THRESHOLD
        ):

            consecutive_blink_frames += 1

        else:

            if (
                consecutive_blink_frames
                >= BLINK_FRAMES
            ):

                blink_count += 1

            consecutive_blink_frames = 0

        # ----------------------------------------------------
        # LIP MOVEMENT
        # ----------------------------------------------------

        lip_height = dist.euclidean(
            lips[14],
            lips[18]
        )

        lip_width = dist.euclidean(
            lips[12],
            lips[16]
        )

        if lip_width == 0:

            continue

        lip_movement_ratio = (
            lip_height / lip_width
        )

        if previous_lip_movement_ratio is not None:

            movement_difference = abs(
                lip_movement_ratio
                - previous_lip_movement_ratio
            )

            if (
                movement_difference
                > LIP_MOVEMENT_THRESHOLD
            ):

                lip_movement_count += 1

        previous_lip_movement_ratio = (
            lip_movement_ratio
        )

    # --------------------------------------------------------
    # Finish video
    # --------------------------------------------------------

    video_capture.release()

    # --------------------------------------------------------
    # Liveness status
    # --------------------------------------------------------

    if multiple_faces_detected:

        liveness_status = "FAIL"

    elif not face_detected:

        liveness_status = "FAIL"

    elif (
        blink_count > 0
        and lip_movement_count > 0
    ):

        liveness_status = "PASS"

    else:

        # IMPORTANT:
        # No movement does NOT mean fake.
        # We simply don't have strong liveness evidence.
        liveness_status = "REVIEW"

    # --------------------------------------------------------
    # Return evidence
    # --------------------------------------------------------

    return {
        "blink_count": blink_count,
        "lip_movement_count": lip_movement_count,
        "face_detected": face_detected,
        "multiple_faces_detected": multiple_faces_detected,
        "liveness_status": liveness_status
    }


# ============================================================
# LOAD KYC POLICY
# ============================================================

@st.cache_data
def load_kyc_policy():

    pdf_path = "final_kyc_verification_policy.pdf"

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text
            text += "\n"


    # ========================================================
    # POLICY SECTION HEADINGS
    # ========================================================

    section_headings = [
        "1. Purpose",
        "2. ID Information Requirements",
        "3. Face Verification",
        "4. Liveness Verification",
        "5. Overall Decision Rules",
        "6. Outcome Definitions",
        "7. Example Evidence",
        "8. Scope of This Policy"
    ]


    # ========================================================
    # CREATE SECTION-BASED CHUNKS
    # ========================================================

    collection_chunks = []

    current_chunk = ""

    for line in text.splitlines():

        line = line.strip()

        if not line:

            continue

        if any(
            line.startswith(heading)
            for heading in section_headings
        ):

            if current_chunk:

                collection_chunks.append(
                    current_chunk.strip()
                )

            current_chunk = line

        else:

            current_chunk += "\n" + line


    if current_chunk:

        collection_chunks.append(
            current_chunk.strip()
        )


    return collection_chunks

# ============================================================
# CREATE POLICY EMBEDDINGS
# ============================================================

@st.cache_data
def create_policy_embeddings(collection_chunks):

    embedding_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=collection_chunks
    )

    chunk_embeddings = [
        item.embedding
        for item in embedding_response.data
    ]

    return chunk_embeddings



# ============================================================
# RETRIEVE RELEVANT KYC POLICY
# ============================================================

def retrieve_kyc_policy(query, collection_chunks):

    # --------------------------------------------------------
    # CREATE EMBEDDINGS FOR POLICY CHUNKS
    # --------------------------------------------------------
    chunk_embeddings = create_policy_embeddings(
    collection_chunks
)


    # --------------------------------------------------------
    # CREATE EMBEDDING FOR THE QUERY
    # --------------------------------------------------------

    query_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    query_embedding = (
        query_response.data[0].embedding
    )


    # --------------------------------------------------------
    # CONVERT QUERY INTO WORDS
    # --------------------------------------------------------

    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9.]+\b",
            query.lower()
        )
    )


    # --------------------------------------------------------
    # CALCULATE HYBRID RETRIEVAL SCORE
    # --------------------------------------------------------

    retrieval_results = []

    for i, chunk_embedding in enumerate(
        chunk_embeddings
    ):

        # ----------------------------------------------------
        # SEMANTIC SIMILARITY
        # ----------------------------------------------------

        semantic_similarity = np.dot(
            query_embedding,
            chunk_embedding
        ) / (
            np.linalg.norm(query_embedding)
            * np.linalg.norm(chunk_embedding)
        )


        # ----------------------------------------------------
        # KEYWORD MATCHING
        # ----------------------------------------------------

        chunk_words = set(
            re.findall(
                r"\b[a-zA-Z0-9.]+\b",
                collection_chunks[i].lower()
            )
        )

        matched_words = (
            query_words.intersection(
                chunk_words
            )
        )


        if len(query_words) > 0:

            keyword_score = (
                len(matched_words)
                / len(query_words)
            )

        else:

            keyword_score = 0


        # ----------------------------------------------------
        # COMBINE BOTH SCORES
        # ----------------------------------------------------

        combined_score = (
            0.80 * semantic_similarity
            + 0.20 * keyword_score
        )


        retrieval_results.append(
            (
                i,
                semantic_similarity,
                keyword_score,
                combined_score
            )
        )


    # --------------------------------------------------------
    # SORT BY COMBINED SCORE
    # --------------------------------------------------------

    retrieval_results.sort(
        key=lambda x: x[3],
        reverse=True
    )


    # --------------------------------------------------------
    # TAKE TOP 3
    # --------------------------------------------------------

    top_chunks = retrieval_results[:3]


    # --------------------------------------------------------
    # COMBINE RETRIEVED POLICY TEXT
    # --------------------------------------------------------

    retrieved_policy = "\n\n".join(
        collection_chunks[result[0]]
        for result in top_chunks
    )


    return retrieved_policy, top_chunks


# ============================================================
# GENERATE KYC ASSESSMENT USING LLM
# ============================================================

def generate_kyc_assessment(
    evidence,
    retrieved_policy
):

    # --------------------------------------------------------
    # CREATE KYC EVIDENCE TEXT
    # --------------------------------------------------------

    kyc_evidence = f"""
Name: {evidence["id_information"]["name"]}
DOB: {evidence["id_information"]["dob"]}
ID Number: {evidence["id_information"]["id_number"]}

Face distance: {evidence["face_distance"]}

Face match: {evidence["face_match"]}

Blink count: {evidence["liveness"]["blink_count"]}
Lip movement count: {evidence["liveness"]["lip_movement_count"]}

Face detected in video: {
    evidence["liveness"]["face_detected"]
}

Multiple faces detected in video: {
    evidence["liveness"]["multiple_faces_detected"]
}

Liveness status: {
    evidence["liveness"]["liveness_status"]
}
"""


    # --------------------------------------------------------
    # CREATE LLM PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an AI assistant supporting a bank's KYC verification team.

Evaluate the provided KYC evidence strictly according to the
retrieved KYC policy.

IMPORTANT INSTRUCTIONS:

1. Use only the KYC evidence and retrieved policy provided below.
2. Do not invent facts, rules, thresholds, or verification results.
3. Do not change or reinterpret the policy rules.
4. If the policy says PASS, return PASS.
5. If the policy says REVIEW, return REVIEW.
6. If the policy says FAIL, return FAIL.
7. Do not treat REVIEW as FAIL.
8. Do not treat FAIL as REVIEW or PASS.
9. Clearly distinguish between a confirmed failure and an
   inconclusive case.
10. The explanation must be concise, professional, and suitable
    for a bank verification team.
11. Do not mention embeddings, vectors, chunks, cosine similarity,
    model names, or other internal technical details.
12. Base the reason directly on the supplied evidence and policy.
13. Do not make assumptions about information that is not provided.
14. Do not introduce specific verification procedures, corrective
    actions, or requirements unless they are supported by the
    provided KYC policy.
15. When multiple conditions are present, select FAIL if any
    conclusive FAIL condition applies. REVIEW conditions must not
    override a conclusive FAIL condition.

KYC EVIDENCE:
{kyc_evidence}

RETRIEVED KYC POLICY:
{retrieved_policy}

Return the result exactly in this format:

Decision: PASS / REVIEW / FAIL

Reason:
Give a concise explanation of why this decision follows from
the KYC evidence and the retrieved policy.

Recommended Action:
State the appropriate next action for the verification team.
"""


    # --------------------------------------------------------
    # SEND REQUEST TO OPENAI
    # --------------------------------------------------------

    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # --------------------------------------------------------
    # RETURN LLM RESPONSE
    # --------------------------------------------------------

    return response.choices[0].message.content


def validate_photo_id(image_path: str) -> bool:
    """Validate that the file exists, is non-empty, and is a valid image."""

    if not isinstance(image_path, str) or not os.path.isfile(image_path):
        return False

    if os.path.getsize(image_path) == 0:
        return False

    try:
        # Check that the image structure is valid
        with Image.open(image_path) as image:
            image.verify()

        # Re-open and fully load the image
        # to catch truncated/corrupted image data
        with Image.open(image_path) as image:
            image.load()

        return True

    except Exception:
        return False


def validate_video(video_path: str) -> bool:
    """Validate that the file exists, is non-empty, opens, and contains readable frames."""
    if not isinstance(video_path, str) or not os.path.isfile(video_path):
        return False

    if os.path.getsize(video_path) == 0:
        return False

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        # Read the first frame to ensure the stream/codec actually works
        ret, frame = cap.read()
        return bool(ret and frame is not None and frame.size > 0)
    except Exception:
        return False
    finally:
        if cap is not None:
            cap.release()

# def validate_photo_id_face(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         return False

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_detector(gray)

#     return len(faces) == 1



# ============================================================
# STREAMLIT UI
# ============================================================

st.title(
    "AI-Assisted KYC Verification System"
)

st.header(
    "KYC Case"
)


# ============================================================
# VIDEO UPLOAD
# ============================================================

st.subheader(
    "Upload KYC Video"
)

video_file = st.file_uploader(
    "Choose your KYC video",
    type=["mp4", "mov", "avi"]
)


# ============================================================
# PHOTO ID UPLOAD
# ============================================================

st.subheader(
    "Upload Photo ID"
)

photo_id_file = st.file_uploader(
    "Choose your Photo ID",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# SUBMIT
# ============================================================

if st.button("Submit"):
    total_start = time.perf_counter()
    logger.info("KYC verification started")

    if (
        video_file is not None
        and photo_id_file is not None
    ):

        # ====================================================
        # READ FILES INTO MEMORY
        # ====================================================

        photo_data = photo_id_file.read()

        video_data = video_file.read()


        # ====================================================
        # UPLOAD PHOTO ID TO S3
        # ====================================================

        s3_start = time.perf_counter()
        photo_name = (
            f"photo_{photo_id_file.name}"
        )

        uploaded_photo = file_upload_to_s3(
            io.BytesIO(photo_data),
            aws_bucket_name,
            photo_name
        )


        # ====================================================
        # UPLOAD VIDEO TO S3
        # ====================================================

        video_name = (
            f"video_{video_file.name}"
        )

        uploaded_video = file_upload_to_s3(
            io.BytesIO(video_data),
            aws_bucket_name,
            video_name
        )

        s3_end = time.perf_counter()
        s3_time = s3_end - s3_start

        # ====================================================
        # S3 RESULT
        # ====================================================

        if (
            uploaded_photo
            and uploaded_video
        ):
            logger.info(
                "Photo ID and video successfully uploaded to S3."

            )

            st.success(
                "Photo ID and video successfully "
                "uploaded to S3."
            )
        else:
            logger.error("S3 upload failed for one or more KYC files")


        # ====================================================
        # CREATE TEMPORARY PHOTO
        # ====================================================

        photo_extension = os.path.splitext(
            photo_id_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=photo_extension
        ) as temp_photo:

            temp_photo.write(
                photo_data
            )

            temp_photo_path = (
                temp_photo.name
            )


        # ====================================================
        # CREATE TEMPORARY VIDEO
        # ====================================================

        video_extension = os.path.splitext(
            video_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=video_extension
        ) as temp_video:

            temp_video.write(
                video_data
            )

            temp_video_path = (
                temp_video.name
            )

        # ====================================================
        # INPUT VALIDATION
        # ====================================================

        def cleanup_and_stop():
            for temp_file in (temp_photo_path, temp_video_path):
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except OSError:
                    pass
            st.stop()

        photo_valid = validate_photo_id(temp_photo_path)
        video_valid = validate_video(temp_video_path) 

        if photo_valid and video_valid:
            logger.info("Photo ID and video passed input validation")

        # 1. File Integrity Check
        if not photo_valid or not video_valid:
            logger.error("KYC input validation failed")
            if not photo_valid and not video_valid:
                st.error("Both the uploaded Photo ID and video are invalid or corrupted.")
            elif not photo_valid:
                st.error("The uploaded Photo ID is invalid or corrupted.")
            else:
                st.error("The uploaded video is invalid or corrupted.")
            cleanup_and_stop()

        # # 2. Face Count Check (runs only if photo file is valid)
        # photo_face_valid = validate_photo_id_face(temp_photo_path)
        # if not photo_face_valid:
        #     st.error("The Photo ID must contain exactly one detectable face.")
        #     cleanup_and_stop()


        # ====================================================
        # OCR
        # ====================================================

        st.subheader(
            "ID Information"
        )

        ocr_start = time.perf_counter()
        id_information = extract_id_information(
            temp_photo_path
        )
        ocr_end = time.perf_counter()
        ocr_time = ocr_end - ocr_start

        logger.info(f"OCR completed successfully in {ocr_time:.3f} seconds.")

        st.write(
            f"Name: {id_information['name']}"
        )

        st.write(
            f"DOB: {id_information['dob']}"
        )

        st.write(
            f"ID Number: {id_information['id_number']}"
        )


        # ====================================================
        # FACE VERIFICATION
        # ====================================================

        st.subheader(
            "Face Verification"
        )

        # Read ID image.
        id_image = cv2.imread(
            temp_photo_path
        )

        face_start = time.perf_counter()
        # Generate ID face embedding.
        id_embedding = get_face_embedding(
            id_image
        )

        # Generate video face embedding.
        video_embedding = get_video_face_embedding(
            temp_video_path
        )

        # Default values.
        face_distance = None

        face_match = None


        # ----------------------------------------------------
        # Face matching result
        # ----------------------------------------------------

        if id_embedding is None:
            logger.warning(
                "Face verification could not proceed: no single face detected in Photo ID"
            )

            st.warning(
                "No single face detected "
                "in the Photo ID."
            )

        elif video_embedding is None:
            logger.warning(
                "Face verification could not proceed: no single face detected in KYC video"
            )

            st.warning(
                "No single face detected "
                "in the KYC video."
            )

        else:
            face_distance, face_match = compare_faces(
                id_embedding,
                video_embedding
            )



            st.write(
                f"Face distance: "
                f"{face_distance:.4f}"
            )

            # st.write(
            #     "Reference threshold: 0.60"
            # )

            if face_match:
                logger.info("Face verification completed: match passed")

                st.success(
                    "Face Match: PASS"
                )

            else:
                logger.warning("Face verification completed: match requires review")

                st.warning(
                    "Face Match: REVIEW"
                )
        face_end = time.perf_counter()
        face_time = face_end - face_start

        logger.info(
            "Face verification stage completed in %.3f seconds",
            face_time
        )



        # ====================================================
        # LIVENESS
        # ====================================================

        st.subheader(
            "Liveness Verification"
        )

        liveness_start = time.perf_counter()
        liveness_result = (
            detect_blinks_and_lip_movement(
                temp_video_path
            )
        )
        liveness_end = time.perf_counter()
        liveness_time = liveness_end - liveness_start

        logger.info(
            "Liveness verification completed with %s status in %.3f seconds.",
            liveness_result["liveness_status"],
            liveness_time
        )

        st.write(
            f"Total blinks detected: "
            f"{liveness_result['blink_count']}"
        )

        st.write(
            f"Total lip movements detected: "
            f"{liveness_result['lip_movement_count']}"
        )

        if liveness_result["liveness_status"] == "PASS":

            st.success(
                "Liveness status: PASS"
            )

        elif liveness_result["liveness_status"] == "REVIEW":

            st.warning(
                "Liveness status: REVIEW"
            )

        else:

            st.error(
                "Liveness status: FAIL"
            )


        # ====================================================
        # FINAL KYC EVIDENCE
        # ====================================================

        st.subheader(
            "KYC Evidence"
        )

        evidence = {

            "id_information": id_information,

            "face_distance": (
                round(face_distance, 4)
                if face_distance is not None
                else None
            ),

            "face_match": (
                bool(face_match)
                if face_match is not None
                else None
            ),

            "liveness": liveness_result
        }

        st.json(
            evidence
        )

        # ============================================================
        # AI KYC ASSESSMENT
        # ============================================================

        st.subheader(
            "Assessment Result"
        )

        # Load the KYC policy
        policy_chunks = load_kyc_policy()


        # ============================================================
        # CREATE QUERY FROM ACTUAL KYC EVIDENCE
        # ============================================================

        query = f"""
        KYC verification evidence:

        ID information:
        Name: {evidence["id_information"]["name"]}
        DOB: {evidence["id_information"]["dob"]}
        ID Number: {evidence["id_information"]["id_number"]}

        Face distance: {evidence["face_distance"]}
        Face match: {evidence["face_match"]}

        Blink count: {evidence["liveness"]["blink_count"]}
        Lip movement count: {evidence["liveness"]["lip_movement_count"]}

        Face detected in video:
        {evidence["liveness"]["face_detected"]}

        Multiple faces detected in video:
        {evidence["liveness"]["multiple_faces_detected"]}

        Liveness status:
        {evidence["liveness"]["liveness_status"]}
        """


        # ============================================================
        # RETRIEVE RELEVANT POLICY
        # ============================================================

        rag_start = time.perf_counter()
        retrieved_policy, top_chunks = retrieve_kyc_policy(
            query,
            policy_chunks
        )
        rag_end = time.perf_counter()
        rag_time = rag_end - rag_start
        logger.info(
        "KYC policy retrieval completed in %.3f seconds",
        rag_time
    )


        # ============================================================
        # GENERATE AI ASSESSMENT
        # ============================================================

        llm_start = time.perf_counter()
        assessment = generate_kyc_assessment(
            evidence,
            retrieved_policy
        )

        llm_end = time.perf_counter()
        llm_time = llm_end - llm_start
        logger.info(
        "LLM KYC assessment completed in %.3f seconds",
        llm_time
    )

        total_end = time.perf_counter()
        total_time = total_end - total_start
        logger.info(
        "KYC verification completed in %.3f seconds",
        total_time
    )


        # ============================================================
        # DISPLAY AI ASSESSMENT
        # ============================================================

        st.write(
            assessment
        )


        # ====================================================
        # DELETE TEMPORARY FILES
        # ====================================================

        os.remove(
            temp_photo_path
        )

        os.remove(
            temp_video_path
        )

    else:
        logger.warning("KYC submission attempted without both required files"
        )

        st.warning(
            "Please upload both video and Photo ID."
        )
from imutils.face_utils import FaceAligner
import imutils
import dlib
import cv2

SHAPE_PREDICTOR_PATH = "Utils/Dlib_Files/shape_predictor_68_face_landmarks.dat"


def face_align(image, face_box, face_width=64):
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=face_width)
    aligned_face = face_aligner.align(image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), face_box)
    return aligned_face
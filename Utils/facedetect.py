
import cv2
import dlib
import Utils.alignment as alignment
import numpy


CASCADE_PATH = "Utils/OpenCV_Files/haarcascade_frontalface_alt.xml"
DLIB_CNN_FACE_DETECTOR_PATH = "Utils/Dlib_Files/mmod_human_face_detector.dat"

def detectFace(frame, face_detector="CV", aligned=False):
	if face_detector == "CV":
		if (len(frame.shape) == 3):
			img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		cascade = cv2.CascadeClassifier(CASCADE_PATH)
		rects = cascade.detectMultiScale(image=img, scaleFactor=1.3, minNeighbors=4, flags=1, minSize=(20, 20))

		if len(rects) == 0:
			face = None
			face_img = None
		else:
			rects[:, 2:] += rects[:, :2]
			x1, y1, x2, y2 = rects[0]
			face = rects[0]
			if aligned:
				bbox_rect = dlib.rectangle(left=int(x1), top=int(y1), right=int(x2), bottom=int(y2))

				face_img = alignment.face_align(image=frame, face_box=bbox_rect, face_width=128)
			else:
				face_img = frame[y1:y2, x1:x2]
		return face, face_img
	elif face_detector == "dlib_HOG":
		detector = dlib.get_frontal_face_detector()
		detected_faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)
		if len(detected_faces) > 0:
			for i, d in enumerate(detected_faces):
				face = [d.left(), d.top(), d.right(), d.bottom()]
				if aligned:
					face_img = alignment.face_align(image=frame, face_box=d, face_width=128)
				else:
					face_img = frame[face[1]:face[3], face[0]:face[2]]
				break
		else:
			face = None
			face_img = None
	else:
		detector = dlib.cnn_face_detection_model_v1(DLIB_CNN_FACE_DETECTOR_PATH)
		detected_faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)
		if len(detected_faces) > 0:
			for i, d in enumerate(detected_faces):
				face = [d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()]
				if aligned:
					face_img = alignment.face_align(image=frame, face_box=d.rect, face_width=128)
				else:
					face_img = frame[face[1]:face[3], face[0]:face[2]]
				break
		else:
			face = None
			face_img = None
	return face, face_img

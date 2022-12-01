import numpy
import cv2
import os
from Utils.GUI import GUIController
from Utils import facedetect, preprocess

CAMERA_ID = 0
FACE_DETECTION_FREQUENCY = 60


def getCameraCapture(CAMERA_ID=0):
	cap = cv2.VideoCapture(CAMERA_ID)
	if not cap.isOpened():
		print("Cannot open camera")
		exit(0)
	else:
		return cap


if __name__ == "__main__":
	capture_device = getCameraCapture(CAMERA_ID)
	count = 0
	face = None
	while True:
		# Capture frame-by-frame
		ret, frame = capture_device.read()
		# if frame is read correctly ret is True
		if not ret:
			print("Can't receive frame (stream end?). Exiting.")
			break
		# Detecting Faces
		if count % FACE_DETECTION_FREQUENCY == 0:
			face, face_img = facedetect.detectFace(frame, "CV")
			if face is not None:
				cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 5)
			if face_img is not None:
				face_img = preprocess.preprocess(image=face_img, imageSize=(64, 64))
		# Display the resulting frame
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) == ord('q'):
			break
	# When everything done, release the capture
	capture_device.release()
	cv2.destroyAllWindows()

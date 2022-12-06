import numpy
import cv2
import os
from Utils.GUI import GUIController
from Utils import facedetect, preprocessing

CAMERA_ID = 2
FACE_DETECTION_FREQUENCY = 1
FACE_DETECTOR = "CV"

USE_CAM 		= 	False
SAVE_FRAMES 	= 	True
SAVE_FACES 		= 	True
FACE_ALIGN		= 	True
FACE_PREPROCESS = 	True

def getCameraCapture(camera_id=0):
	cap = cv2.VideoCapture(camera_id)
	if not cap.isOpened():
		print("Cannot open camera")
		exit(0)
	else:
		return cap

def orderDataFolder(folder):
	import re

	def sort_nicely(l):
		""" Sort a given list."""
		convert = lambda text: int(text) if text.isdigit() else text
		alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
		l.sort(key=alphanum_key)
		return l

	datalist = sort_nicely(os.listdir(folder))
	return datalist


if __name__ == "__main__":
	if USE_CAM:
		capture_device = getCameraCapture(CAMERA_ID)
		count = 0
		face = None
		while True:
			count += 1
			# Capture frame-by-frame
			ret, frame = capture_device.read()
			# if frame is read correctly ret is True
			if not ret:
				print("Can't receive frame (stream end?). Exiting.")
				break
			else:
				if SAVE_FRAMES:
					cv2.imwrite('Images/Frames1/Frame_' + str(count) + '.jpg', frame)
			# Detecting Faces
			if count % FACE_DETECTION_FREQUENCY == 0:
				face, face_img = facedetect.detectFace(frame=frame, face_detector=FACE_DETECTOR, aligned=FACE_ALIGN)
				if FACE_PREPROCESS:
					face_img = preprocessing.preprocess(image=face_img, grayscale=True, normalise=True, equalise=True)
				if face is not None:
					if SAVE_FACES:
						if FACE_ALIGN:
							cv2.imwrite('Images/Faces_Aligned/Face_' + str(count) + '.jpg', face_img)
						else:
							cv2.imwrite('Images/Faces/Face_' + str(count) + '.jpg', face_img)
					cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 5)

			cv2.imshow('frame', face_img)
			if cv2.waitKey(1) == ord('q'):
				break

		# When everything done, release the capture
		capture_device.release()
	else:
		frames_path = "Images/Frames/"
		count = 0
		for frame_path in orderDataFolder(frames_path):
			count += 1
			frame = cv2.imread(frames_path + frame_path)
			face, face_img_unprocessed = facedetect.detectFace(frame=frame, face_detector=FACE_DETECTOR, aligned=FACE_ALIGN)
			if face_img_unprocessed is None:
				continue
			else:
				if FACE_PREPROCESS:
					face_img = preprocessing.preprocess(image=face_img_unprocessed, grayscale=True, normalise=True, equalise=True)
				else:
					face_img = face_img_unprocessed
				if SAVE_FACES:
					if FACE_ALIGN:
						cv2.imwrite('Images/Faces_Aligned/Face_' + str(count) + '.jpg', face_img)
					else:
						cv2.imwrite('Images/Faces/Face_' + str(count) + '.jpg', face_img)
				# cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 5)
				cv2.imshow('frame', face_img)
				if cv2.waitKey(1) == ord('q'):
					break

	cv2.destroyAllWindows()

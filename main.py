import numpy
import cv2
import os
from Utils.GUI import GUIController
from Utils import facedetect, preprocessing

CAMERA_ID = 0
FACE_DETECTION_FREQUENCY = 1
FACE_DETECTOR = "CV"

USE_CAM = True

def getCameraCapture(CAMERA_ID=0):
	cap = cv2.VideoCapture(CAMERA_ID)
	if not cap.isOpened():
		print("Cannot open camera")
		exit(0)
	else:
		return cap

def orderDataFolder(folder):
	import re
	def sort_nicely(l):
		""" Sort the given list in the way that humans expect.
        """
		convert = lambda text: int(text) if text.isdigit() else text
		alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
		l.sort(key=alphanum_key)
		return l

	dataList = sort_nicely(os.listdir(folder))
	return dataList


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

			# Detecting Faces
			if count % FACE_DETECTION_FREQUENCY == 0:
				# face, face_img = facedetect.detectFace(frame=frame, face_detector=FACE_DETECTOR, aligned=False)
				# face_img = preprocess.preprocess(image=face_img, imageSize=(64, 64))
				face_aligned, face_img_aligned = facedetect.detectFace(frame=frame, face_detector=FACE_DETECTOR, aligned=True)
				if face_aligned is not None:
					cv2.rectangle(frame, (face_aligned[0], face_aligned[1]), (face_aligned[2], face_aligned[3]), (0, 255, 0), 5)
				if face_img_aligned is not None:
					face_img_aligned = preprocessing.preprocess(image=face_img_aligned, grayscale=True, normalise=True, imageSize=(64, 64), resize=True)


					cv2.imwrite('Images/Faces_Aligned/Face_' + str(count) + '.jpg', face_img_aligned)

			# Display the resulting frame
			# 	frame = numpy.concatenate((face_img, face_img_aligned), axis=1)
			cv2.imshow('frame', frame)
			if cv2.waitKey(1) == ord('q'):
				break

		# When everything done, release the capture
		capture_device.release()
		cv2.destroyAllWindows()
	else:
		frames_path = "Images/Frames/"
		count = 0
		for frame_path in orderDataFolder(frames_path):
			count += 1
			print("Reading Frame " + str(count) + "/" + str(len(orderDataFolder(frames_path))))
			print("Reading Frame " + frame_path)
			frame = cv2.imread(frames_path + frame_path)

			face, face_img = facedetect.detectFace(frame=frame, face_detector=FACE_DETECTOR, aligned=False)
			face_img = preprocessing.preprocess(image=face_img, grayscale=True, normalise=True, imageSize=(64, 64), resize=True)
			cv2.imwrite('Images/Faces/Face_' + str(count) + '.jpg', face_img)
			if face is not None:
				face_aligned, face_img_aligned = facedetect.detectFace(frame=frame, face_detector=FACE_DETECTOR, aligned=True)
				face_img_aligned = preprocessing.preprocess(image=face_img_aligned, grayscale=True, normalise=True, imageSize=(64, 64), resize=True)
				cv2.imwrite('Images/Faces_Aligned/Face_'+str(count)+'.jpg', face_img_aligned)


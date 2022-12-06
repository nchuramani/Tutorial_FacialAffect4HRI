import numpy
import cv2
import os
from Utils.GUI import GUIController
from Utils import facedetect, preprocessing

from FaceChannel.faceChannel import FaceChannelV1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

CAMERA_ID = 2
FACE_DETECTION_FREQUENCY = 2
FACE_DETECTOR = "CV"

USE_CAM 		= 	False
FACE_ALIGN 		= 	False
DETECT_FACE 	= 	True
USEAFFECT 		= 	True
FACECHANNEL 	= 	True
SAVE_FRAMES      =  False
FACE_PREPROCESS = 	True

"""Information about the created image"""
finalImageSize = (1024, 768) # Size of the final image generated for display
categoricalInitialPosition = 10 # Initial position for adding the categorical plot in the final image
faceSize = (64, 64) # Input size for the FACECHANNEL

def loadModels():
	facechannel_cat = FaceChannelV1(type="Cat", loadModel=True)
	facechannel_dim = FaceChannelV1(type="Dim", loadModel=True)
	return facechannel_cat, facechannel_dim

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
	GUI = GUIController()
	if USEAFFECT:
		config = tf.compat.v1.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.compat.v1.Session(config=config)
		if FACECHANNEL:
			facechannel_cat, facechannel_dim = loadModels()
		arousals = []
		valences = []
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
			if DETECT_FACE:
				# Detecting Faces
				if count % FACE_DETECTION_FREQUENCY == 0:
					face, face_img = facedetect.detectFace(frame=frame, face_detector=FACE_DETECTOR, aligned=FACE_ALIGN)
					if face_img is None:
						continue
					else:
						face_img_preprocessed = preprocessing.preprocess(image=face_img, imageSize=(64, 64), resize=True, normalise=True, grayscale=True)
					if SAVE_FRAMES:
						cv2.imwrite('Images/Faces_Aligned/Face_' + str(count) + '.jpg', face_img_preprocessed)

						if USEAFFECT:
								if FACECHANNEL:
									# create display image and copy the captured frame to it
									image = numpy.zeros((finalImageSize[1], finalImageSize[0], 3), numpy.uint8)
									image[0:480, 0:640] = frame
									frame_view = image
									if face is not None:
										# Obtain dimensional classification
										dimensionalRecognition = numpy.array(
											facechannel_dim.predict([face_img_preprocessed], preprocess=False))
										# Obtain Categorical classification
										categoricalRecognition = \
										facechannel_cat.predict([face_img_preprocessed], preprocess=False)[0]
										# Print the square around the categorical face
										frame = GUI.createDetectedFaceGUI(frame=frame_view, detectedFace=[face],
																		  faceFrameColor=facechannel_cat.CAT_CLASS_COLOR,
																		  categoricalClassificationReport=categoricalRecognition)

										# # # Create the categorical classification
										frame = GUI.createCategoricalEmotionGUI(categoricalRecognition, frame_view,
																				facechannel_cat.CAT_CLASS_COLOR,
																				facechannel_cat.CAT_CLASS_ORDER,
																				initialPosition=categoricalInitialPosition)
										# # Create the dimensional graph
										frame = GUI.createDimensionalEmotionGUI(
											classificationReport=dimensionalRecognition, frame=frame,
											categoricalReport=[], categoricalDictionary=None)

										if len(arousals) > 100:
											arousals.pop(0)
											valences.pop(0)

										# Create dimensional plot
										arousals.append(dimensionalRecognition[0][0][0])
										valences.append(dimensionalRecognition[1][0][0])

										frame = GUI.createDimensionalPlotGUI(arousals=arousals, valences=valences,frame=frame)
						cv2.imshow('frame', frame)
						if cv2.waitKey(1) == ord('q'):
							break
			else:
				cv2.imshow('frame', frame)
				if cv2.waitKey(1) == ord('q'):
					break

		# When everything done, release the capture
		capture_device.release()
		cv2.destroyAllWindows()
	else:
		frames_path = "Images/Frames1/"
		count = 0
		for frame_path in orderDataFolder(frames_path):
			count += 1
			frame = cv2.imread(frames_path + frame_path)
			face, face_img = facedetect.detectFace(frame=frame, face_detector=FACE_DETECTOR, aligned=FACE_ALIGN)
			if face_img is None:
				face_img_preprocessed = None
				continue
			else:
				cv2.imwrite('Images/Faces/Face_' + str(count) + '.jpg', face_img)

				face_img_preprocessed = preprocessing.preprocess(image=face_img, imageSize=(64, 64), resize=True,
																 normalise=True, grayscale=True)

				if USEAFFECT:
					if FACECHANNEL:
						# create display image and copy the captured frame to it
						image = numpy.zeros((finalImageSize[1], finalImageSize[0], 3), numpy.uint8)
						image[0:480, 0:640] = frame
						frame_view = image
						if face is not None:
							# Obtain dimensional classification
							dimensionalRecognition = numpy.array(
								facechannel_dim.predict([face_img_preprocessed], preprocess=False))
							# Obtain Categorical classification
							categoricalRecognition = facechannel_cat.predict([face_img_preprocessed], preprocess=False)[
								0]
							# Print the square around the categorical face
							frame = GUI.createDetectedFaceGUI(frame=frame_view, detectedFace=[face],
															  faceFrameColor=facechannel_cat.CAT_CLASS_COLOR,
															  categoricalClassificationReport=categoricalRecognition)

							# # # Create the categorical classification
							frame = GUI.createCategoricalEmotionGUI(categoricalRecognition, frame_view,
																	facechannel_cat.CAT_CLASS_COLOR,
																	facechannel_cat.CAT_CLASS_ORDER,
																	initialPosition=categoricalInitialPosition)
							# # Create the dimensional graph
							frame = GUI.createDimensionalEmotionGUI(classificationReport=dimensionalRecognition,
																	frame=frame, categoricalReport=[],
																	categoricalDictionary=None)

							if len(arousals) > 100:
								arousals.pop(0)
								valences.pop(0)

							# Create dimensional plot
							arousals.append(dimensionalRecognition[0][0][0])
							valences.append(dimensionalRecognition[1][0][0])

							frame = GUI.createDimensionalPlotGUI(arousals=arousals, valences=valences, frame=frame)
					else:
						continue
				cv2.imshow('frame', frame)
				if cv2.waitKey(1) == ord('q'):
					break


	cv2.destroyAllWindows()

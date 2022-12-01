
import cv2
import numpy
from pathlib import Path
import os
import sys
import dlib

CASCADE_PATH = "Utils/OpenCV_FaceCascade/haarcascade_frontalface_alt.xml"
def detectFace(frame, detector="CV"):
	if detector == "CV":
		if (len(frame.shape) == 3):
			img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		cascade = cv2.CascadeClassifier(CASCADE_PATH)
		rects = cascade.detectMultiScale(img, 1.3, 4, 1, (20, 20))
		
		if len(rects) == 0:
			face = None
			face_img = None
		else:
			rects[:, 2:] += rects[:, :2]
			x1, y1, x2, y2 = rects[0]
			face = rects[0]
			face_img = frame[y1:y2, x1:x2]
			return face, face_img
	else:
		detector = dlib.get_frontal_face_detector()
		detected_faces = detector(frame, 1)
		if len(detected_faces) > 0:
			for i, d in enumerate(detected_faces):
				face = [d.left(), d.top(), d.right(), d.bottom()]
				face_img = frame[face[1]:face[3], face[0]:face[2]]
				break
		else:
			face = None
			face_img = None
	return face, face_img

# class imageProcessingUtil:
#
#
#
#     faceDetectionMaximumFrequency = 10
#     """Search for a new face every x frames."""
#
#     previouslyDetectedface = None
#     """Identify if in the previous frame, a face was detected"""
#
#
#     currentFaceDetectionFrequency = -1
#     """A counter to identify which is the current frame for face detection"""
#
#
#     @property
#     def faceDetector(self):
#         """get the cv2 face detector
#         """
#         return self._faceDetector
#
#
#     def __init__(self):
#         """Constructor of the FaceChannelV1 image processing module
#         """
#
#         folderName = Path(os.path.abspath(sys.modules[imageProcessingUtil.__module__].__file__)).parent / "TrainedNetworks" / "faceDetector"
#
#         prototxtPath = folderName /  "deploy.prototxt"
#         weightsPath = folderName / "res10_300x300_ssd_iter_140000.caffemodel"
#         self._faceDetector=  cv2.dnn.readNet(str(prototxtPath), str(weightsPath))
#
#
#     def preProcess(self, image, imageSize= (64,64)):
#         """Pre-process an image to make it ready to be used as input to the FaceChannelV1
#                 :param image: ndarray with the image to be processed.
#                 :param imageSize: tuple with the final image size, default is (64x64)
#                 :return: The pre-processed image
#                 :rtype: ndarray
#         """
#
#         image = numpy.array(image)
#
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#         image = numpy.array(cv2.resize(image, imageSize))
#
#         image = numpy.expand_dims(image, axis=0)
#
#         image = image.astype('float32')
#
#         image /= 255
#
#         return image
#
#
#
#
#     def detectFace(self, image, multiple=False):
#         """Detect a face using the cv2 face detector. It detects a face every "faceDetectionMaximumFrequency" frames.
#                 :param image: ndarray with the image to be processed.
#                 :param multiple: allows the code to detect multiple faces in one single frame
#                 :return: dets: the tuple of the position of the recognized face. using the format: startX, startY, endX, endY. A list if multiple faces are detected.
#                 :rtype: ndarray
#                 :return: face: the image of the detected face. A list if multiple feces are detected.
#                 :rtype: ndarray
#         """
#
#         image = numpy.array(image)
#         (h, w) = image.shape[:2]
#
#         # print ("Image shape:" + str((h,w)))
#         # construct a blob from the image
#         blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
#                                      (104.0, 177.0, 123.0))
#
#         self._faceDetector.setInput(blob)
#         detections = self._faceDetector.forward()
#
#         # print ("detections: " + str(detections.shape))
#         # input("here")
#         face = image
#
#         if multiple:
#             face = []
#             face.append(image)
#
#             dets = []
#
#         for i in range(0, detections.shape[2]):
#             # extract the confidence (i.e., probability) associated with
#             # the detection
#             confidence = detections[0, 0, i, 2]
#             # print ("Confidence:" + str(confidence))
#
#             # filter out weak detections by ensuring the confidence is
#             # greater than the minimum confidence
#             if confidence > 0.5:
#                 # compute the (x, y)-coordinates of the bounding box for
#                 # the object
#                 box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#
#                 # ensure the bounding boxes fall within the dimensions of
#                 # the frame
#                 (startX, startY) = (max(0, startX), max(0, startY))
#                 (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
#
#                 # extract the face ROI, convert it from BGR to RGB channel
#                 # ordering, resize it to 224x224, and preprocess it
#
#
#                 if multiple:
#                     face.append(image[startY:endY, startX:endX])
#                     dets.append([[startX,startY,  endX, endY]])
#                     self.previouslyDetectedface = dets[-1]
#                 else:
#                     face = image[startY:endY, startX:endX]
#                     dets = [[startX, startY, endX, endY]]
#                     self.previouslyDetectedface = dets
#
#                 # print("--shape Image:" + str(image.shape))
#                 # print("--shape Face:" + str(face.shape))
#                 #
#                 # print("--Detected XY: (" + str(startX) + "),(" + str(startY) + "),(" + str(startX) + "+" + str(
#                 #     endX) + ") - " + str(startY) + "+" + str(endY))
#
#         # input("here")
#         if multiple:
#          dets.append(self.previouslyDetectedface)
#         else:
#          dets = self.previouslyDetectedface
#
#
#         return dets, face

import numpy
import cv2


def preprocess(image, imageSize=(64, 64)):
	image = numpy.array(image)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = numpy.array(cv2.resize(image, imageSize))
	image = numpy.expand_dims(image, axis=0)
	image = image.astype('float32')
	image /= 255
	return image

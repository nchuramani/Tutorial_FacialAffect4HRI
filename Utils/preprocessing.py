import numpy
import cv2


def preprocess(image, grayscale=False, normalise=False, imageSize=(64, 64), resize=False):
	image = numpy.array(image)
	if grayscale:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# image = numpy.expand_dims(image, axis=0)
	if normalise:
		image = cv2.normalize(src=image, dst=image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		image = image.astype('float32')
	if resize:
		image = numpy.array(cv2.resize(image, imageSize))
	return image

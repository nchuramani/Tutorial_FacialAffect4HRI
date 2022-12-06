import matplotlib.pyplot as plt
from skimage.filters import gaussian, sobel
import numpy
import cv2
from Utils import facedetect
image = cv2.imread("Sample_Images/sample1.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

face, face_img = facedetect.detectFace(frame=image, face_detector="CV", aligned=False)
norm_image_max = [i / numpy.max(face_img) for i in face_img]
norm_image_mean = [i - numpy.mean(face_img) for i in face_img]
norm_image_mean_std = [(i - numpy.mean(face_img))/ numpy.std(face_img) for i in face_img]

# apply Gaussian blur, creating a new image
sigma = 3.0
blurred_image = gaussian(face_img, sigma=(sigma, sigma), truncate=3.5)

filt_real = sobel(face_img)

fig = plt.figure(tight_layout='auto', figsize=(12, 7))
fig.add_subplot(231)
plt.title('Grayscale')
plt.imshow(face_img)
fig.add_subplot(232)
plt.title('Normalised Divide by Max')
plt.imshow(norm_image_max)
fig.add_subplot(233)
plt.title('Normalised Subtracting Mean')
plt.imshow(norm_image_mean)
fig.add_subplot(234)
plt.title('Normalised Subtracting Mean Divide by Std')
plt.imshow(norm_image_mean_std)

fig.add_subplot(235)
plt.title('Gaussian Blur')
plt.imshow(blurred_image)

fig.add_subplot(236)
plt.title('Sobel Filter')
plt.imshow(filt_real)
# plt.gray()
plt.show()
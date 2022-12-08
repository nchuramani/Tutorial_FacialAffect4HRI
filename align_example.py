import matplotlib.pyplot as plt
import cv2
from Utils import facedetect


image = cv2.imread("Sample_Images/sample3.jpeg")

face, face_img = facedetect.detectFace(frame=image, face_detector="CV", aligned=False)

face_aligned, face_img_aligned = facedetect.detectFace(frame=image, face_detector="CV", aligned=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), cmap=plt.cm.gray)
ax1.set_title('Input image')

ax2.axis('off')
ax2.imshow(cv2.cvtColor(face_img_aligned, cv2.COLOR_BGR2GRAY), cmap=plt.cm.gray)
ax2.set_title('Input image')
plt.show()
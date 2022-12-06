import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern, canny
import cv2

image = cv2.imread("Sample_Images/sample3.jpeg")

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True)

lbp_image = local_binary_pattern(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), P=8, R=1.5)

canny_image = canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), sigma=1.0, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False)

fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

ax1[0].axis('off')
ax1[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap=plt.cm.gray)
ax1[0].set_title('Input image')


ax1[1].axis('off')
ax1[1].imshow(hog_image, cmap=plt.cm.gray)
ax1[1].set_title('Histogram of Oriented Gradients')

ax2[0].axis('off')
ax2[0].imshow(lbp_image, cmap=plt.cm.gray)
ax2[0].set_title('Local Binary Patterns')


ax2[1].axis('off')
ax2[1].imshow(canny_image, cmap=plt.cm.gray)
ax2[1].set_title('Canny Edge Features')
plt.show()
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Load grayscale image
img = cv2.imread("camera_mapping_image.jpg", cv2.IMREAD_GRAYSCALE)

# 2. Apply Histogram Equalization for contrast enhancement
equalized = cv2.equalizeHist(img)

# 3. Plot images and their histograms
plt.figure(figsize=(10, 6))

# Original image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Original Image")
plt.axis("off")

# Original histogram
plt.subplot(2, 2, 2)
plt.hist(img.ravel(), bins=256, range=(0, 256), color="black")
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# Equalized image
plt.subplot(2, 2, 3)
plt.imshow(equalized, cmap="gray", vmin=0, vmax=255)
plt.title("Equalized Image")
plt.axis("off")

# Equalized histogram
plt.subplot(2, 2, 4)
plt.hist(equalized.ravel(), bins=256, range=(0, 256), color="black")
plt.title("Equalized Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("3.png", dpi=300)
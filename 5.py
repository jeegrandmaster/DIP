import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Load grayscale image
img = cv2.imread("camera_mapping_image.jpg", cv2.IMREAD_GRAYSCALE)

# 2. Edge detection (Canny)
edges = cv2.Canny(img, threshold1=100, threshold2=200)

# 3. Basic global thresholding
T = 128
_, thresh_img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)

# 4. Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap="gray", vmin=0, vmax=255)
plt.title("Edge Detection (Canny)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(thresh_img, cmap="gray", vmin=0, vmax=255)
plt.title(f"Global Thresholding (T={T})")
plt.axis("off")

plt.tight_layout()
plt.savefig("5.png", dpi=300)
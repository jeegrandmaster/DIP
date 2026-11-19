import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Load image in grayscale (0 to 255)
gray = cv2.imread("camera_mapping_image.jpg", cv2.IMREAD_GRAYSCALE)
print("Grayscale shape:", gray.shape, "dtype:", gray.dtype)

# 2. Convert to binary (0 and 255)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
print("Binary shape:", binary.shape, "dtype:", binary.dtype)

# 3. Display both images
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary, cmap="gray", vmin=0, vmax=255)
plt.title("Binary Image")
plt.axis("off")

plt.tight_layout()
plt.savefig("1.png", dpi=300)
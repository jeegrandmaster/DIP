import cv2
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load grayscale image
image_path = Path(__file__).with_name("camera_mapping_image.jpg")
img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")
h, w = img.shape

# 1. Translation (shift right by 50px, down by 30px)
M_trans = np.float32([[1, 0, 50], [0, 1, 30]])
translated = cv2.warpAffine(img, M_trans, (w, h))

# 2. Rotation (30 degrees around center)
M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
rotated = cv2.warpAffine(img, M_rot, (w, h))

# 3. Scaling (1.5x width, 0.75x height)
scaled = cv2.resize(img, None, fx=1.5, fy=0.75, interpolation=cv2.INTER_LINEAR)

# 4. Shearing (horizontal shear by 0.3)
M_shear = np.float32([[1, 0.3, 0], [0, 1, 0]])
sheared = cv2.warpAffine(img, M_shear, (int(w + 0.3 * h), h))

# Display results
titles = ["Original", "Translated", "Rotated", "Scaled", "Sheared"]
images = [img, translated, rotated, scaled, sheared]

plt.figure(figsize=(15, 3))
for i in range(5):
  plt.subplot(1, 5, i + 1)
  plt.imshow(images[i], cmap="gray", vmin=0, vmax=255)
  plt.title(titles[i])
  plt.axis("off")

plt.tight_layout()
plt.savefig("2.png", dpi=300)
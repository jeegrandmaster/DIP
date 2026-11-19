import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# 1. Load grayscale image
img = cv2.imread("camera_mapping_image.jpg", cv2.IMREAD_GRAYSCALE)

# 2. Compute 2D FFT and shift zero-frequency component to the center
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# 3. Compute magnitude spectrum (log scale for visualization)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# 4. Inverse FFT to reconstruct the image
f_ishift = np.fft.ifftshift(fshift)
reconstructed = np.abs(np.fft.ifft2(f_ishift))

# 5. Display original, magnitude spectrum, and reconstructed image
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("FFT Magnitude Spectrum")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(reconstructed, cmap="gray")
plt.title("Reconstructed Image (IFFT)")
plt.axis("off")

plt.tight_layout()
plt.savefig("4.png", dpi=300)
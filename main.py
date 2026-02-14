import cv2
import enhancer

img = cv2.imread("input.jpg")

img = enhancer.reduce_noise(img)
img = enhancer.enhance_contrast(img)
img = enhancer.sharpen(img)
img = enhancer.upscale(img, 2)

cv2.imwrite("enhanced.jpg", img)
print("Done! Saved as enhanced.jpg")

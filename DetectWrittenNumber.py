import cv2
import numpy as np
import pytesseract
from PIL import Image
img = cv2.imread("01.jpg")
scale_percent = 20
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
hsv_img = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image", hsv_img)
lower_green = np.array([60, 111, 16])
upper_green = np.array([179, 255, 255])
masking = cv2.inRange(hsv_img, lower_green, upper_green)
cv2.imshow("Orginal Image", resized)
cv2.imshow("Green Color Detection", masking)
invert_masking = cv2.subtract(255, masking)
cv2.imshow("Invert Image", invert_masking)
cv2.waitKey(0)
custom_config = r"--psm 8 -c tessedit_char_whitelist=0123456789"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
numbers_string = pytesseract.image_to_string(invert_masking, config=custom_config)
print(numbers_string)
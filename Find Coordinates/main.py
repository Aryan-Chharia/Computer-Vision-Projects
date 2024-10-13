import numpy as np
import cv2
import matplotlib.pyplot as plt

# Reading the image
font = cv2.FONT_HERSHEY_COMPLEX
img2 = cv2.imread('C:\\Contributions\\Programs\\ML_DL\\Find Coordinates\\hey.jpg', cv2.IMREAD_COLOR)

# Reading the same image in grayscale format
img = cv2.imread('C:\\Contributions\\Programs\\ML_DL\\Find Coordinates\\hey.jpg', cv2.IMREAD_GRAYSCALE)

# Converting image to a binary image (black and white)
_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

# Detecting contours in the image
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Going through each contour found in the image
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

    # Draws boundary of contours
    cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5)

    # Flattens the array containing the coordinates of the vertices
    n = approx.ravel()
    i = 0

    for j in n:
        if(i % 2 == 0):
            x = n[i]
            y = n[i + 1]

            # String containing the coordinates
            string = str(x) + " " + str(y)

            if(i == 0):
                # Text on topmost coordinate
                cv2.putText(img2, "Arrow tip", (x, y), font, 0.5, (255, 0, 0))
            else:
                # Text on remaining coordinates
                cv2.putText(img2, string, (x, y), font, 0.5, (0, 255, 0))
        i = i + 1

# Using matplotlib to display the final image with contours
plt.figure(figsize=(10, 7))

# Converting from BGR to RGB for display
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Plot the image
plt.imshow(img2_rgb)
plt.title('Image with Contours and Coordinates')
plt.axis('off')  # Hide the axis
plt.show()

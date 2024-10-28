import cv2

img = cv2.imread(r"C:\Users\Sridi\Desktop\mini2\images.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_gray = cv2.GaussianBlur(gray, (7,7), 0)
edges = cv2.Canny(blurred_gray, 30, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if cv2.contourArea(contour) > 20 and len(approx) > 4:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)

cv2.imshow("Cars", img)
cv2.waitKey(0)
cv2.destroyAllWindows()













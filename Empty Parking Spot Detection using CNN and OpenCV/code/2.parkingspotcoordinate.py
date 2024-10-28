import cv2

drawing = False
ix, iy = -1,-1
img = cv2.imread(r"C:\Users\Sridi\Desktop\mini2\images.jpg")
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
           img_rect =  img.copy()
           cv2.rectangle(img_rect, (ix,iy), (x,y), (0, 255, 0), 2)
           cv2.imshow("image",img_rect)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix,iy), (x,y), (0, 255, 0), 2)
        cv2.imshow("image",img)
        print("Top Left:", (ix,iy))
        print("Bottom Right:", (x,y))

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)
cv2.imshow('image', img)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()









    
import cv2
import numpy as np
from pyzbar.pyzbar import decode  # type: ignore # Decodes barcodes/QR codes.

# Opens the default webcam.
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    
    if not success:
        print("Failed to capture image from camera.")
        break  # Exit the loop if the image capture fails.

    # Decode barcodes/QR codes in the current frame.
    decoded_objects = decode(img)
    for barcode in decoded_objects:
        myData = barcode.data.decode('utf-8')
        print(myData)

        # Write scanned data into a text file.
        with open("result.txt", "a") as file:
            file.write(myData + "\n")
        
        # Draw a polygon around the detected barcode/QR code.
        pts = np.array(barcode.polygon, dtype=np.int32) if len(barcode.polygon) == 4 else np.array(barcode.rect)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 0, 255), 5)

        # Display the decoded data on the image.
        pts2 = barcode.rect
        cv2.putText(img, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # Show the result in a window.
    cv2.imshow('Result', img)
    
    # Break the loop if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window.
cap.release()
cv2.destroyAllWindows()

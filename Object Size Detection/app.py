import streamlit as st
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def process_image(image, width):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    
    pixelsPerMetric = None
    results = []
    
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width
        
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
        
        results.append((box, dimA, dimB))
    
    return results

def draw_results(image, results):
    for (box, dimA, dimB) in results:
        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
        
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        
        cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        
        cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
        
        cv2.putText(image, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(image, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    
    
    return image

st.title("Object Size Measurement")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
width = st.number_input("Width of the left-most object (in inches)", value=0.955, step=0.001)

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    results = process_image(image, width)
    
    output_image = draw_results(image.copy(), results)
    
    st.image(output_image, channels="BGR", caption="Processed Image")
    
    st.write(f"Number of objects detected: {len(results)}")
    
    for i, (_, dimA, dimB) in enumerate(results, 1):
        st.write(f"Object {i}: {dimA:.1f}in x {dimB:.1f}in")
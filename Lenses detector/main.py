# importing libraries
import cv2
import datetime

# Function to detect eyelenses in the frame
def detect_eyelenses(frame, cascade_classifier):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyelenses = cascade_classifier.detectMultiScale(gray, 1.3, 5)
    return eyelenses

# Function to log detection results to an HTML file
def log_to_htmlfile(timestamp, coordinates, htmlfile_path):
    with open(htmlfile_path, 'a') as file:
        file.write(f"""
        <tr>
            <td>{timestamp}</td>
            <td>{coordinates}</td>
        </tr>
        """)

# Function to initialize the HTML file structure
def initialize_htmlfile(htmlfile_path):
    with open(htmlfile_path, 'w') as file:
        file.write("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Detection Results</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f4f4f4;
                }
                h1 {
                    text-align: center;
                    color: #333;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #4CAF50;
                    color: white;
                }
                tr:hover {
                    background-color: #f1f1f1;
                }
            </style>
        </head>
        <body>
            <h1>Eyeglass Detection Results</h1>
            <table>
                <tr>
                    <th>Timestamp</th>
                    <th>Coordinates (x, y, width, height)</th>
                </tr>
        """)

# Function to finalize the HTML file structure
def finalize_htmlfile(htmlfile_path):
    with open(htmlfile_path, 'a') as file:
        file.write("""
            </table>
        </body>
        </html>
        """)
def main():
    # Load the Haar cascade for eyelenses detection
    eyelenses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    
    htmlfile_path = 'results.html'
    initialize_htmlfile(htmlfile_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Detect eyelenses
        eyelenses = detect_eyelenses(frame, eyelenses_cascade)

        # Creating rectangle on the detected lenses and log results
        for (x, y, w, h) in eyelenses:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Get timestamp and coordinates
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            coordinates = f"x={x}, y={y}, width={w}, height={h}"

            # results to HTML file
            log_to_htmlfile(timestamp, coordinates, htmlfile_path)

        # Display the resulting frame
        cv2.imshow('Eyelenses Detector', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Finalize the HTML file
    finalize_htmlfile(htmlfile_path)

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

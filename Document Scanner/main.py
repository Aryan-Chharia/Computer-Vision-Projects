import cv2
from scanner import scan_document
from ocr import extract_text
from tkinter import filedialog, Tk

def main():
    # Initialize Tkinter window to browse files
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select Document to Scan")

    if file_path:
        # Scan document (perspective correction and enhancement)
        scanned_image = scan_document(file_path)
        
        # Show the scanned image
        cv2.imshow("Scanned Document", scanned_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Perform OCR to extract text
        extracted_text = extract_text(scanned_image)
        print("\nExtracted Text:\n", extracted_text)

if __name__ == "__main__":
    main()

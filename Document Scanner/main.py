import cv2
from scanner import scan_document
from ocr import extract_text
from tkinter import filedialog, Tk, messagebox, Button, Label, Text, END
import os

def save_scanned_image(image):
    # Ask user for the file format to save the image
    format_choice = filedialog.askquestion("Save Image", "Do you want to save the scanned image as PNG?")
    if format_choice == 'yes':
        file_path = filedialog.asksaveasfilename(defaultextension=".png", title="Save Image", filetypes=[("PNG files", "*.png")])
        if file_path:
            cv2.imwrite(file_path, image)
            messagebox.showinfo("Image Saved", f"Scanned image saved as {os.path.basename(file_path)}")
    else:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", title="Save Image", filetypes=[("JPEG files", "*.jpg")])
        if file_path:
            cv2.imwrite(file_path, image)
            messagebox.showinfo("Image Saved", f"Scanned image saved as {os.path.basename(file_path)}")

def show_extracted_text(text):
    # Create a new Tkinter window for displaying extracted text
    text_window = Tk()
    text_window.title("Extracted Text")

    label = Label(text_window, text="Extracted Text:")
    label.pack()

    text_box = Text(text_window, wrap='word', width=80, height=20)
    text_box.pack()
    text_box.insert(END, text)

    # Button to close the text window
    close_button = Button(text_window, text="Close", command=text_window.destroy)
    close_button.pack()

    text_window.mainloop()

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

        # Option to save the scanned image
        save_scanned_image(scanned_image)

        # Perform OCR to extract text
        extracted_text = extract_text(scanned_image)
        show_extracted_text(extracted_text)

if __name__ == "__main__":
    main()

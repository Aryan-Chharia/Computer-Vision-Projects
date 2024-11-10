from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def solve_captcha(filename):
    threshold = 140

    # Open and process the selected image
    original = Image.open(filename)
    black_and_white = original.convert("L")
    processed_image = black_and_white.point(lambda p: p > threshold and 255)
    
    # Save the processed image for reference
    processed_image.save('processed_captcha.png')

    # Perform OCR on the processed image
    captcha_text = pytesseract.image_to_string(processed_image)
    captcha_text = captcha_text.replace(' ', '').strip().upper()

    # Display the result in the label
    result_label.config(text=f"Solved Captcha: {captcha_text}")
    return captcha_text

def upload_image():
    filename = filedialog.askopenfilename(
        title="Select CAPTCHA Image",
        filetypes=(("PNG files", "*.png"), ("All files", "*.*"))
    )
    if filename:
        img = Image.open(filename)
        img.thumbnail((250, 150))
        img_tk = ImageTk.PhotoImage(img)
        
        # Update the label to show the selected CAPTCHA image
        captcha_image_label.config(image=img_tk)
        captcha_image_label.image = img_tk
        
        # Enable the solve button and link it to solve_captcha function
        solve_button.config(state="normal", command=lambda: solve_captcha(filename))

# Initialize the GUI window
root = Tk()
root.title("CAPTCHA Solver")
root.geometry("300x300")

# Label for displaying the CAPTCHA Image
captcha_image_label = Label(root)
captcha_image_label.pack(pady=10)

# Button to Upload CAPTCHA Image
upload_button = Button(root, text="Upload CAPTCHA Image", command=upload_image)
upload_button.pack(pady=10)

# Button to Solve CAPTCHA, initially disabled
solve_button = Button(root, text="Solve CAPTCHA", state="disabled")
solve_button.pack(pady=10)

# Label to Display Result
result_label = Label(root, text="Solved Captcha: ")
result_label.pack(pady=10)

root.mainloop()

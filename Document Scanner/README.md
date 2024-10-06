# 📄 Document Scanner App 🔍

This app allows you to scan documents using your camera and extract text from them! 📸✨

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Aryan-Chharia/Computer-Vision-Projects.git
```

### 2. Navigate to the Project Directory

```bash
cd document-scanner
```

### 3. Install Required Python Packages

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

#### For Windows:

- [Download the installer from Tesseract GitHub](https://github.com/tesseract-ocr/tesseract)
- Install Tesseract OCR
- Add the Tesseract installation directory to your system PATH

#### For macOS:

```bash
brew install tesseract
```

#### For Linux:

```bash
sudo apt-get install tesseract-ocr
```

### 5. Set up the Tesseract Path in Your Python Script

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path for your system
```

## 🎯 How It Works

- It processes the image to enhance readability 🖼️
- Tesseract OCR (`pytesseract`) is used to extract text from the processed image 🔤

## 🛠️ Usage

Run the main script to start the document scanner:

```bash
python main.py
```

import cv2
import face_recognition
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import pickle
import numpy as np
import pickle
import random
import os
import urllib.request


def save_face(image_path, name=None, mobile_number_1=None, mobile_number_2=None, email_id=None, address=None, state=None, city=None, country=None):
    # Load image and get face encoding
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if not encodings:
        raise ValueError("No face encodings found in the image.")
    
    encoding = encodings[0]
    
    # Serialize encoding to binary
    encoding_binary = pickle.dumps(encoding)
    
    # Read the image file to binary data
    with open(image_path, 'rb') as f:
        image_binary = f.read()

    # Save encoding and image to database
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()

    c.execute('''
        INSERT INTO faces (name, mobile_number_1, mobile_number_2, email_id, address, state, city, country, encoding, image)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, mobile_number_1, mobile_number_2, email_id, address, state, city, country, encoding_binary, image_binary))
    
    conn.commit()
    conn.close()


# Insert faces

save_face("arul.jpeg", "Arul", "9876543210", "8765432109", "arul@gmail.com", "456 Maple Avenue", "Karnataka", "Bangalore", "India")
save_face("navin.jpg", "Navin", "9123456789", "8912345678", "navin@gmail.com", "789 Oak Boulevard", "Maharashtra", "Mumbai", "India")
save_face("modi.jpg", "Bob Brown", "9345678901", "8098765432", "bob.brown@example.com", "101 Pine Road", "Delhi", "New Delhi", "India")

conn = sqlite3.connect('faces.db')
c = conn.cursor()

# Function to generate random datetime within the last year
def random_date_within_year():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    random_date = start_date + (end_date - start_date) * random.random()
    return random_date

# Function to generate random total time in hours
def random_total_time():
    return round(random.uniform(0.5, 8.0), 2)

# Function to insert recognition log
def save_recognition_log(face_id, name, date, start_time, total_time):
    c.execute('''
        INSERT INTO recognition_logs (face_id, name, date, start_time, total_time)
        VALUES (?, ?, ?, ?, ?)
    ''', (face_id, name, date, start_time, total_time))
    conn.commit()

# Retrieve face data
c.execute('SELECT id, name FROM faces')
faces = c.fetchall()

# Generate and insert recognition logs
for face_id, name in faces:
    for _ in range(50):  # 50 logs per face for one year
        random_datetime = random_date_within_year()
        date_str = random_datetime.strftime('%Y-%m-%d')
        start_time_str = random_datetime.strftime('%H:%M:%S')
        total_time_hours = random_total_time()
        save_recognition_log(face_id, name, date_str, start_time_str, total_time_hours)


print("Data generation complete.")


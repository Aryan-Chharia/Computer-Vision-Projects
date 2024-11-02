import pickle
import cv2
import face_recognition
import sqlite3
import numpy as np
from datetime import datetime

def save_face(name, mobile_number_1, encoding, frame, mobile_number_2=None, email_id=None, address=None, state=None, city=None, country=None):
    if name is None:
        name = "Unknown"
    
    encoding_binary = pickle.dumps(encoding)
    _, image_binary = cv2.imencode('.jpg', frame)

    conn = sqlite3.connect('faces.db')
    c = conn.cursor()

    c.execute('''
        INSERT INTO faces (name, mobile_number_1, mobile_number_2, email_id, address, state, city, country, encoding, image)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, mobile_number_1, mobile_number_2, email_id, address, state, city, country, encoding_binary, image_binary.tobytes()))
    
    conn.commit()
    conn.close()

def load_encodings():
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute('SELECT id, name, mobile_number_1, encoding FROM faces')
    data = c.fetchall()
    conn.close()
    
    known_face_encodings = []
    known_face_names = []
    known_face_metadata = []
    for face_id, name, phone, encoding in data:
        known_face_encodings.append(pickle.loads(encoding))
        known_face_names.append(name)
        known_face_metadata.append((face_id, name, phone))
    
    return known_face_encodings, known_face_names, known_face_metadata

def is_encoding_in_database(encoding):
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute('SELECT encoding FROM faces')
    data = c.fetchall()
    conn.close()
    
    for stored_encoding in data:
        stored_encoding = pickle.loads(stored_encoding[0])
        matches = face_recognition.compare_faces([stored_encoding], encoding)
        if True in matches:
            return True
    return False

def is_unknown_face_in_database(encoding):
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute('SELECT encoding FROM faces WHERE name = "Unknown"')
    data = c.fetchall()
    conn.close()
    
    for stored_encoding in data:
        stored_encoding = pickle.loads(stored_encoding[0])
        matches = face_recognition.compare_faces([stored_encoding], encoding)
        if True in matches:
            return True
    return False

def log_recognition(face_id, name, is_known):
    today = datetime.now().strftime('%Y-%m-%d')
    now = datetime.now()

    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    
    if is_known:
        c.execute('''
            SELECT id, start_time, total_time FROM recognition_logs 
            WHERE face_id = ? AND date = ?
        ''', (face_id, today))
        
        result = c.fetchone()
        
        if result:
            log_id, start_time, total_time = result
            start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            duration = (now - start_time).total_seconds() / 60.0
            new_total_time = total_time + duration
            c.execute('''
                UPDATE recognition_logs 
                SET total_time = ?, start_time = ?
                WHERE id = ?
            ''', (new_total_time, now.strftime('%Y-%m-%d %H:%M:%S'), log_id))
        else:
            c.execute('''
                INSERT INTO recognition_logs (face_id, name, date, start_time, total_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (face_id, name, today, now.strftime('%Y-%m-%d %H:%M:%S'), 0))
    
    else:
        c.execute('''
            SELECT id, start_time, total_time FROM recognition_logs 
            WHERE face_id IS NULL AND date = ?
        ''', (today,))
        
        result = c.fetchone()
        
        if result:
            log_id, start_time, total_time = result
            start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            duration = (now - start_time).total_seconds() / 60.0
            new_total_time = total_time + duration
            c.execute('''
                UPDATE recognition_logs 
                SET total_time = ?, start_time = ?
                WHERE id = ?
            ''', (new_total_time, now.strftime('%Y-%m-%d %H:%M:%S'), log_id))
        else:
            c.execute('''
                INSERT INTO recognition_logs (face_id, name, date, start_time, total_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (None, 'Unknown', today, now.strftime('%Y-%m-%d %H:%M:%S'), 0))
    
    conn.commit()
    conn.close()

# Load encodings and metadata at startup
known_face_encodings, known_face_names, known_face_metadata = load_encodings()

def generate_frames():
    video_capture = cv2.VideoCapture(0)
    recognized_unknown_faces = set()
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            phone = ""

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                face_id, _, _ = known_face_metadata[first_match_index]
                log_recognition(face_id, name, is_known=True)
            else:
                face_encoding_tuple = tuple(face_encoding)
                if face_encoding_tuple not in recognized_unknown_faces:
                    if not is_unknown_face_in_database(face_encoding):
                        recognized_unknown_faces.add(face_encoding_tuple)
                        save_face('Unknown', 'Unknown', face_encoding, frame)
                        log_recognition(None, 'Unknown', is_known=False)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({phone})", (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()
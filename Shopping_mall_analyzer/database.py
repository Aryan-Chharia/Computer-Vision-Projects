import sqlite3

# Connect to the database (or create it if it doesn't exist)
conn = sqlite3.connect('faces.db')
conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign key constraints
c = conn.cursor()

# Create faces table
c.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        mobile_number_1 TEXT,
        mobile_number_2 TEXT,
        email_id TEXT,
        address TEXT,
        state TEXT,
        city TEXT,
        country TEXT,
        encoding BLOB,
        image BLOB
    )
''')

# Create recognition_logs table with a foreign key constraint
c.execute('''
    CREATE TABLE IF NOT EXISTS recognition_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        face_id INTEGER,
        name TEXT,
        date TEXT,
        start_time TEXT,
        total_time REAL,
        FOREIGN KEY (face_id) REFERENCES faces(id)
    )
''')

# Commit changes and close the connection
conn.commit()
conn.close()

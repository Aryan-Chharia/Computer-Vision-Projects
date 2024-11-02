from flask import Flask, request,jsonify,Response,render_template, redirect,session, url_for ,send_file, flash,logging
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email

import pandas as pd
import plotly.express as px
import plotly.io as pio
import bcrypt
import sqlite3
import secrets
import base64
import cv2
import io
import face_recognition
from io import BytesIO
import numpy as np
import os
import json
from PIL import Image
from datetime import datetime
import pickle
from model import save_face,load_encodings,generate_frames
# Initialize the Flask application


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.secret_key = 'secret_key'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()


# Function to fetch faces from the database
def fetch_faces(sort_order='all'):
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    if sort_order == 'latest':
        c.execute('SELECT id, name FROM faces ORDER BY id DESC')
    else:
        c.execute('SELECT id, name FROM faces ORDER BY id ASC')
    faces = c.fetchall()
    conn.close()
    return faces

def fetch_data(state):
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute('SELECT city, COUNT(*) FROM faces WHERE state = ? GROUP BY city', (state,))
    data = c.fetchall()
    conn.close()
    return data

def plot_charts(logs):
    df = pd.DataFrame(logs, columns=['date', 'total_time'])
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.strftime('%b')
    df['year'] = df['date'].dt.year

    # Line chart for total hours by day
    day_line = df.groupby(df['date'].dt.date)['total_time'].sum().reset_index()
    day_line_fig = px.line(day_line, x='date', y='total_time', title='Total Hours by Day', labels={'date': 'Date', 'total_time': 'Total Hours'})
    day_line_html = pio.to_html(day_line_fig, full_html=False)

    # Pie chart for total hours by month
    month_pie = df.groupby('month')['total_time'].sum().reset_index()
    month_pie_fig = px.pie(month_pie, values='total_time', names='month', title='Total Hours by Month')
    month_pie_html = pio.to_html(month_pie_fig, full_html=False)

    # Pie chart for total hours by year
    year_pie = df.groupby('year')['total_time'].sum().reset_index()
    year_pie_fig = px.pie(year_pie, values='total_time', names='year', title='Total Hours by Year')
    year_pie_html = pio.to_html(year_pie_fig, full_html=False)

    # Line plot for month-wise total hours
    month_line = df.groupby(df['date'].dt.strftime('%Y-%m'))['total_time'].sum().reset_index()
    month_line['month'] = pd.to_datetime(month_line['date']).dt.strftime('%b')
    month_line_fig = px.line(month_line, x='date', y='total_time', title='Month-wise Total Hours', labels={'date': 'Date', 'total_time': 'Total Hours'})
    month_line_html = pio.to_html(month_line_fig, full_html=False)

    return day_line_html, month_pie_html, year_pie_html, month_line_html


def fetch_recognition_logs(face_id):
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute('SELECT date, total_time FROM recognition_logs WHERE face_id = ?', (face_id,))
    logs = c.fetchall()
    conn.close()
    return logs

def fetch_image(face_id):
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute('SELECT image FROM faces WHERE id = ?', (face_id,))
    data = c.fetchone()
    conn.close()
    if data:
        image_data = data[0]
        image = Image.open(BytesIO(image_data))
        return image
    else:
        return None









# Define a route for the root URL ("/")
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid user')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect('/login')
    
    user = User.query.filter_by(email=session['email']).first()
    
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute('SELECT DISTINCT state FROM faces')
    states = [row[0] for row in c.fetchall()]
    conn.close()

    return render_template('dashboard.html', user=user, states=states)

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/')


@app.route('/faces', methods=['GET'])
def display_faces():
    try:
        sort_order = request.args.get('sort_order', 'all')  # default to 'all' if not provided
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        if sort_order == 'latest':
            c.execute('SELECT id, name, mobile_number_1, city FROM faces ORDER BY id DESC')
        else:
            c.execute('SELECT id, name, mobile_number_1, city FROM faces ORDER BY id ASC')
        faces = c.fetchall()
        conn.close()
        return render_template('faces.html', faces=faces, sort_order=sort_order)
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return render_template('faces.html', faces=[], sort_order='all')





@app.route('/delete/<int:face_id>', methods=['POST'])
def delete_face(face_id):
    try:
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute('DELETE FROM faces WHERE id = ?', (face_id,))
        conn.commit()
        conn.close()
        flash('Face deleted successfully.', 'success')
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
    return redirect(url_for('display_faces'))

@app.route('/modify/<int:face_id>')
def modify_face(face_id):
    try:
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute('SELECT * FROM faces WHERE id = ?', (face_id,))
        face = c.fetchone()
        conn.close()
        if face:
            return render_template('modify.html', face=face)
        else:
            flash('Face not found.', 'error')
            return redirect(url_for('display_faces'))
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('display_faces'))

@app.route('/update/<int:face_id>', methods=['POST'])
def update_face(face_id):
    try:
        name = request.form['name']
        mobile_number_1 = request.form['phone1']
        mobile_number_2 = request.form['phone2']
        email_id = request.form['email']
        address = request.form['address']
        state = request.form['state']
        city = request.form['city']
        country = request.form['country']
        
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute('''UPDATE faces 
                     SET name = ?, mobile_number_1 = ?, mobile_number_2 = ?, email_id = ?, 
                         address = ?, state = ?, city = ?, country = ? 
                     WHERE id = ?''', 
                  (name, mobile_number_1, mobile_number_2, email_id, address, state, city, country, face_id))
        conn.commit()
        conn.close()
        flash('Face updated successfully.', 'success')
        return redirect(url_for('display_faces'))
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('modify_face', face_id=face_id))

@app.route('/image/<int:face_id>')
def serve_image(face_id):
    try:
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute('SELECT image FROM faces WHERE id = ?', (face_id,))
        image_data = c.fetchone()
        conn.close()
        
        if image_data and image_data[0]:
            return send_file(
                io.BytesIO(image_data[0]),
                mimetype='image/jpeg',
                as_attachment=False,
                download_name=f'face_{face_id}.jpg'
            )
        else:
            flash('Image not found.', 'error')
            return redirect(url_for('display_faces'))
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('display_faces'))




@app.route('/update_chart', methods=['POST'])
def update_chart():
    selected_state = request.json.get('state')
    data = fetch_data(selected_state)
    df = pd.DataFrame(data, columns=['City', 'Count'])
    fig = px.pie(df, values='Count', names='City')
    fig.update_traces(textinfo='value', textfont_size=14)
    fig.update_layout(title_text='')
    graph_html = pio.to_html(fig, full_html=False)
    return jsonify({'graph_html': graph_html})

@app.route('/analyze', methods=['GET'])
def analyze():
    sort_order = request.args.get('sort_order', 'all')
    faces = fetch_faces(sort_order)
    return render_template('analyze.html', faces=faces, sort_order=sort_order)

@app.route('/analyze/<int:face_id>', methods=['GET'])
def analyze_face(face_id):
    logs = fetch_recognition_logs(face_id)
    if logs:
        day_line_html, month_pie_html, year_pie_html, month_line_html = plot_charts(logs)
        return render_template('face_modal.html', face_id=face_id, day_line_html=day_line_html, month_pie_html=month_pie_html, year_pie_html=year_pie_html, month_line_html=month_line_html)
    else:
        flash('No recognition logs found for the selected face.', 'warning')
        return redirect(url_for('analyze'))


@app.route('/recognize')
def recognize():
    return render_template('recogonize.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



    






# Run the application
if __name__ == '__main__':
    app.run(debug=True)

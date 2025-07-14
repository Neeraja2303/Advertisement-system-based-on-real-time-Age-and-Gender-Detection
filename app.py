import cv2
import numpy as np
import threading
import json
import time
import os
import sqlite3
import logging
from flask import Flask, render_template, Response, jsonify, request
from datetime import datetime, timedelta
from io import StringIO
import csv
from utils.prediction_utils import predict_age_gender

# Configure logging to file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

# Configurations
DB_PATH = "advertisement.db"
ADS_JSON_PATH = "ads.json"
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
FRAME_SKIP = 3

# Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Failed to open webcam")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

# Global States
lock = threading.Lock()
is_detecting = False
detection_thread = None
current_ad = "static/default_ad.jpg"
current_gender = None
current_age = None
last_detections = {}

# Ensure database exists
def init_db():
    try:
        conn = sqlite3.sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS demographics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gender TEXT,
                age TEXT,
                timestamp TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_demographics ON demographics (gender, age, timestamp)")
        conn.commit()
        logging.info("Database and index initialized successfully")
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
    finally:
        conn.close()

# Load ads from JSON
def load_ads():
    try:
        with open(ADS_JSON_PATH, 'r') as f:
            ads = json.load(f)
            logging.debug(f"Loaded {len(ads)} ads from {ADS_JSON_PATH}")
            return ads
    except FileNotFoundError:
        logging.error(f"{ADS_JSON_PATH} not found")
        return []
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in {ADS_JSON_PATH}")
        return []

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        logging.error("Failed to load haarcascade_frontalface_default.xml")
        return []
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def save_demographics(gender, age):
    with lock:
        detection_key = f"{gender}_{age}"
        current_time = datetime.now()
        if detection_key in last_detections:
            if (current_time - last_detections[detection_key]).total_seconds() < 2:
                logging.debug(f"Skipped duplicate detection: {detection_key}")
                return
        last_detections[detection_key] = current_time

        for key in list(last_detections.keys()):
            if (current_time - last_detections[key]).total_seconds() > 3600:
                del last_detections[key]

        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO demographics (gender, age, timestamp) VALUES (?, ?, ?)", (gender, age, timestamp))
            conn.commit()
            logging.debug(f"Saved demographics: gender={gender}, age={age}, timestamp={timestamp}")
        except sqlite3.Error as e:
            logging.error(f"Error saving demographics: {e}")
        finally:
            conn.close()

def detect():
    global current_gender, current_age
    frame_count = 0
    while is_detecting:
        success, frame = cap.read()
        if not success:
            logging.warning("Failed to read frame from webcam")
            continue

        if frame_count % FRAME_SKIP == 0:
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue
                try:
                    gender, age = predict_age_gender(face_img)
                    with lock:
                        current_gender = gender
                        current_age = age
                    logging.debug(f"Detected: gender={gender}, age={age}")
                    save_demographics(gender, age)

                    color = (255, 0, 0) if gender == "male" else (147, 20, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    label = f"{gender}, {age}"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                except Exception as e:
                    logging.error(f"Error in predict_age_gender: {e}")
                    continue

        frame_count += 1
        time.sleep(0.1)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            logging.warning("Failed to read frame for video feed")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global is_detecting, detection_thread
    data = request.get_json()
    is_detecting = data.get('detecting', False)
    if is_detecting:
        detection_thread = threading.Thread(target=detect)
        detection_thread.start()
    logging.debug(f"Detection toggled: is_detecting={is_detecting}")
    return jsonify({'success': True, 'detecting': is_detecting})

@app.route('/get_demographic')
def get_demographic():
    with lock:
        data = {"gender": current_gender, "age": current_age}
        logging.debug(f"Returning demographic: {data}")
        return jsonify(data)

@app.route('/get_counts')
def get_counts():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        query = "SELECT gender, age, COUNT(*) FROM demographics GROUP BY gender, age"
        logging.debug(f"Executing query: {query}")
        c.execute(query)
        rows = c.fetchall()
        data = {"male": {}, "female": {}}
        for gender, age, count in rows:
            if gender in data:
                data[gender][age] = count
            else:
                logging.warning(f"Unexpected gender value: {gender}")
        conn.close()
        logging.debug(f"get_counts returned: {data}")
        return jsonify(data)
    except sqlite3.Error as e:
        logging.error(f"Error in get_counts: {e}")
        return jsonify({"male": {}, "female": {}}), 200

@app.route('/get_ad')
def get_ad():
    global current_gender, current_age
    with lock:
        gender = current_gender
        age = current_age
    if gender and age:
        ads = load_ads()
        for ad in ads:
            if ad["gender"].lower() == gender.lower() and ad["age"] == age:
                ad_image = ad["image_path"]
                if os.path.exists(ad_image):
                    logging.debug(f"Selected ad: {ad_image} for {gender}, {age}")
                    return jsonify({"ad_image": ad_image})
                else:
                    logging.warning(f"Ad image not found: {ad_image}")
        logging.warning(f"No ad found for {gender}, {age}")
    else:
        logging.debug("No demographic detected, returning default ad")
    ad_image = "static/default_ad.jpg"
    return jsonify({"ad_image": ad_image})

@app.route('/status')
def status():
    global current_gender, current_age
    with lock:
        no_face = current_gender is None or current_age is None
    logging.debug(f"Status: no_face={no_face}")
    return jsonify({"no_face": no_face})

@app.route('/get_unique_count')
def get_unique_count():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        one_hour_ago = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        query = """
            SELECT COUNT(*) 
            FROM (
                SELECT DISTINCT gender, age, timestamp 
                FROM demographics 
                WHERE timestamp >= ?
            )
        """
        logging.debug(f"Executing query: {query} with params: {one_hour_ago}")
        c.execute(query, (one_hour_ago,))
        count = c.fetchone()[0]
        conn.close()
        logging.debug(f"get_unique_count returned: {count}")
        return jsonify({"unique_visitors": count})
    except sqlite3.Error as e:
        logging.error(f"Error in get_unique_count: {e}")
        return jsonify({"unique_visitors": 0}), 200

@app.route('/filter_data')
def filter_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        query = "SELECT gender, age, COUNT(*) as count FROM demographics WHERE 1=1"
        params = []
        
        gender = request.args.get('gender')
        if gender:
            query += " AND gender = ?"
            params.append(gender)
        age = request.args.get('age')
        if age:
            query += " AND age = ?"
            params.append(age)
        start_date = request.args.get('start_date')
        if start_date:
            query += " AND timestamp >= ?"
            params.append(f"{start_date} 00:00:00")
        end_date = request.args.get('end_date')
        if end_date:
            query += " AND timestamp <= ?"
            params.append(f"{end_date} 23:59:59")
        
        query += " GROUP BY gender, age"
        
        logging.debug(f"Executing query: {query} with params: {params}")
        c.execute(query, params)
        rows = [{"gender": row[0], "age": row[1], "count": row[2]} for row in c.fetchall()]
        conn.close()
        logging.debug(f"filter_data returned: {rows}")
        return jsonify(rows)
    except sqlite3.Error as e:
        logging.error(f"Error in filter_data: {e}")
        return jsonify([]), 200

@app.route('/export_csv')
def export_csv():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        query = "SELECT gender, age, COUNT(*) as count FROM demographics WHERE 1=1"
        params = []
        
        gender = request.args.get('gender')
        if gender:
            query += " AND gender = ?"
            params.append(gender)
        age = request.args.get('age')
        if age:
            query += " AND age = ?"
            params.append(age)
        start_date = request.args.get('start_date')
        if start_date:
            query += " AND timestamp >= ?"
            params.append(f"{start_date} 00:00:00")
        end_date = request.args.get('end_date')
        if end_date:
            query += " AND timestamp <= ?"
            params.append(f"{end_date} 23:59:59")
        
        query += " GROUP BY gender, age"
        
        logging.debug(f"Executing query: {query} with params: {params}")
        c.execute(query, params)
        rows = c.fetchall()
        conn.close()
        
        si = StringIO()
        cw = csv.writer(si)
        cw.writerow(['Gender', 'Age', 'Count'])
        cw.writerows(rows)
        
        output = si.getvalue()
        si.close()
        
        logging.debug("CSV exported successfully")
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=demographics.csv"}
        )
    except sqlite3.Error as e:
        logging.error(f"Error in export_csv: {e}")
        return jsonify({"error": "Database error"}), 500

@app.route('/debug_db')
def debug_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM demographics LIMIT 10")
        rows = c.fetchall()
        conn.close()
        data = [{"id": row[0], "gender": row[1], "age": row[2], "timestamp": row[3]} for row in rows]
        logging.debug(f"debug_db returned: {data}")
        return jsonify(data)
    except sqlite3.Error as e:
        logging.error(f"Error in debug_db: {e}")
        return jsonify({"error": "Database error"}), 500

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        cap.release()
import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import face_recognition
from datetime import datetime, date
import pandas as pd
import json
import time

DB_PATH = r"D:\intern\l&t\task 6 - face recognization\mine\face_db\images"
ATTENDANCE_FILE = 'attendance_data.json'
THRESHOLD = 0.6
MIN_CONFIDENCE = 0.5
ATTENDANCE_COLUMNS = ['Name', 'Date', 'Time', 'Confidence']

if 'attendance' not in st.session_state:
    st.session_state.attendance = pd.DataFrame(columns=ATTENDANCE_COLUMNS)
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = {'total': 0, 'correct': 0}
if 'db_embeddings' not in st.session_state or 'db_names' not in st.session_state:
    st.session_state.db_embeddings = []
    st.session_state.db_names = []

def save_attendance():
    clean_attendance = st.session_state.attendance.copy()
    clean_attendance['Confidence'] = clean_attendance['Confidence'].fillna(0.0)
    clean_attendance = clean_attendance.where(pd.notnull(clean_attendance), None)
    data = {'attendance': clean_attendance.to_dict(orient='records')}
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(data, f, indent=4, default=str)

def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data.get('attendance', []))
            for col in ATTENDANCE_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce').fillna(0.0)
            df['Name'] = df['Name'].fillna("Unknown")
            df['Date'] = df['Date'].fillna(datetime.now().date())
            df['Time'] = df['Time'].fillna(datetime.now().time())
            st.session_state.attendance = df[ATTENDANCE_COLUMNS]
        except (json.JSONDecodeError, KeyError, ValueError):
            st.session_state.attendance = pd.DataFrame(columns=ATTENDANCE_COLUMNS)
    else:
        st.session_state.attendance = pd.DataFrame(columns=ATTENDANCE_COLUMNS)

@st.cache_data
def load_all_face_encodings():
    embeddings, names = [], []
    if not os.path.exists(DB_PATH):
        return embeddings, names
    for entry in os.listdir(DB_PATH):
        entry_path = os.path.join(DB_PATH, entry)
        if os.path.isdir(entry_path):
            for img_file in os.listdir(entry_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(entry_path, img_file)
                    try:
                        image = face_recognition.load_image_file(img_path)
                        encs = face_recognition.face_encodings(image)
                        if encs:
                            embeddings.append(encs[0])
                            names.append(entry)
                    except Exception:
                        pass
        elif entry.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = entry_path
            try:
                image = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    name = os.path.splitext(entry)[0]
                    embeddings.append(encs[0])
                    names.append(name)
            except Exception:
                pass
    return embeddings, names

def match_face(test_embedding, threshold=THRESHOLD, min_conf=MIN_CONFIDENCE):
    if not st.session_state.db_embeddings:
        return "Unknown", 0.0
    distances = np.linalg.norm(np.array(st.session_state.db_embeddings) - test_embedding, axis=1)
    idx = np.argmin(distances)
    min_distance = distances[idx]
    confidence = 1 - (min_distance / np.sqrt(2))
    name = st.session_state.db_names[idx] if min_distance < threshold and confidence >= min_conf else "Unknown"
    return name, confidence

def is_attendance_marked_today(name):
    today = date.today().strftime('%Y-%m-%d')
    if not st.session_state.attendance.empty:
        return ((st.session_state.attendance['Name'] == name) & 
                (st.session_state.attendance['Date'] == today)).any()
    return False

def mark_attendance(name, confidence):
    if not is_attendance_marked_today(name):
        now = datetime.now()
        new_entry = pd.DataFrame([[
            name,
            now.strftime('%Y-%m-%d'),
            now.strftime('%H:%M:%S'),
            confidence
        ]], columns=ATTENDANCE_COLUMNS)
        st.session_state.attendance = pd.concat([st.session_state.attendance, new_entry], ignore_index=True)
        save_attendance()
        return True
    return False

with st.spinner("Loading face database..."):
    st.session_state.db_embeddings, st.session_state.db_names = load_all_face_encodings()
load_attendance()

st.title("Face Recognition Attendance System")

total = st.session_state.accuracy['total']
correct = st.session_state.accuracy['correct']
accuracy = (correct / total * 100) if total > 0 else 0
st.sidebar.metric("Recognition Accuracy", f"{accuracy:.1f}%")
st.sidebar.write(f"Total Attempts: {total}")
st.sidebar.write(f"Correct Recognitions: {correct}")

with st.expander("Database Status"):
    st.write(f"Loaded encodings: {len(st.session_state.db_names)}")
    if st.session_state.db_names:
        unique_people = list(set(st.session_state.db_names))
        st.write("People in database:", ", ".join(unique_people))
    if st.button("Refresh Encodings"):
        st.session_state.db_embeddings, st.session_state.db_names = load_all_face_encodings()
        st.rerun()

st.header("Register New Person")
person_name = st.text_input("Enter name for registration:")
num_images = st.number_input("Number of images to capture", min_value=1, value=30)

if st.button("Capture Images") and person_name:
    person_folder = os.path.join(DB_PATH, person_name)
    os.makedirs(person_folder, exist_ok=True)
    cap = cv2.VideoCapture(0)
    captured = 0
    stframe = st.empty()
    progress_bar = st.progress(0)
    while captured < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(person_folder, f"{person_name}_{captured+1:02d}.jpg")
        cv2.imwrite(filename, frame)
        stframe.image(frame, channels="BGR", caption=f"Captured Image {captured+1}/{num_images}")
        captured += 1
        progress_bar.progress(captured/num_images)
        time.sleep(0.3)
    cap.release()
    st.session_state.db_embeddings, st.session_state.db_names = load_all_face_encodings()
    st.rerun()

st.header("Recognize from Uploaded Image")
uploaded_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    st.image(image, use_column_width=True)
    encs = face_recognition.face_encodings(img_np)
    if encs:
        emb = encs[0]
        name, confidence = match_face(emb)
        st.session_state.accuracy['total'] += 1
        if name != "Unknown":
            st.session_state.accuracy['correct'] += 1
            if mark_attendance(name, confidence):
                st.success(f"Attendance marked for {name} (Confidence: {confidence:.2f})")
            else:
                st.info(f"{name} already marked today")
        else:
            st.warning(f"Unknown person (Confidence: {confidence:.2f})")
    else:
        st.warning("No face detected")

st.header("Real-Time Recognition")
if st.button("Start Webcam Recognition"):
    cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    result_placeholder = st.empty()
    stop_recognition = False

    while cap.isOpened() and not stop_recognition:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name, confidence = match_face(face_encoding)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{name} {confidence*100:.1f}%", (left+6, bottom-6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if name != "Unknown":
                if mark_attendance(name, confidence):
                    result_placeholder.success(f"Attendance marked for {name}")
                else:
                    result_placeholder.info(f"{name} already marked today")
                stop_recognition = True
                break
            else:
                result_placeholder.warning("Unknown person detected")
                stop_recognition = True
                break
        st_frame.image(frame, channels="BGR")
        if stop_recognition:
            time.sleep(2)
            break
    cap.release()

st.header("Attendance Records")
st.dataframe(st.session_state.attendance)

col1, col2 = st.columns(2)
with col1:
    if st.button("Clear Records"):
        st.session_state.attendance = pd.DataFrame(columns=ATTENDANCE_COLUMNS)
        save_attendance()
with col2:
    st.download_button(
        label="Download CSV",
        data=st.session_state.attendance.to_csv(index=False),
        file_name="attendance.csv",
        mime="text/csv"
    )

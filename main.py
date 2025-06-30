# --- main.py ---
import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from datetime import datetime
import joblib
import csv
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import TFSMLayer
import dlib

# Load models
embedding_model = TFSMLayer("models\embedding_model\embedding_model_savedmodel", call_endpoint="serving_default")
knn_model = joblib.load("models/knn_model.pkl")

# Inisialisasi absensi
absen_tercatat = set()
tanggal_hari_ini = datetime.now().strftime("%Y-%m-%d")
folder_absen = f"absen/{tanggal_hari_ini}"
os.makedirs(folder_absen, exist_ok=True)
csv_path = f"absen/{tanggal_hari_ini}.csv"

if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Nama", "Waktu", "Status"])

# Preprocessing image
def preprocess_frame(frame):
    img = cv2.resize(frame, (220, 220))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Catat absensi
def catat_absensi(nama, frame):
    if nama in absen_tercatat:
        return
    waktu = datetime.now().strftime("%H:%M:%S")
    filename = os.path.join(folder_absen, f"{nama}_{waktu.replace(':', '-')}.jpg")
    cv2.imwrite(filename, frame)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([nama, waktu, "Hadir"])
    absen_tercatat.add(nama)
    app.update_absen_list(nama, waktu)

# Dlib face detector
detector = dlib.get_frontal_face_detector()

# GUI
class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Absensi Wajah")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        self.title_label = tk.Label(root, text="Sistem Absensi Berbasis Face Recognition", 
                                    font=("Helvetica", 18, "bold"), bg="#f0f0f0")
        self.title_label.pack(pady=10)

        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill="both", expand=True)

        self.left_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.left_frame.pack(side="left", padx=10, pady=10)

        self.video_frame = tk.Label(self.left_frame, bg="#d0d0d0", bd=2, relief="ridge")
        self.video_frame.pack()

        self.button_frame = tk.Frame(self.left_frame, bg="#f0f0f0")
        self.button_frame.pack(pady=10)

        self.start_button = tk.Button(self.button_frame, text="Start Camera", width=15,
                                      command=self.start_camera, bg="#4caf50", fg="white", font=("Helvetica", 12, "bold"))
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = tk.Button(self.button_frame, text="Stop Camera", width=15,
                                     command=self.stop_camera, bg="#f44336", fg="white", font=("Helvetica", 12, "bold"))
        self.stop_button.grid(row=0, column=1, padx=10)

        self.right_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.right_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.absen_label = tk.Label(self.right_frame, text="Daftar Hadir", font=("Helvetica", 14, "bold"), bg="#f0f0f0")
        self.absen_label.pack()

        self.absen_listbox = tk.Listbox(self.right_frame, width=40, height=20, font=("Helvetica", 12))
        self.absen_listbox.pack(pady=5)

        self.total_label = tk.Label(self.right_frame, text="Total Hadir: 0", font=("Helvetica", 12), bg="#f0f0f0")
        self.total_label.pack()

        self.cap = None
        self.running = False

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def stop_camera(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
            self.video_frame.config(image='')

    def update_frame(self):
        if self.cap and self.running:
            ret, frame = self.cap.read()
            if not ret:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                x, y = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                w, h = x2 - x, y2 - y

                face_crop = frame[y:y+h, x:x+w]

                try:
                    face_input = preprocess_frame(face_crop)
                    output = embedding_model(face_input)
                    vector = output['output_0'].numpy()
                    prediction = knn_model.predict(vector)[0]
                    catat_absensi(prediction, face_crop)

                    cv2.putText(frame, f"{prediction}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                except Exception as e:
                    print("Gagal memproses wajah:", e)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

            self.root.after(10, self.update_frame)

    def update_absen_list(self, nama, waktu):
        self.absen_listbox.insert(tk.END, f"{nama} - {waktu}")
        self.total_label.config(text=f"Total Hadir: {len(absen_tercatat)}")

# Run app
if __name__ == "__main__":
    os.makedirs("absen", exist_ok=True)
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()

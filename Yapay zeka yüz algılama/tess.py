import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QInputDialog
from PyQt6.QtCore import Qt
import cv2
import face_recognition
import pickle
import os
from qdarkstyle import load_stylesheet_pyqt6

class FaceRecognitionApp(QWidget):
    def __init__(self, users, filename):
        super().__init__()

        self.users = users
        self.filename = filename

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Yüz algılama uygulaması')
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        register_button = QPushButton('Kullanıcı kayıtı yap', self)
        register_button.clicked.connect(self.register_user)
        layout.addWidget(register_button)

        recognize_button = QPushButton('Yüzü tanınmış kullanıcıları gör', self)
        recognize_button.clicked.connect(self.recognize_users)
        layout.addWidget(recognize_button)

        self.setLayout(layout)

    def register_user(self):
        user_name, ok = QInputDialog.getText(self, 'User Registration', 'Kullanıcı Adınızı Girin')
        if ok:
            self.users = register_user(self.users, self.filename, user_name)

    def recognize_users(self):
        recognize_user(self.users)

def load_users(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

def save_users(users, filename):
    with open(filename, 'wb') as f:
        pickle.dump(users, f)

def register_user(users, filename, user_name):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Register User - Press Space to Capture', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            user_image = frame
            break

    cap.release()
    cv2.destroyAllWindows()

    face_locations = face_recognition.face_locations(user_image)
    if face_locations:
        users[user_name] = face_recognition.face_encodings(user_image)[0]
        save_users(users, filename)
        print(f"{user_name} kullanıcı başarıyla kaydedildi.")
    else:
        print("Error: Yüz bulunamadı.Lütfen tekrar deneyin.")

    return users

def recognize_user(users):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            for user_name, user_face_encoding in users.items():
                match = face_recognition.compare_faces([user_face_encoding], face_encoding)
                if match[0]:
                    name = f"Recognized: {user_name}"
                    break

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet_pyqt6())

    filename = "users.pkl"
    users = load_users(filename)

    face_recognition_app = FaceRecognitionApp(users, filename)
    face_recognition_app.show()
    sys.exit(app.exec())
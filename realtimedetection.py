import cv2
from keras.models import model_from_json
import numpy as np
import webbrowser

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.keras")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

webcam = cv2.VideoCapture(0)

first_emotion = None

while True:
    ret, frame = webcam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        if len(faces) > 0 and first_emotion is None:
            (x, y, w, h) = faces[0]  
            face_image = gray[y:y + h, x:x + w]

            face_image = cv2.resize(face_image, (48, 48))

            face_features = extract_features(face_image)

            prediction = model.predict(face_features)
            predicted_label = labels[prediction.argmax()]

            first_emotion = predicted_label

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, predicted_label, (x - 10, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    except cv2.error:
        pass 

    cv2.imshow("Emotion Detection", frame)

    if (first_emotion is not None or cv2.waitKey(1) & 0xFF == 27):
        break

webcam.release()
cv2.destroyAllWindows()

print("First detected emotion:", first_emotion)

x =  first_emotion;
url = f"https://web-omega-blue.vercel.app/{x}";

webbrowser.open_new(url)
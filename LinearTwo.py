import cv2
import glob
import face_recognition
import os

known_faces = []
known_names = []
known_faces_paths = []

registered_faces_path = 'C:\\Users\\aelne\\Desktop\\project_linear_code_images\\Registered/'

for name in os.listdir(registered_faces_path):
        images_mask = '%s%s/*.jpg' % (registered_faces_path, name)
        images_paths = glob.glob(images_mask)
        known_faces_paths += images_paths
        known_names += [name for x in images_paths]



def get_encodings(img_path):
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) > 0:
        encoding = face_recognition.face_encodings(image, face_locations)[0]
        return encoding
    else:
        return None

known_faces = [get_encodings(img_path) for img_path in known_faces_paths]
known_faces = [face_encoding for face_encoding in known_faces if face_encoding is not None]


vc = cv2.VideoCapture(0)

while True:
        ret, frame = vc.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(frame_rgb)
        for face in faces:
            top, right, bottom, left = face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            encoding = face_recognition.face_encodings(frame_rgb, [face])[0]

            results = face_recognition.compare_faces(known_faces, encoding)
            if any(results):
                name = known_names[results.index(True)]
            else:
                name = 'unknown'
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        cv2.imshow('win', frame)
        k = cv2.waitKey(1)
        if ord('q') == k:
            break

cv2.destroyAllWindows()
vc.release()


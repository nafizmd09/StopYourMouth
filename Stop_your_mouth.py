import cv2
import mediapipe as mp

cap = cv2.VideoCapture("mouth.mp4")

face_mesh = mp.solutions.face_mesh.FaceMesh()

up_lip_y_axis = 0
low_lip_y_axis = 0

# Meta.
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Video writer.
video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)


while True:
    _, img = cap.read()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_img)
    face_landmarks = output.multi_face_landmarks
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_h, img_w, _ = img.shape
    # print(face_landmarks)

    if face_landmarks:
        landmarks = face_landmarks[0].landmark
        for id, landmark in enumerate(landmarks):
            x = int(landmark.x * img_w)
            y = int(landmark.y * img_h)
            # print((x, y))

            if id == 0:
                up_lip_y_axis = y
                cv2.circle(img, (x, y), 3, (55, 198, 160))

            if id == 14:
                low_lip_y_axis = y
                cv2.circle(img, (x, y), 3, (226, 96, 31))

            if (low_lip_y_axis - up_lip_y_axis) > 5:
                print('Mouth open')
                cv2.putText(img, 'Mouth open', (35, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 0), 1,
                            cv2.LINE_AA)

            else:
                print('Mouth close')
                cv2.putText(img, 'Mouth close', (425, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 50), 1,
                            cv2.LINE_AA)

    video_output.write(img)

    cv2.imshow("STOP YOUR MOUTH", img)
    cv2.waitKey(10)

cap.release()
cv2.destroyAllWindows()
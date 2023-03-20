import cv2
import mediapipe as mp
import time

class face_mesh_detector():

    def __init__(self, static_mode=False, max_faces=2, refine_landmarks=False, min_detect_conf=0.5, min_track_conf=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks
        self.min_detect_conf = min_detect_conf
        self.min_track_conf = min_track_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_mode, self.max_faces, self.refine_landmarks,
                                                 self.min_detect_conf, self.min_track_conf)

        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def find_face_mesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                  self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
                    #print(id, x, y)
                    face.append([id,x,y])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0) #"2.mp4"
    pTime = 0
    detector = face_mesh_detector()

    while True:
        success, img = cap.read()
        img, faces = detector.find_face_mesh(img, True)
        if len(faces)!=0:
            print(faces[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Image", img)

        cv2.waitKey(20)


if __name__ == "__main__":
    main()
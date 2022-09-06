import cv2 as cv

eye_cascade = cv.CascadeClassifier("./cascades/haarcascade_eye.xml")
face_cascade = cv.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")


video_capture = cv.VideoCapture(0)
video_capture1 = cv.VideoCapture("./resources/video.mp4")
image = cv.imread("./resources/image.jpg")


class FaceDetector:
    def __init__(self, video):
        self.video = video

    def draw_rect_faces(self, frame, faces, gray):
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y : y + h, x : x + w]
            roi_colour = frame[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_colour, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)

    def face_detector(self):
        while True:
            ret, frame = self.video.read()
            try:
                frame = cv.flip(frame, 1)
                frame = cv.resize(frame, (800, 800))
            except:
                break

            if not ret:
                print("Can't receive signal. Exiting")
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
            )

            self.draw_rect_faces(frame, gray, faces)

            cv.imshow("Video", frame)
            self.quit_from_button()

        self.video.release()

    def quit_from_button(self):
        if cv.waitKey(1) & 0xFF == ord("q"):
            return


def draw_rect_faces(img, gray, faces):
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_colour = img[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_colour, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)


def face_detector_video(video):
    while True:
        ret, frame = video.read()
        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, (800, 800))

        if not ret:
            print("Can't receive signal. Exiting")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
        )

        draw_rect_faces(frame, gray, faces)

        cv.imshow("Video", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            return

    video.release()


def face_detector_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    draw_rect_faces(image, gray, faces)

    cv.imshow("img", image)
    cv.waitKey(0)


def main():
    face = FaceDetector(video_capture)
    face.face_detector()

    # face_detector_image(image)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

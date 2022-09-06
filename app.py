from flask import Flask, render_template, Response, redirect, url_for
import cv2 as cv

app = Flask(__name__)
camera = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")


@app.route("/")
def index():
    return render_template("index.html")


def generate_frames():
    while True:
        ret, frame = camera.read()
        frame = cv.flip(frame, 1)

        if not ret:
            print("Can't read frame")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break

        ret, buffer = cv.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/face-detection-video")
def face_detection_video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/enable")
def enable():
    return redirect(url_for("face_detection", enable=enable))


@app.route("/face-detection")
def face_detection():
    return render_template("face_detection.html", enable=False)


if __name__ == "__main__":
    app.run(debug=True)

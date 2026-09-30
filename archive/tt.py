from flask import Flask, Response, render_template_string, jsonify, send_from_directory
import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("best.pt")

# VIDEO STREAM FROM RASPBERRY PI
STREAM_URL = "http://192.168.1.109:5000/video_feed"
cap = cv2.VideoCapture(STREAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# FILES
csv_file = "detections_tracking_log_no_telem.csv"
screenshots_dir = "detection_screenshots_no_telem"
os.makedirs(screenshots_dir, exist_ok=True)

logged_track_ids = set()

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "track_id", "class", "confidence", "latitude", "longitude", "image"])


def get_gps():
    return 0, 0


def log_detection(track_id, cls_name, conf, annotated_frame):
    lat, lon = get_gps()
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_name = f"{safe_time}_{cls_name}_ID{track_id}.jpg"
    image_path = os.path.join(screenshots_dir, image_name)

    cv2.imwrite(image_path, annotated_frame)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time_now, track_id, cls_name, round(conf, 2), lat, lon, image_name])


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            continue

        results = model.track(
            frame,
            conf=0.25,
            imgsz=416,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        annotated = results[0].plot()

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue

            for box in r.boxes:
                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]

                unique_key = f"{cls_name}_{track_id}"

                if unique_key not in logged_track_ids:
                    logged_track_ids.add(unique_key)
                    log_detection(track_id, cls_name, conf, annotated)

        ret, buffer = cv2.imencode(".jpg", annotated)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Eagle Aid Dashboard</title>
        <style>
            body { font-family: Arial; background: #111; color: white; text-align: center; }
            img { width: 70%; border: 3px solid #00ff99; border-radius: 10px; }
            .cards { display: flex; justify-content: center; gap: 20px; margin: 20px; }
            .card { background: #222; border: 2px solid #00ff99; width: 200px; padding: 15px; font-size: 20px; border-radius: 10px; }
            .number { font-size: 34px; font-weight: bold; color: #00ff99; }
            table { margin: 20px auto; border-collapse: collapse; width: 98%; font-size: 14px; }
            th, td { border: 1px solid #555; padding: 8px; }
            th { background: #00aa77; }
            tr:nth-child(even) { background: #222; }
            a { color: #00ff99; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Eagle Aid Tracking Dashboard</h1>
        <h3>No Telemetry Mode</h3>

        <img src="/video_feed">

        <h2>Unique Detection Counters</h2>
        <div class="cards">
            <div class="card">Unique Victims<div class="number" id="victimCount">0</div></div>
            <div class="card">Unique Fire<div class="number" id="fireCount">0</div></div>
            <div class="card">Unique Damaged<div class="number" id="damageCount">0</div></div>
        </div>

        <h2>Tracked Detections</h2>
        <table id="detectionsTable"></table>

        <script>
            function loadData() {
                fetch('/data')
                    .then(response => response.json())
                    .then(data => {
                        let table = document.getElementById("detectionsTable");

                        table.innerHTML = `
                            <tr>
                                <th>Time</th>
                                <th>Track ID</th>
                                <th>Class</th>
                                <th>Confidence</th>
                                <th>Latitude</th>
                                <th>Longitude</th>
                                <th>Image</th>
                            </tr>
                        `;

                        let victimCount = 0;
                        let fireCount = 0;
                        let damageCount = 0;

                        data.rows.forEach(row => {
                            let cls = row[2].toLowerCase();

                            if (cls.includes("victim")) victimCount++;
                            if (cls.includes("fire")) fireCount++;
                            if (cls.includes("damage")) damageCount++;

                            let tr = document.createElement("tr");

                            row.forEach((cell, index) => {
                                let td = document.createElement("td");

                                if (index === 6) {
                                    let link = document.createElement("a");
                                    link.href = "/screenshots/" + cell;
                                    link.target = "_blank";
                                    link.innerText = "View";
                                    td.appendChild(link);
                                } else {
                                    td.innerText = cell;
                                }

                                tr.appendChild(td);
                            });

                            table.appendChild(tr);
                        });

                        document.getElementById("victimCount").innerText = victimCount;
                        document.getElementById("fireCount").innerText = fireCount;
                        document.getElementById("damageCount").innerText = damageCount;
                    });
            }

            setInterval(loadData, 2000);
            loadData();
        </script>
    </body>
    </html>
    """)


@app.route("/data")
def data():
    rows = []
    try:
        with open(csv_file, "r") as f:
            reader = list(csv.reader(f))
            rows = reader[1:]
            rows.reverse()
    except:
        pass

    return jsonify({"rows": rows})


@app.route("/screenshots/<filename>")
def screenshots(filename):
    return send_from_directory(screenshots_dir, filename)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
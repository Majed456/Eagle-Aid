"""Ground-station dashboard with one inference worker for all viewers."""
import argparse
import csv
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HEADER = ['time', 'track_id', 'class', 'confidence', 'latitude', 'longitude', 'image']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Pi stream URL, video path or camera index')
    parser.add_argument('--model', default=str(Path(__file__).resolve().parents[1] / 'models/best.pt'))
    parser.add_argument('--telemetry', help='Optional MAVLink endpoint, e.g. udp:127.0.0.1:14551')
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--output', default='outputs/sessions')
    args = parser.parse_args()
    if not Path(args.model).is_file():
        parser.error(f'Model not found: {args.model}')

    import cv2
    from flask import Flask, Response, jsonify, send_from_directory
    from ultralytics import YOLO

    logging.basicConfig(level=logging.INFO)
    model = YOLO(args.model)
    session = Path(args.output).resolve() / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    images = session / 'screenshots'
    images.mkdir(parents=True)
    csv_path = session / 'detections.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(HEADER)
    state = {'jpeg': None, 'sequence': 0, 'rows': [], 'status': 'Starting', 'gps': None}
    condition = threading.Condition()
    stop = threading.Event()
    logging.info('Session output: %s', session)

    def read_telemetry():
        connection = None
        try:
            from pymavlink import mavutil
            connection = mavutil.mavlink_connection(args.telemetry)
            while not stop.is_set():
                msg = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
                if msg:
                    lat, lon = msg.lat / 1e7, msg.lon / 1e7
                    if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat, lon) != (0, 0):
                        with condition:
                            state['gps'] = (lat, lon, time.monotonic())
        except Exception:
            logging.exception('Telemetry unavailable; detections continue without GPS')
        finally:
            if connection is not None:
                connection.close()

    def infer():
        source = int(args.source) if args.source.isdecimal() else args.source
        capture = cv2.VideoCapture(source)
        seen = set()
        try:
            if not capture.isOpened():
                raise RuntimeError(f'Cannot open video source: {args.source}')
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    with condition:
                        state['status'] = 'Source ended or disconnected; restart to reconnect'
                        condition.notify_all()
                    break
                result = model.track(frame, conf=args.conf, imgsz=args.imgsz, device=args.device,
                                     persist=True, tracker='bytetrack.yaml', verbose=False)[0]
                annotated = result.plot()
                if result.boxes is not None and result.boxes.id is not None:
                    for box in result.boxes:
                        track_id, class_id = int(box.id[0]), int(box.cls[0])
                        label = model.names[class_id]
                        key = (class_id, track_id)
                        if key in seen:
                            continue
                        now = datetime.now(timezone.utc)
                        safe_label = re.sub(r'[^A-Za-z0-9_-]', '_', label)
                        filename = f'{now:%Y%m%d_%H%M%S_%f}_{safe_label}_ID{track_id}.jpg'
                        if not cv2.imwrite(str(images / filename), annotated):
                            logging.warning('Could not save %s', filename)
                            continue
                        with condition:
                            gps = state['gps']
                        lat, lon = gps[:2] if gps and time.monotonic() - gps[2] <= 5 else ('', '')
                        row = [now.isoformat(), track_id, label, round(float(box.conf[0]), 4), lat, lon, filename]
                        with csv_path.open('a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerow(row)
                        seen.add(key)
                        with condition:
                            state['rows'].append(row)
                ok, buffer = cv2.imencode('.jpg', annotated)
                if ok:
                    with condition:
                        state['jpeg'] = buffer.tobytes()
                        state['sequence'] += 1
                        state['status'] = 'Running'
                        condition.notify_all()
        except Exception as exc:
            logging.exception('Inference stopped')
            with condition:
                state['status'] = f'Inference stopped: {exc}'
                condition.notify_all()
        finally:
            capture.release()
            stop.set()
            with condition:
                condition.notify_all()

    app = Flask(__name__)

    @app.get('/')
    def index():
        return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eagle-Aid</title><style>body{font:16px Arial;background:#111827;color:#eee;margin:24px}
img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #455}
a{color:#6ee7b7}.scroll{overflow:auto}</style><h1>Eagle-Aid Ground Station</h1>
<p id="status">Starting</p><p>Counts refer to tracker IDs in this session; objects can receive new IDs.</p>
<p>Coordinates are the drone position when logged, not the detected object's ground location.</p>
<p id="counts"></p><img src="/video_feed" alt="Detection stream"><div class="scroll"><table>
<thead><tr><th>Time (UTC)</th><th>ID</th><th>Class</th><th>Confidence</th><th>Lat</th><th>Lon</th><th>Image</th></tr></thead>
<tbody id="rows"></tbody></table></div><script>
async function refresh(){try{const d=await(await fetch('/data')).json();
document.getElementById('status').textContent=d.status;
const counts={};d.rows.forEach(r=>counts[r[2]]=(counts[r[2]]||0)+1);
document.getElementById('counts').textContent=Object.entries(counts).map(([k,v])=>k+': '+v).join(' | ');
const body=document.getElementById('rows');body.replaceChildren();
d.rows.slice().reverse().forEach(r=>{const tr=document.createElement('tr');r.forEach((v,i)=>{
const td=document.createElement('td');if(i===6){const a=document.createElement('a');a.href='/screenshots/'+encodeURIComponent(v);
a.textContent='View';a.target='_blank';td.append(a)}else{td.textContent=v===''?'Unavailable':v}tr.append(td)});body.append(tr)})
}catch(e){document.getElementById('status').textContent='Dashboard connection unavailable'}}
refresh();setInterval(refresh,2000);</script></html>'''

    @app.get('/data')
    def data():
        with condition:
            return jsonify(rows=list(state['rows']), status=state['status'])

    @app.get('/screenshots/<filename>')
    def screenshot(filename):
        return send_from_directory(images, filename)

    @app.get('/video_feed')
    def video_feed():
        def frames():
            sequence = -1
            while True:
                with condition:
                    condition.wait_for(lambda: state['sequence'] != sequence or stop.is_set(), timeout=2)
                    if state['sequence'] == sequence and stop.is_set():
                        break
                    jpeg = state['jpeg']
                    sequence = state['sequence']
                if jpeg:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
        return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    if args.telemetry:
        threading.Thread(target=read_telemetry, daemon=True).start()
    threading.Thread(target=infer, daemon=True).start()
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    finally:
        stop.set()
        with condition:
            condition.notify_all()


if __name__ == '__main__':
    main()

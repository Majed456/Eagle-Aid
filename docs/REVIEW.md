# Package review

## Verified from supplied files

- Six readable annotated JPEGs, three Python scripts, two CSVs and one checkpoint.
- Checkpoint class names: damaged_buildings, victims, fire.
- Embedded architecture identifier: yolov8n.pt; Ultralytics version 8.4.19.
- Embedded mAP50 0.82442 and mAP50–95 0.51448.
- Checkpoint metadata was inspected as pickle opcodes without deserializing the model.
- detections_tracking_log.csv: 22 rows (19 victims, 3 fire).
- detections_tracking_log_no_telem.csv: 2 rows (2 fire).
- All 24 log rows contain zero coordinates. None of the six supplied screenshot filenames match image references in the two CSVs. The CSVs and screenshots are separate examples; missing log screenshots are not fabricated.
- The inspected victim screenshot contains a hand/arm false positive; the damaged-building screenshot detects a photo on a phone. These are explicitly labeled demonstrations/failure examples.

## Preparation checks

- Python syntax compilation and CLI --help passed for the two new entrypoints.
- Mocked dashboard smoke check passed: two viewers share one inference worker, repeated track IDs produce one CSV row, and missing GPS stays empty. This does not replace real dependency/hardware validation.
- Original script/model/log/image bytes preserved in the organized copy.
- JPEG decoding and archive integrity checked.
- Runtime dependency installation and model inference were not run; neither the Raspberry Pi stream nor MAVLink was available. Multi-viewer and telemetry behavior still need runtime validation.

## Remaining source material

- Original Raspberry Pi camera streaming script and camera configuration.
- Training/merge scripts, original data.yaml, source dataset exports and exact split manifests.
- Validation plots/results.csv and raw, unannotated test images/videos.
- Flight demonstration video and evidence for real telemetry capture.

## Operational limitations

- Dataset metrics do not establish deployment reliability on unseen disaster scenes.
- Session tracker IDs are not permanent identities; counters may overcount objects after tracking loss.
- Drone coordinates are not object coordinates. Current telemetry freshness is checked, but camera/GPS timestamp synchronization and GPS fix-quality checks are not implemented.
- The organized dashboard stops at source disconnection; it does not automatically reconnect.
- Live model.track processing is synchronous within one worker; it does not guarantee minimal stream latency or an FPS target.
- Archive map variants rely on external Leaflet/OpenStreetMap resources. The new local dashboard shows coordinates in its table and does not reproduce the map.

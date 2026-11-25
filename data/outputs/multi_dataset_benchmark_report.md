# Multi-Dataset Tracking & Movement Benchmark Report

Evaluation of object-specific detection, tracking, speed, and trajectory analysis across distinct video domains.

## Comparative Performance Summary

| Dataset | Target Object | Tracked Frames | Coverage % | Total Dist (px) | Net Disp (px) | Mean Speed (px/s) | Peak Speed (px/s) | Path Efficiency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Traffic Scene (Vehicles)** | #2 (CAR) | 8 | 2.12% | 53.44 px | 53.36 px | 85.76 px/s | 109.06 px/s | 0.999 |
| **Pedestrian Walkway (Person)** | #2 (PERSON) | 56 | 9.4% | 550.61 px | 502.95 px | 120.69 px/s | 215.08 px/s | 0.913 |
| **Multi-Object Street Scene (Pedestrian / Car / Bike)** | #1 (PERSON) | 68 | 10.51% | 372.68 px | 335.22 px | 66.43 px/s | 173.41 px/s | 0.899 |
| **Aerial Drone Surveillance (Drone)** | #1 (PERSON) | 26 | 14.44% | 174.13 px | 150.77 px | 201.59 px/s | 653.56 px/s | 0.866 |


## Artifacts Generated

### Traffic Scene (Vehicles)
- **Annotated Video**: `tracked_id_2_traffic_cars.mp4`
- **Analytical Chart**: `tracking_dashboard_id_2_traffic_cars.png`
- **Telemetry Log**: `telemetry_id_2_traffic_cars.csv`
- **Individual Report**: `evaluation_report_id_2_traffic_cars.md`

### Pedestrian Walkway (Person)
- **Annotated Video**: `tracked_id_2_pedestrians_walk.mp4`
- **Analytical Chart**: `tracking_dashboard_id_2_pedestrians_walk.png`
- **Telemetry Log**: `telemetry_id_2_pedestrians_walk.csv`
- **Individual Report**: `evaluation_report_id_2_pedestrians_walk.md`

### Multi-Object Street Scene (Pedestrian / Car / Bike)
- **Annotated Video**: `tracked_id_1_multi_object_street.mp4`
- **Analytical Chart**: `tracking_dashboard_id_1_multi_object_street.png`
- **Telemetry Log**: `telemetry_id_1_multi_object_street.csv`
- **Individual Report**: `evaluation_report_id_1_multi_object_street.md`

### Aerial Drone Surveillance (Drone)
- **Annotated Video**: `tracked_id_1_drone_surveillance.mp4`
- **Analytical Chart**: `tracking_dashboard_id_1_drone_surveillance.png`
- **Telemetry Log**: `telemetry_id_1_drone_surveillance.csv`
- **Individual Report**: `evaluation_report_id_1_drone_surveillance.md`


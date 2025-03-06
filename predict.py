"""Inference entrypoint for YOLOv7 eddy detection.

This script keeps the original multi-mode behavior (single image/video/fps/folder),
while making the `dir_predict` workflow easier to configure and maintain.
"""

import datetime
import glob
import json
import os
import time
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch
from netCDF4 import Dataset
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm

from yolo import YOLO

# ----------------------------------------------------------------------------
# Project/domain constants
# ----------------------------------------------------------------------------
CLASS_NAMES = ["AE", "CE"]
GOCI_GRID_FILE = "img/2017071707CHL.nc"
DEFAULT_START_DATE = datetime.date(2011, 4, 1)
DEFAULT_END_DATE = datetime.date(2021, 3, 31)


def load_geo_grid(nc_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load longitude/latitude grid used to map pixel detections to geolocation."""
    data = Dataset(nc_file)
    lon = data.variables["lonl"][:]
    lat = data.variables["latl"][:]
    return lon, lat


def project_geo(point: Tuple[int, int], lon: np.ndarray, lat: np.ndarray) -> Tuple[float, float]:
    """Convert a pixel index (row, col) to (lon, lat)."""
    row, col = np.array(point, dtype=int)
    return float(lon[row, col]), float(lat[row, col])


def project_daypicture(
    boxes: np.ndarray,
    tile_name: str,
    patch_size: int,
    lon_shape: Tuple[int, int],
    eddy_radius_km: int = 100,
    resolution_m: int = 500,
) -> np.ndarray:
    """Project tile-local bbox coordinates to full-day mosaic pixel coordinates.

    tile_name follows `<date>_<row>_<col>.jpg`, where row/col are tile indices.
    """
    overlap = eddy_radius_km * 1000 / resolution_m / patch_size
    overlap_size = int(patch_size * overlap)

    # Parse tile row/column index from filename.
    _, row_str, col_ext = os.path.basename(tile_name).split("_")
    row = int(row_str)
    col = int(col_ext.split(".")[0])

    max_row_idx = int((lon_shape[0] - patch_size) / (patch_size - overlap_size)) + 1
    max_col_idx = int((lon_shape[1] - patch_size) / (patch_size - overlap_size)) + 1

    if row == max_row_idx:
        upper = lon_shape[0] - patch_size
    else:
        upper = row * (patch_size - overlap_size)

    if col == max_col_idx:
        left = lon_shape[1] - patch_size
    else:
        left = col * (patch_size - overlap_size)

    out = boxes.copy()
    out[:, [0, 2]] += upper
    out[:, [1, 3]] += left
    return out


def collect_dates(start_date: datetime.date, end_date: datetime.date) -> Iterable[datetime.date]:
    """Yield date range [start_date, end_date)."""
    delta = datetime.timedelta(days=1)
    for n in range((end_date - start_date).days):
        yield start_date + delta * n


def run_dir_predict(
    yolo: YOLO,
    lon: np.ndarray,
    lat: np.ndarray,
    hour: str,
    dir_origin_path: str,
    dir_save_path: str,
    dir_json: str,
    start_date: datetime.date = DEFAULT_START_DATE,
    end_date: datetime.date = DEFAULT_END_DATE,
) -> None:
    """Run folder inference, fuse tiled detections and export per-day JSON."""
    os.makedirs(dir_save_path, exist_ok=True)
    os.makedirs(dir_json, exist_ok=True)

    all_classes_nums = np.zeros(len(CLASS_NAMES))

    for current_date in tqdm(list(collect_dates(start_date, end_date)), desc=f"hour={hour}"):
        date_file = current_date.strftime("%Y%m%d")
        day_results = np.empty((0, 7), dtype=np.float32)

        img_paths = sorted(glob.glob(os.path.join(dir_origin_path, f"{date_file}*.jpg")))
        for image_path in img_paths:
            image = Image.open(image_path)
            img_name = os.path.basename(image_path)
            pred_image, results = yolo.detect_image(image, eddy_minradius=100)
            if results is None:
                continue

            day_results_tile = project_daypicture(
                boxes=results,
                tile_name=img_name,
                patch_size=yolo._defaults["input_shape"][0],
                lon_shape=lon.shape,
            )
            day_results = np.concatenate((day_results, day_results_tile), axis=0)
            pred_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

        if day_results.shape[0] == 0:
            continue

        day_results_t = torch.from_numpy(day_results)
        detections = day_results_t[:, :4]
        scores = day_results_t[:, 4] * day_results_t[:, 5]
        keep_idx = nms(detections, scores, yolo._defaults["nms_iou"])
        day_results = day_results[keep_idx]

        if len(day_results.shape) == 1:
            day_results = np.reshape(day_results, (1, 7))

        classes_nums = np.zeros(len(CLASS_NAMES))
        for class_idx in range(len(CLASS_NAMES)):
            classes_nums[class_idx] = np.sum(day_results[:, -1] == class_idx)

        tempdict = {
            "time": date_file,
            "AE or CE label": "0 or 1",
            "Anticyclonic": float((day_results[:, -1] == 0).sum()),
            "Cyclonic": float((day_results[:, -1] == 1).sum()),
            "results": {
                "predict": day_results.tolist(),
                "eddy_type": [int(day_results[i][-1]) for i in range(day_results.shape[0])],
                "eddy_center_lon_lat": [
                    project_geo(
                        ((day_results[i, 2] + day_results[i, 0]) // 2, (day_results[i, 3] + day_results[i, 1]) // 2),
                        lon,
                        lat,
                    )
                    for i in range(day_results.shape[0])
                ],
                "box_min_lon_lat": [project_geo((day_results[i, 0], day_results[i, 1]), lon, lat) for i in range(day_results.shape[0])],
                "box_max_lon_lat": [project_geo((day_results[i, 2], day_results[i, 3]), lon, lat) for i in range(day_results.shape[0])],
                "eddy_inradius": [
                    float(min(day_results[i, 2] - day_results[i, 0], day_results[i, 3] - day_results[i, 1]) * 500 / 2)
                    for i in range(day_results.shape[0])
                ],
                "eddy_internal_ellipse_area": [
                    float(0.25 * np.pi * (day_results[i, 2] - day_results[i, 0]) * 500 * 500 * (day_results[i, 3] - day_results[i, 1]))
                    for i in range(day_results.shape[0])
                ],
                "confidence": [float(day_results[i, 4] * day_results[i, 5]) for i in range(day_results.shape[0])],
            },
        }

        out_json = os.path.join(dir_json, f"{date_file}{hour}.json")
        if not os.path.isfile(out_json):
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(tempdict, f)

        all_classes_nums += classes_nums
        print(classes_nums, f"{date_file}{hour}", all_classes_nums, sep=" ")


def main() -> None:
    yolo = YOLO()
    lon, lat = load_geo_grid(GOCI_GRID_FILE)

    # Modes: predict / video / fps / dir_predict / heatmap / export_onnx
    mode = "dir_predict"
    crop = False
    count = True

    video_path = 0
    video_save_path = ""
    video_fps = 25.0

    test_interval = 100
    fps_image_path = "img/street.jpg"

    heatmap_save_path = "model_data/heatmap_vision.png"
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        while True:
            img = input("Input image filename:")
            try:
                image = Image.open(img)
            except Exception:
                print("Open Error! Try again!")
                continue
            r_image, _ = yolo.detect_image(image, crop=crop, count=count)
            r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, _ = capture.read()
        if not ref:
            raise ValueError("Failed to read camera/video. Please check the device or video path.")

        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame, _ = yolo.detect_image(frame)
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

            fps = (fps + (1.0 / (time.time() - t1))) / 2
            frame = cv2.putText(frame, f"fps= {fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xFF
            if video_save_path:
                out.write(frame)
            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path:
            print(f"Save processed video to: {video_save_path}")
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(f"{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1")

    elif mode == "dir_predict":
        # NOTE: Replace these paths with your local GOCI clipped tile directories.
        for hour in ["00", "01", "02", "03", "04", "05", "06", "07"]:
            dir_origin_path = f"E:/photo/clip/{hour}/"
            dir_save_path = f"E:/photo/predict/{hour}/"
            dir_json = f"E:/photo/predict/{hour}/dataset/"
            run_dir_predict(yolo, lon, lat, hour, dir_origin_path, dir_save_path, dir_json)

    elif mode == "heatmap":
        while True:
            img = input("Input image filename:")
            try:
                image = Image.open(img)
            except Exception:
                print("Open Error! Try again!")
                continue
            yolo.detect_heatmap(image, heatmap_save_path)

    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError(
            "Please specify mode in: 'predict', 'video', 'fps', 'dir_predict', 'heatmap', 'export_onnx'."
        )


if __name__ == "__main__":
    main()

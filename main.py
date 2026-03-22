import numpy as np
import cv2
import sort
import utilities
import homography_tracker
from ultralytics import YOLO


def main(opts):
    video1 = cv2.VideoCapture(opts.video1)
    assert video1.isOpened(), f"Could not open video1 source {opts.video1}"

    video2 = cv2.VideoCapture(opts.video2)
    assert video2.isOpened(), f"Could not open video2 source {opts.video2}"

    cam4_H_cam1 = np.load(opts.homography)
    cam1_H_cam4 = np.linalg.inv(cam4_H_cam1)

    homographies = list()
    homographies.append(np.eye(3))
    homographies.append(cam1_H_cam4)

    # Load YOLOv8n and force CPU inference
    detector = YOLO("yolov8n.pt")
    detector.to("cpu")

    trackers = [
        sort.Sort(
            max_age=opts.max_age, min_hits=opts.min_hits, iou_threshold=opts.iou_thres
        )
        for _ in range(2)
    ]

    global_tracker = homography_tracker.MultiCameraTracker(homographies, iou_thres=0.20)

    num_frames1 = video1.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames2 = video2.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames = min(num_frames2, num_frames1)
    num_frames = int(num_frames)

    # NOTE: Second video 'cam4.mp4' is 17 frames behind the first video 'cam1.mp4'
    video2.set(cv2.CAP_PROP_POS_FRAMES, 17)

    for idx in range(num_frames):
        # Get frames (kept as BGR — YOLOv8 accepts BGR natively)
        frame1 = video1.read()[1]
        frame2 = video2.read()[1]

        if frame1 is None or frame2 is None:
            break

        frames = [frame1, frame2]

        # Run YOLOv8n inference on CPU
        # classes=[0] → person only; verbose=False silences per-frame logs
        results = detector(
            frames,
            device="cpu",
            classes=[0],
            conf=opts.conf,
            verbose=False,
        )

        dets, tracks = [], []

        for i in range(len(results)):
            # YOLOv8 boxes.data → (x1, y1, x2, y2, conf, cls)
            det = results[i].boxes.data.cpu().numpy()

            if det.shape[0] == 0:
                # No detections: provide empty array with correct shape
                det = np.empty((0, 6))

            det[:, :4] = np.intp(det[:, :4])
            dets.append(det)

            # SORT tracker expects (x1, y1, x2, y2) bboxes + class labels
            tracker = trackers[i].update(det[:, :4], det[:, -1])
            tracks.append(tracker)

        global_ids = global_tracker.update(tracks)

        for i in range(2):
            frames[i] = utilities.draw_tracks(
                frames[i],
                tracks[i],
                global_ids[i],
                i,
                classes=detector.names,
            )

        vis = np.hstack(frames)
        cv2.namedWindow("Vis", cv2.WINDOW_NORMAL)
        cv2.imshow("Vis", vis)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    video1.release()
    video2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", type=str, default="./epfl/cam1.mp4")
    parser.add_argument("--video2", type=str, default="./epfl/cam4.mp4")
    parser.add_argument("--homography", type=str, default="./cam4_H_cam1.npy")
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.3,
        help="IOU threshold to consider a match between two bounding boxes.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Max age of a track, i.e., how many frames will we keep a track alive.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Minimum number of matches to consider a track.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold for the YOLOv8n detector.",
    )

    opts = parser.parse_args()
    main(opts)

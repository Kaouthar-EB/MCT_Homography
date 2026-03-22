import cv2
import numpy as np

centroids = {}


def apply_homography(uv, H):
    uv_ = np.zeros_like(uv)
    for idx, (u, v) in enumerate(uv):
        uvs = (H @ np.array([u, v, 1]).reshape(3, 1)).flatten()
        u_, v_, s_ = uvs
        uv_[idx] = [u_ / s_, v_ / s_]
    return uv_


def apply_homography_xyxy(xyxy, H):
    xyxy_ = np.zeros_like(xyxy)
    for idx, (x1, y1, x2, y2) in enumerate(xyxy):
        # .flatten() converts the (3,1) result to (3,) so unpacking yields plain scalars
        x1_, y1_, s1 = (H @ np.array([x1, y1, 1]).reshape(3, 1)).flatten()
        x2_, y2_, s2 = (H @ np.array([x2, y2, 1]).reshape(3, 1)).flatten()
        xyxy_[idx] = [x1_ / s1, y1_ / s1, x2_ / s2, y2_ / s2]
    return xyxy_


def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image given a list of (x1, y1, x2, y2) coordinates.
    """
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = np.intp(bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_matches(img1, kpts1, img2, kpts2, matches):
    vis = np.hstack([img1, img2])
    MAX_DIST_VAL = max([match.distance for match in matches])
    WIDTH = img2.shape[1]

    for idx, (src, dst, match) in enumerate(zip(kpts1, kpts2, matches)):
        src_x, src_y = src
        dst_x, dst_y = dst
        dst_x += WIDTH
        COLOR = (0, int(255 * (match.distance / MAX_DIST_VAL)), 0)
        vis = cv2.line(vis, (src_x, src_y), (dst_x, dst_y), COLOR, 1)

    return vis


def color_from_id(id):
    np.random.seed(id)
    return np.random.randint(0, 255, size=3).tolist()


def draw_tracks(image, tracks, ids_dict, src, classes=None):
    """
    Draw bounding boxes and tracking IDs on an image.
    """
    vis = np.array(image)

    bboxes = tracks[:, :4]
    ids    = tracks[:, 4]
    labels = tracks[:, 5]

    centroids[src] = centroids.get(src, {})

    for i, box in enumerate(bboxes):
        id    = ids_dict[ids[i]]
        color = color_from_id(id)

        x1, y1, x2, y2 = np.intp(box)

        centroids[src][id] = centroids[src].get(id, [])
        centroids[src][id].append(((x1 + x2) // 2, (y1 + y2) // 2))
        vis = draw_history(vis, box, centroids[src][id], color)

        label_idx = int(labels[i])
        text = f"{classes[label_idx] if classes else label_idx} {id}"
        vis = cv2.putText(
            vis, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2,
        )

    return vis


def draw_label(image, x, y, label, track_id, color):
    vis  = np.array(image)
    text = f"{label} {track_id}"
    vis  = cv2.putText(
        vis, text, (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2,
    )
    return vis


def draw_history(image, box, centroids, color):
    """
    Draw a bounding box and the trail of historical centroids.
    """
    vis = np.array(image)

    x1, y1, x2, y2 = np.intp(box)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    centroids = np.intp(centroids)
    for i, centroid in enumerate(centroids):
        if i == 0:
            cv2.circle(vis, centroid, 2, color, thickness=-1)
        else:
            cv2.line(vis, centroids[i - 1], centroid, color, thickness=2)

    return vis
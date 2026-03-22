import numpy as np
from sort import associate_detections_to_trackers


def modify_bbox_source(bboxes, homography):
    """
    Project bounding boxes from one camera's coordinate space into another
    using a homography matrix.

    Args:
        bboxes (np.ndarray): Bounding boxes (N, 5+) with columns (x0,y0,x1,y1,...).
        homography (np.ndarray): 3x3 homography matrix.

    Returns:
        np.ndarray: Projected bounding boxes.
    """
    bboxes_ = list()
    for bbox in bboxes:
        x0, y0, x1, y1, *keep = bbox

        p0 = (homography @ np.array([x0, y0, 1]).reshape(3, 1)).flatten()
        p1 = (homography @ np.array([x1, y1, 1]).reshape(3, 1)).flatten()

        x0 = int(p0[0] / p0[2])
        y0 = int(p0[1] / p0[2])
        x1 = int(p1[0] / p1[2])
        y1 = int(p1[1] / p1[2])

        bboxes_.append([x0, y0, x1, y1] + keep)

    return np.asarray(bboxes_)


class MultiCameraTracker:
    def __init__(self, homographies: list, iou_thres=0.2):
        """Multi-camera tracker constructor."""
        self.num_sources = len(homographies)
        self.homographies = homographies
        self.iou_thres = iou_thres
        self.next_id = 1
        self.ids = [{} for _ in range(self.num_sources)]
        self.age = [{} for _ in range(self.num_sources)]

    def update(self, tracks: list):
        # Project every camera's tracks into the shared reference frame
        proj_tracks = []
        for i, trks in enumerate(tracks):
            proj_tracks.append(modify_bbox_source(trks, self.homographies[i]))

        # For each pair of cameras, match tracks across views
        for i in range(self.num_sources):
            for j in range(i + 1, self.num_sources):
                matched = {}
                matches, unmatches_i, unmatches_j = associate_detections_to_trackers(
                    proj_tracks[i], proj_tracks[j], iou_threshold=self.iou_thres
                )

                for idx_i, idx_j in matches:
                    id_i = proj_tracks[i][idx_i][4]
                    id_j = proj_tracks[j][idx_j][4]

                    match_i = self.ids[i].get(id_i)
                    match_j = self.ids[j].get(id_j)

                    if (
                        match_i is not None
                        and self.age[i].get(id_i, 0) >= self.age[j].get(id_j, 0)
                        and not matched.get(match_i, False)
                    ):
                        self.ids[j][id_j] = match_i
                        matched[match_i] = True

                    elif match_j is not None and not matched.get(match_j, False):
                        self.ids[i][id_i] = match_j
                        matched[match_j] = True

                    else:
                        self.ids[i][id_i] = self.next_id
                        self.ids[j][id_j] = self.next_id
                        matched[self.next_id] = True
                        self.next_id += 1

                    self.age[i][id_i] = self.age[i].get(id_i, 0) + 1
                    self.age[j][id_j] = self.age[j].get(id_j, 0) + 1

                for idx_i in unmatches_i:
                    id_i = proj_tracks[i][idx_i][4]
                    match_i = self.ids[i].get(id_i)
                    if match_i is None or matched.get(match_i, False):
                        self.ids[i][id_i] = self.next_id
                        matched[self.next_id] = True
                        self.next_id += 1
                    self.age[i][id_i] = self.age[i].get(id_i, 0) + 1

                for idx_j in unmatches_j:
                    id_j = proj_tracks[j][idx_j][4]
                    match_j = self.ids[j].get(id_j)
                    if match_j is None or matched.get(match_j, False):
                        self.ids[j][id_j] = self.next_id
                        matched[self.next_id] = True
                        self.next_id += 1
                    self.age[j][id_j] = self.age[j].get(id_j, 0) + 1

        return self.ids
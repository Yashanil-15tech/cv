import cv2
import numpy as np
from collections import deque, defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')


class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale, dtype=np.uint8)
        width = first_frame_grayscale.shape[1]
        mask_features[:, 0:min(20, width)] = 1
        if width > 900:
            mask_features[:, min(900, width):min(1050, width)] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def get_camera_movement(self, frames):
        camera_movement = [[0, 0]] * len(frames)
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            if old_features is None or len(old_features) < 10:
                old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
            if old_features is None:
                old_gray = frame_gray
                continue

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            if new_features is None or status is None:
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                old_gray = frame_gray
                continue

            good_old = old_features[status == 1]
            good_new = new_features[status == 1]

            if len(good_old) < 5:
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                old_gray = frame_gray
                continue

            movements = good_new - good_old
            camera_movement_x = np.median(movements[:, 0])
            camera_movement_y = np.median(movements[:, 1])
            movement_magnitude = np.sqrt(camera_movement_x**2 + camera_movement_y**2)

            if movement_magnitude > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        return camera_movement


class ImprovedHomographyEstimator():
    """FIXED: Corrected homography with proper pitch dimensions"""
    def __init__(self):
        self.pitch_length = 105.0
        self.pitch_width = 68.0
       
    def get_manual_keypoints(self, frame):
        """Manual keypoint selection for accurate calibration"""
        print("\n" + "="*60)
        print("MANUAL CALIBRATION - Select 4 corners of a known feature")
        print("="*60)
        print("Options:")
        print("  1. Penalty box: 40.3m wide × 16.5m deep")
        print("  2. Center circle: 18.3m diameter")
        print("  3. Goal area: 18.3m wide × 5.5m deep")
        print("\nOrder: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
        print("Press 'r' to reset, 'q' when done")
        print("="*60 + "\n")
       
        points = []
       
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 4:
                    points.append([x, y])
                    cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(frame_copy, f"{len(points)}", (x+10, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Calibration', frame_copy)
       
        frame_copy = frame.copy()
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.imshow('Calibration', frame_copy)
        cv2.setMouseCallback('Calibration', click_event)
       
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and len(points) == 4:
                break
            elif key == ord('r'):
                points = []
                frame_copy = frame.copy()
                cv2.imshow('Calibration', frame_copy)
       
        cv2.destroyAllWindows()
       
        if len(points) == 4:
            return np.float32(points)
        return None
   
    def compute_homography_manual(self, frame):
        """Compute homography from manual calibration"""
        src_points = self.get_manual_keypoints(frame)
       
        if src_points is None:
            print("⚠ Manual calibration failed")
            return None
       
        # Ask user which feature they selected
        print("\nWhich feature did you select?")
        print("1. Penalty box (40.3m × 16.5m)")
        print("2. Center circle (18.3m diameter)")
        print("3. Goal area (18.3m × 5.5m)")
        choice = input("Enter 1, 2, or 3: ")
       
        if choice == '1':
            # Penalty box
            dst_points = np.float32([
                [0, 0],
                [16.5, 0],
                [16.5, 40.3],
                [0, 40.3]
            ])
        elif choice == '2':
            # Center circle
            dst_points = np.float32([
                [0, 0],
                [18.3, 0],
                [18.3, 18.3],
                [0, 18.3]
            ])
        elif choice == '3':
            # Goal area
            dst_points = np.float32([
                [0, 0],
                [5.5, 0],
                [5.5, 18.3],
                [0, 18.3]
            ])
        else:
            print("Invalid choice, using penalty box")
            dst_points = np.float32([
                [0, 0],
                [16.5, 0],
                [16.5, 40.3],
                [0, 40.3]
            ])
       
        H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
       
        if H is not None:
            print("✓ Homography computed from manual calibration")
       
        return H
   
    def compute_homography_auto(self, frame):
        """
        FIXED: Improved automatic homography with corrected aspect ratio
        """
        h, w = frame.shape[:2]
       
        # More conservative ROI
        roi_top = 0.30
        roi_bottom = 0.15
        roi_left = 0.08
        roi_right = 0.08
       
        src_points = np.float32([
            [w * roi_left, h * roi_top],
            [w * (1 - roi_right), h * roi_top],
            [w * (1 - roi_right), h * (1 - roi_bottom)],
            [w * roi_left, h * (1 - roi_bottom)]
        ])
       
        # FIX: Corrected pitch dimensions accounting for camera angle
        # Typical broadcast camera sees ~60-70% of length, ~50-60% of width
        visible_length = self.pitch_length * 0.50  # ~52m
        visible_width = self.pitch_width * 0.55    # ~37m
       
        dst_points = np.float32([
            [0, 0],
            [visible_length, 0],
            [visible_length, visible_width],
            [0, visible_width]
        ])
       
        H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
       
        print("⚠ Using automatic homography (less accurate)")
        print(f"  Assumed visible area: {visible_length:.1f}m × {visible_width:.1f}m")
        print("  → For best results, use method='manual'")
       
        return H


class KalmanTracker:
    def __init__(self, initial_position):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.kf.statePost = np.array([
            initial_position[0],
            initial_position[1],
            0, 0
        ], np.float32)

    def predict(self, camera_movement=[0, 0]):
        prediction = self.kf.predict()
        compensated_x = prediction[0].item() - camera_movement[0]
        compensated_y = prediction[1].item() - camera_movement[1]
        return (int(compensated_x), int(compensated_y))

    def update(self, measurement, camera_movement=[0, 0]):
        compensated_measurement = [
            measurement[0] + camera_movement[0],
            measurement[1] + camera_movement[1]
        ]
        self.kf.correct(np.array([
            [np.float32(compensated_measurement[0])],
            [np.float32(compensated_measurement[1])]
        ]))
        return (int(self.kf.statePost[0]), int(self.kf.statePost[1]))


class OptimizedPlayerTracker:
    """
    FIXED: Tracker with Wu & Swartz (2022) multi-frame smoothing
    and proper acceleration filtering
    """
    def __init__(self, max_age=40, min_hits=6, reid_memory=200, max_total_ids=25,
                 fps=20.0, homography_matrix=None):
        self.next_id = 0
        self.tracks = {}
        self.max_age = max_age
        self.min_hits = min_hits
        self.reid_memory = reid_memory
        self.max_total_ids = max_total_ids
        self.deleted_tracks = {}
        self.frame_count = 0
        self.initialization_window = 120

        # FIXED: Wu & Swartz parameters
        self.fps = fps
        self.homography_matrix = homography_matrix
        self.track_speeds = {}
        self.track_distances = {}
        self.track_position_history = {}
       
        # CRITICAL FIXES:
        self.position_history_length = 9  # Wu & Swartz: Use Δ=4 frames = 8 positions
        self.speed_smoothing_sigma = 1.5  # Gaussian smoothing
        self.speed_history_window = 20    # Increased smoothing window
       
        # Validation thresholds
        self.MAX_REALISTIC_SPEED = 35.0  # km/h
        self.MAX_REALISTIC_ACCELERATION = 10.0  # m/s²
        self.MAX_FRAME_TO_FRAME_SPEED_CHANGE = 15.0  # km/h (filter spikes)

        # Team assignment
        self.jersey_colors_hsv = defaultdict(list)
        self.team_assignment = {}

    def calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def try_reidentify(self, detection, current_frame):
        if len(self.deleted_tracks) == 0:
            return None

        x, y, w, h = detection
        det_center = (x + w//2, y + h//2)

        best_match_id = None
        best_score = float('inf')

        for track_id, track_info in self.deleted_tracks.items():
            last_pos = track_info['last_position']
            last_size = track_info['last_size']
            deleted_frame = track_info['deleted_frame']

            if current_frame - deleted_frame > self.reid_memory:
                continue

            distance = np.linalg.norm(np.array(det_center) - np.array(last_pos))
            size_diff = abs(w - last_size[0]) + abs(h - last_size[1])
            score = distance + size_diff * 0.3

            if score < 120 and score < best_score:
                best_score = score
                best_match_id = track_id

        return best_match_id

    def pixel_to_world(self, pixel_pos):
        """Transform pixel coordinates to world coordinates"""
        if self.homography_matrix is None:
            return None
       
        try:
            point = np.array([[pixel_pos]], dtype=np.float32)
            world_point = cv2.perspectiveTransform(point, self.homography_matrix)
            return world_point[0][0]
        except:
            return None

    def calculate_speed_wu_swartz(self, track_id, current_pos_pixel):
        """
        FIXED: Wu & Swartz (2022) method with multi-frame smoothing
        Speed(t) = distance(t-Δ, t+Δ) / (2*Δ*dt)
        """
        current_world = self.pixel_to_world(current_pos_pixel)
       
        if current_world is None:
            return 0, 0
       
        # Initialize position history
        if track_id not in self.track_position_history:
            self.track_position_history[track_id] = deque(maxlen=self.position_history_length)
       
        self.track_position_history[track_id].append(current_world)
       
        positions = list(self.track_position_history[track_id])
       
        # Need at least 5 positions for Δ=2 frames
        if len(positions) < 5:
            return 0, 0
       
        # Wu & Swartz method: Use positions t-Δ and t+Δ
        # For Δ=2 frames (0.08s at 25fps):
        delta_frames = 2
       
        if len(positions) >= 2 * delta_frames + 1:
            # Central difference: speed at position[center]
            center_idx = len(positions) // 2
            past_idx = max(0, center_idx - delta_frames)
            future_idx = min(len(positions) - 1, center_idx + delta_frames)
           
            distance_meters = np.linalg.norm(positions[future_idx] - positions[past_idx])
            time_interval = (future_idx - past_idx) / self.fps
        else:
            # Fallback: Use first and last positions
            distance_meters = np.linalg.norm(positions[-1] - positions[0])
            time_interval = (len(positions) - 1) / self.fps
       
        if time_interval <= 0:
            return 0, 0
       
        speed_mps = distance_meters / time_interval
        speed_kmh = speed_mps * 3.6
       
        # CRITICAL: Validate against previous speed (filter spikes)
        if track_id in self.track_speeds and len(self.track_speeds[track_id]) > 0:
            prev_speed = self.track_speeds[track_id][-1]
            speed_change = abs(speed_kmh - prev_speed)
           
            if speed_change > self.MAX_FRAME_TO_FRAME_SPEED_CHANGE:
                # Spike detected - use previous speed instead
                speed_kmh = prev_speed
                distance_meters = prev_speed / 3.6 / self.fps
       
        # Hard cap at realistic max
        if speed_kmh > self.MAX_REALISTIC_SPEED:
            speed_kmh = 0
            distance_meters = 0
       
        # Distance for this frame step
        frame_distance = distance_meters / (len(positions) - 1) if len(positions) > 1 else distance_meters
       
        return speed_kmh, frame_distance

    def smooth_speed_history(self, track_id):
        """Apply Gaussian smoothing to speed history"""
        if track_id not in self.track_speeds or len(self.track_speeds[track_id]) < 3:
            return
       
        speeds = np.array(self.track_speeds[track_id])
        smoothed = gaussian_filter1d(speeds, sigma=self.speed_smoothing_sigma)
        self.track_speeds[track_id] = smoothed.tolist()

    def get_speed_stats(self, track_id):
        """Get smoothed speed statistics"""
        if track_id not in self.track_speeds or len(self.track_speeds[track_id]) == 0:
            return {
                'current_speed': 0,
                'avg_speed': 0,
                'avg_moving_speed': 0,
                'max_speed': 0,
                'total_distance': 0
            }

        speeds = self.track_speeds[track_id]
        recent_speeds = speeds[-self.speed_history_window:]
        current_speed = np.mean(recent_speeds) if recent_speeds else 0

        moving_speeds = [s for s in speeds if s > 1.8]

        return {
            'current_speed': current_speed,
            'avg_speed': np.mean(speeds),
            'avg_moving_speed': np.mean(moving_speeds) if moving_speeds else 0,
            'max_speed': np.max(speeds) if speeds else 0,
            'total_distance': self.track_distances.get(track_id, 0)
        }

    def update(self, detections, camera_movement=[0, 0]):
        self.frame_count += 1

        # Predict
        predicted_positions = {}
        for track_id, track in self.tracks.items():
            if track['kalman'] is not None:
                predicted_pos = track['kalman'].predict(camera_movement)
                predicted_positions[track_id] = predicted_pos

        # Cost matrix
        track_ids = list(self.tracks.keys())
        cost_matrix = np.full((len(track_ids), len(detections)), 1e6)

        if len(track_ids) > 0 and len(detections) > 0:
            for i, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                predicted_pos = predicted_positions.get(track_id, track['position'])
                last_w, last_h = track.get('size', (0, 0))
                predicted_box = (
                    predicted_pos[0] - last_w//2,
                    predicted_pos[1] - last_h//2,
                    last_w, last_h
                )

                for j, det in enumerate(detections):
                    det_center = (det[0] + det[2]//2, det[1] + det[3]//2)
                    distance = np.linalg.norm(
                        np.array(predicted_pos) - np.array(det_center)
                    )
                    iou = self.calculate_iou(predicted_box, det)
                    cost = distance * (1.5 - iou)

                    if distance < 100 or iou > 0.15:
                        cost_matrix[i, j] = cost

        # Hungarian matching
        matched = []
        unmatched_tracks = list(range(len(track_ids)))
        unmatched_detections = list(range(len(detections)))

        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 120:
                    matched.append((track_ids[r], c))
                    if r in unmatched_tracks:
                        unmatched_tracks.remove(r)
                    if c in unmatched_detections:
                        unmatched_detections.remove(c)

        # Update matched tracks
        for track_id, det_idx in matched:
            x, y, w, h = detections[det_idx]
            center = (x + w//2, y + h//2)

            if self.tracks[track_id]['kalman'] is not None:
                updated_pos = self.tracks[track_id]['kalman'].update(
                    center, camera_movement
                )
            else:
                updated_pos = center

            # Calculate speed with Wu & Swartz method
            speed_kmh, distance_m = self.calculate_speed_wu_swartz(track_id, updated_pos)

            if track_id not in self.track_speeds:
                self.track_speeds[track_id] = []
            self.track_speeds[track_id].append(speed_kmh)

            if track_id not in self.track_distances:
                self.track_distances[track_id] = 0
            self.track_distances[track_id] += distance_m

            # Apply Gaussian smoothing every 10 frames
            if len(self.track_speeds[track_id]) % 10 == 0:
                self.smooth_speed_history(track_id)

            self.tracks[track_id]['position'] = updated_pos
            self.tracks[track_id]['bbox'] = (x, y, w, h)
            self.tracks[track_id]['size'] = (w, h)
            self.tracks[track_id]['age'] = 0
            self.tracks[track_id]['hits'] += 1
            self.tracks[track_id]['time_since_update'] = 0
            self.tracks[track_id]['history'].append(updated_pos)

        # Update unmatched
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            self.tracks[track_id]['age'] += 1
            self.tracks[track_id]['time_since_update'] += 1

            if track_id in predicted_positions and self.tracks[track_id]['time_since_update'] < 20:
                self.tracks[track_id]['position'] = predicted_positions[track_id]
                x, y = predicted_positions[track_id]
                w, h = self.tracks[track_id]['size']
                self.tracks[track_id]['bbox'] = (x - w//2, y - h//2, w, h)

        # Re-ID
        remaining_unmatched = []
        reidentified_count = 0

        for det_idx in unmatched_detections:
            reid_track_id = self.try_reidentify(detections[det_idx], self.frame_count)

            if reid_track_id is not None:
                x, y, w, h = detections[det_idx]
                center = (x + w//2, y + h//2)

                old_track_info = self.deleted_tracks[reid_track_id]

                self.tracks[reid_track_id] = {
                    'position': center,
                    'bbox': (x, y, w, h),
                    'size': (w, h),
                    'age': 0,
                    'hits': old_track_info['hits'] + 1,
                    'time_since_update': 0,
                    'history': deque(maxlen=50),
                    'kalman': KalmanTracker(center),
                    'color': old_track_info['color']
                }

                del self.deleted_tracks[reid_track_id]
                reidentified_count += 1
            else:
                remaining_unmatched.append(det_idx)

        if reidentified_count > 0:
            print(f"Frame {self.frame_count}: Re-identified {reidentified_count} players!")

        # Create new tracks
        if (self.frame_count <= self.initialization_window and
            self.next_id < self.max_total_ids and
            len(remaining_unmatched) > 0):

            det_idx = remaining_unmatched[0]
            x, y, w, h = detections[det_idx]
            center = (x + w//2, y + h//2)

            self.tracks[self.next_id] = {
                'position': center,
                'bbox': (x, y, w, h),
                'size': (w, h),
                'age': 0,
                'hits': 1,
                'time_since_update': 0,
                'history': deque(maxlen=50),
                'kalman': KalmanTracker(center),
                'color': tuple(np.random.randint(50, 255, 3).tolist())
            }

            self.track_speeds[self.next_id] = []
            self.track_distances[self.next_id] = 0

            self.next_id += 1

        elif self.frame_count == self.initialization_window + 1:
            print(f"\n*** Initialization complete at frame {self.frame_count} ***")
            print(f"*** Total IDs created: {self.next_id} ***\n")

        # Remove dead tracks
        dead_tracks = [
            tid for tid in self.tracks
            if self.tracks[tid]['age'] > self.max_age
        ]

        for tid in dead_tracks:
            self.deleted_tracks[tid] = {
                'last_position': self.tracks[tid]['position'],
                'last_size': self.tracks[tid]['size'],
                'deleted_frame': self.frame_count,
                'color': self.tracks[tid]['color'],
                'hits': self.tracks[tid]['hits']
            }
            del self.tracks[tid]

        # Clean up
        old_deleted = [
            tid for tid, info in self.deleted_tracks.items()
            if self.frame_count - info['deleted_frame'] > self.reid_memory
        ]
        for tid in old_deleted:
            del self.deleted_tracks[tid]

        return {
            k: v for k, v in self.tracks.items()
            if v['hits'] >= self.min_hits and v['time_since_update'] < 25
        }


def extract_jersey_color_hsv(frame, x, y, w, h):
    try:
        roi_y_start = max(0, y)
        roi_y_end = min(frame.shape[0], y + int(h * 0.5))
        roi_x_start = max(0, x + int(w * 0.2))
        roi_x_end = min(frame.shape[1], x + int(w * 0.8))

        roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = (hsv[:,:,2] > 50) & (hsv[:,:,2] < 220) & (hsv[:,:,1] > 40)

        if np.sum(mask) < 20:
            return None

        valid_hsv = hsv[mask]

        if len(valid_hsv) < 20:
            return None

        valid_hsv_float = valid_hsv.reshape(-1, 3).astype(float)
        n_clusters = min(2, len(valid_hsv_float))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        kmeans.fit(valid_hsv_float)

        cluster_centers = kmeans.cluster_centers_
        best_cluster = np.argmax(cluster_centers[:, 1])
        dominant_hsv = cluster_centers[best_cluster]

        return dominant_hsv

    except:
        return None


def is_referee_jersey(hsv_color):
    if hsv_color is None:
        return True
    h, s, v = hsv_color
    if v < 80 and s < 100:
        return True
    return False


def assign_teams_kmeans_hsv(tracker, frame, confirmed_tracks):
    for track_id, track in confirmed_tracks.items():
        x, y, w, h = track['bbox']
        hsv_color = extract_jersey_color_hsv(frame, x, y, w, h)

        if hsv_color is not None and not is_referee_jersey(hsv_color):
            tracker.jersey_colors_hsv[track_id].append(hsv_color)

    valid_tracks = []
    valid_colors = []

    for track_id, colors in tracker.jersey_colors_hsv.items():
        if len(colors) >= 5:
            avg_color = np.mean(colors, axis=0)
            if not is_referee_jersey(avg_color):
                valid_tracks.append(track_id)
                valid_colors.append(avg_color)

    if len(valid_colors) >= 2:
        colors_array = np.array(valid_colors)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
        team_labels = kmeans.fit_predict(colors_array)

        for i, track_id in enumerate(valid_tracks):
            tracker.team_assignment[track_id] = int(team_labels[i])

        return True

    return False


def non_max_suppression(detections, iou_threshold=0.3):
    if len(detections) == 0:
        return []

    boxes = np.array(detections)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]

    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]


def get_playing_field_boundaries(frame):
    h, w = frame.shape[:2]
    return int(h * 0.25), int(h * 0.85), int(w * 0.05), int(w * 0.95)


# ============ MAIN PROCESSING ============
print("="*60)
print("FIXED FOOTBALL TRACKER - Wu & Swartz (2022) Method")
print("="*60)

video_path = '/content/08fd33_4.mp4'
frames = []
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

if not frames:
    print("Error: Could not load video or video is empty.")
else:
    print(f"\nLoaded {len(frames)} frames at {fps} FPS\n")

    # STEP 1: Compute Homography
    print("="*60)
    print("HOMOGRAPHY CALIBRATION")
    print("="*60)
    homography_estimator = ImprovedHomographyEstimator()
   
    # Use 'manual' for best accuracy, 'auto' for quick testing
    calibration_method = 'auto'  # Change to 'manual' for production
   
    if calibration_method == 'manual':
        H = homography_estimator.compute_homography_manual(frames[0])
    else:
        H = homography_estimator.compute_homography_auto(frames[0])
   
    if H is not None:
        print("✓ Homography matrix computed\n")
    else:
        print("✗ Homography computation failed\n")

    # STEP 2: Camera Movement
    print("Estimating camera movement...")
    camera_estimator = CameraMovementEstimator(frames[0])
    camera_movements = camera_estimator.get_camera_movement(frames)
    print("✓ Camera movement estimated\n")

    h, w = frames[0].shape[:2]
    roi_top, roi_bottom, roi_left, roi_right = get_playing_field_boundaries(frames[0])

    # STEP 3: Initialize Tracker
    tracker = OptimizedPlayerTracker(
        max_age=40,
        min_hits=6,
        reid_memory=200,
        max_total_ids=25,
        fps=fps,
        homography_matrix=H
    )

    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=True)
    detected_frames = []
    teams_assigned = False

    print("Processing frames...")
    print("="*60)

    for idx, frame in enumerate(frames):
        out_frame = frame.copy()
        camera_movement = camera_movements[idx] if idx > 0 else [0, 0]

        cv2.rectangle(out_frame, (roi_left, roi_top), (roi_right, roi_bottom), (255, 255, 0), 2)

        fgmask = fgbg.apply(frame, learningRate=0.003)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        fgmask_roi = np.zeros_like(fgmask)
        fgmask_roi[roi_top:roi_bottom, roi_left:roi_right] = fgmask[roi_top:roi_bottom, roi_left:roi_right]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, (0, 0, 160), (180, 70, 255))
        green_mask = cv2.inRange(hsv, (35, 55, 40), (85, 255, 255))
        pitch_mask = cv2.inRange(hsv, (25, 25, 25), (90, 255, 170))

        players_mask = cv2.bitwise_and(
            cv2.bitwise_or(white_mask, green_mask),
            cv2.bitwise_not(pitch_mask)
        )

        players_mask = cv2.morphologyEx(players_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        players_mask = cv2.morphologyEx(players_mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))

        mask_roi = np.zeros_like(players_mask)
        mask_roi[roi_top:roi_bottom, roi_left:roi_right] = players_mask[roi_top:roi_bottom, roi_left:roi_right]

        ref_mask = cv2.inRange(hsv, (0,0,0), (180,70,80))
        mask_roi = cv2.bitwise_and(mask_roi, cv2.bitwise_not(ref_mask))

        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            area = w_box * h_box

            if area < 300 or area > 5500:
                continue
            aspect = h_box / float(w_box)
            if aspect < 1.3 or aspect > 4.5:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(cv2.contourArea(cnt)) / hull_area
                if solidity < 0.55:
                    continue

            detections.append((x, y, w_box, h_box))

        detections = non_max_suppression(detections, iou_threshold=0.3)
        confirmed_tracks = tracker.update(detections, camera_movement)

        # Team assignment
        if 50 <= idx <= 150:
            assign_teams_kmeans_hsv(tracker, frame, confirmed_tracks)

        if idx == 151 and not teams_assigned:
            teams_assigned = True
            team0_count = sum(1 for tid in tracker.team_assignment if tracker.team_assignment[tid] == 0)
            team1_count = sum(1 for tid in tracker.team_assignment if tracker.team_assignment[tid] == 1)
            print(f"✓ Teams assigned: Team 1={team0_count}, Team 2={team1_count}\n")

        active_count = 0
        for track_id, track in confirmed_tracks.items():
            x, y, w, h = track['bbox']
            center = track['position']

            if not (roi_left < center[0] < roi_right and roi_top < center[1] < roi_bottom):
                continue

            team = tracker.team_assignment.get(track_id, -1)
            if team == 0:
                color = (0, 255, 0)
            elif team == 1:
                color = (255, 100, 0)
            else:
                color = track['color']

            active_count += 1

            if track['time_since_update'] > 0:
                cv2.rectangle(out_frame, (x, y), (x+w, y+h), color, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(out_frame, (x, y), (x+w, y+h), color, 2)

            cv2.circle(out_frame, center, 4, color, -1)

            if len(track['history']) > 1:
                pts = np.array(track['history'], dtype=np.int32)
                cv2.polylines(out_frame, [pts], False, color, 2)

            speed_stats = tracker.get_speed_stats(track_id)

            if team == 0:
                label = f"P{track_id} T1"
            elif team == 1:
                label = f"P{track_id} T2"
            else:
                label = f"P{track_id}"

            if speed_stats['current_speed'] > 1.0:
                label += f" {speed_stats['current_speed']:.1f} km/h"

            cv2.putText(out_frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2, cv2.LINE_AA)

        status = "INIT" if tracker.frame_count <= tracker.initialization_window else "TRACK"
        cv2.putText(out_frame, f'{status} | Active: {active_count}/{tracker.next_id}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(out_frame, f'F:{idx} | Det:{len(detections)}',
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        detected_frames.append(out_frame)

        if idx % 100 == 0:
            print(f"Frame {idx:03d} | {status} | Active:{active_count}/{tracker.next_id}")

    # Save video
    output_path = '/content/tracked_fixed.mp4'
    height, width = detected_frames[0].shape[:2]
    out_vid = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in detected_frames:
        out_vid.write(frame)
    out_vid.release()

    print(f"\n{'='*60}")
    print(f"✓ VIDEO SAVED: {output_path}")
    print(f"{'='*60}")

    # Statistics
    print(f"\n{'='*60}")
    print(f"FIXED PLAYER STATISTICS")
    print(f"Expected: Avg 10-13 km/h | Max 30-35 km/h | Sprint <8%")
    print(f"{'='*60}")

    team1 = []
    team2 = []
    others = []

    for tid in sorted(tracker.track_distances.keys()):
        stats = tracker.get_speed_stats(tid)
        entry = (tid, stats['total_distance'], stats['avg_moving_speed'], stats['max_speed'])

        team = tracker.team_assignment.get(tid, -1)
        if team == 0:
            team1.append(entry)
        elif team == 1:
            team2.append(entry)
        else:
            others.append(entry)

    print("\nTEAM 1")
    print("-" * 60)
    for tid, dist, avg_speed, max_speed in team1:
        flag = "✓" if max_speed <= 35 and 8 <= avg_speed <= 16 else "⚠"
        print(f"{flag} Player {tid}: Dist={dist:.1f}m | Avg={avg_speed:.1f} km/h | Max={max_speed:.1f} km/h")

    print("\nTEAM 2")
    print("-" * 60)
    for tid, dist, avg_speed, max_speed in team2:
        flag = "✓" if max_speed <= 35 and 8 <= avg_speed <= 16 else "⚠"
        print(f"{flag} Player {tid}: Dist={dist:.1f}m | Avg={avg_speed:.1f} km/h | Max={max_speed:.1f} km/h")

    print("\nUNASSIGNED")
    print("-" * 60)
    for tid, dist, avg_speed, max_speed in others:
        print(f"ID {tid}: Dist={dist:.1f}m | Avg={avg_speed:.1f} km/h | Max={max_speed:.1f} km/h")

    # Save CSV
    import csv
    csv_path = '/content/player_stats_fixed.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Player_ID', 'Team', 'Total_Distance_m', 'Avg_Speed_kmh',
                         'Avg_Moving_Speed_kmh', 'Max_Speed_kmh'])

        for tid in sorted(tracker.track_distances.keys()):
            stats = tracker.get_speed_stats(tid)
            team = tracker.team_assignment.get(tid, -1)
            team_label = "Team1" if team == 0 else "Team2" if team == 1 else "Unassigned"

            writer.writerow([
                tid,
                team_label,
                f"{stats['total_distance']:.2f}",
                f"{stats['avg_speed']:.2f}",
                f"{stats['avg_moving_speed']:.2f}",
                f"{stats['max_speed']:.2f}"
            ])

    print(f"\n✓ Stats saved to: {csv_path}")
    print(f"{'='*60}")

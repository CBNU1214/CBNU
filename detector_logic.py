import numpy as np
import cv2
import time
import os
import shutil
# [제거됨] import pygame
# [제거됨] pygame.mixer.init()
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Dict, Set, Tuple, Optional, List
from enum import Enum
import mediapipe as mp

# 랜드마크 인덱스
RIGHT_EYE = [33, 133, 159, 158, 145, 153]
LEFT_EYE = [362, 263, 386, 385, 374, 380]
EYES = RIGHT_EYE + LEFT_EYE
CHIN_POINT = 152
NOSE_TIP = 1

# 설정값 (AI 로직 - 원본 유지)
MIN_EAR = 0.25
TITLE_NAME = ''
YOLO_WORLD_CLASSES = [
    "phone", "mobile phone", "smartphone",
    "hat", "cap", "baseball cap", "beanie", "helmet", "fedora", "cowboy hat"
]
STACK_THRESHOLD = 80
STACK_DECREASE_RATE = 2
BASE_CHIN_NOSE_DISTANCE = 80
BASE_CHIN_THRESHOLD = 120
ATTENTION_THRESHOLD = 20

class DetectionType(Enum):
    PHONE = ('P', None, 'Phone', 'Please put your phone aside')
    HAT = ('H', None, 'Hat', 'Please remove your hat')
    EYE = ('E', None, 'Eye', 'Do not doze off')
    ATTENTION = ('A', None, 'Attention', 'Please pay attention')
    CHIN = ('C', None, 'Chin', 'Do not hold your chin up')
    
@dataclass
class PointInfo:
    x: int
    y: int
    success_stack: int
    is_fixed: bool
    stacks: Optional[Dict[str, int]] = None

class DrowsinessDetectorYOLOWorld:
    def __init__(self):
        self.setup_directories()
        self.load_models()
        self.center_points: Dict[int, PointInfo] = {}
        self.captured_images: Dict[int, str] = {}
        self.detection_windows: Dict[str, Dict[int, bool]] = {dt.value[0]: {} for dt in DetectionType}
        self.detection_images: Dict[Tuple[int, str], str] = {}
        
        # [추가됨] Streamlit 대시보드로 보낼 현재 활성화된 경고
        self.active_alerts: Dict[Tuple[int, str], Tuple[str, np.ndarray]] = {}

    def setup_directories(self):
        for directory in ['temp_images', 'escape_images']:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def load_models(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.1
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.3, min_tracking_confidence=0.3
        )
        
        try:
            self.yolo_model = YOLO('yolov8n-pose.pt')
        except Exception as e:
            print(f"치명적 오류: yolov8n-pose.pt 로드 실패: {e}")
            raise  # app.py가 이 에러를 잡도록 다시 발생시킴

        try:
            self.yolo_world_model = YOLO('yolov8s-world.pt')
            self.yolo_world_model.set_classes(YOLO_WORLD_CLASSES)
            print(f"YOLO 로드 성공")
        except Exception as e:
            print(f"YOLO 로드 실패: {e}")
            self.yolo_world_model = None
        
    # (이하 님의 모든 AI 로직 함수들은 원본 그대로 유지됩니다)
    # classify_yolo_world_detection, create_new_stacks, get_next_available_id,
    # is_point_in_box, get_ear, calculate_distance, calculate_adaptive_threshold,
    # ... (중략) ...
    # analyze_person_in_roi, capture_box_image 등 모든 AI 로직은 동일합니다.
    
    def classify_yolo_world_detection(self, class_name):
        class_name_lower = class_name.lower()
        phone_keywords = ['phone', 'cell', 'mobile', 'smartphone']
        if any(keyword in class_name_lower for keyword in phone_keywords):
            return 'PHONE'
        hat_keywords = ['hat', 'cap', 'beanie', 'helmet', 'fedora', 'cowboy']
        if any(keyword in class_name_lower for keyword in hat_keywords):
            return 'HAT'
        return None

    @staticmethod
    def create_new_stacks() -> Dict[str, int]:
        return {dt.value[0]: 0 for dt in DetectionType} | {'S': 0}

    @staticmethod
    def get_next_available_id(used_ids: Set[int]) -> int:
        next_id = 1
        while next_id in used_ids:
            next_id += 1
        return next_id

    @staticmethod
    def is_point_in_box(point: Tuple[int, int], box: Tuple[int, int, int, int]) -> bool:
        x, y = point
        bx, by, bw, bh = box
        return bx <= x <= bx + bw and by <= y <= by + bh

    @staticmethod
    def get_ear(points: np.ndarray) -> float:
        if len(points) != 6: return 0.3
        A = np.linalg.norm(points[2] - points[4])
        B = np.linalg.norm(points[3] - points[5])
        C = np.linalg.norm(points[0] - points[1])
        return (A + B) / (2.0 * C) if C > 0 else 0.3

    @staticmethod
    def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    @staticmethod
    def calculate_adaptive_threshold(face_box_size: float, base_threshold: float, min_threshold: float, max_threshold: float, base_face_size: float = 150.0) -> float:
        scale_factor = face_box_size / base_face_size
        return max(min_threshold, min(max_threshold, base_threshold * scale_factor))
    
    @staticmethod
    def calculate_adaptive_chin_threshold(face_box_size: float) -> float:
        return DrowsinessDetectorYOLOWorld.calculate_adaptive_threshold(
            face_box_size, BASE_CHIN_THRESHOLD, 30.0, 200.0
        )

    @staticmethod
    def calculate_adaptive_attention_threshold(face_box_size: float) -> float:
        return DrowsinessDetectorYOLOWorld.calculate_adaptive_threshold(
            face_box_size, ATTENTION_THRESHOLD, 10.0, 60.0
        )

    def get_landmarks_points(self, landmarks, indices: List[int], roi_width: int, roi_height: int, roi_offset: Tuple[int, int]) -> np.ndarray:
        points = []
        offset_x, offset_y = roi_offset
        for idx in indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * roi_width) + offset_x
            y = int(landmark.y * roi_height) + offset_y
            points.append([x, y])
        return np.array(points)

    def detect_attention(self, face_id: int, landmarks: object, roi_width: int, roi_height: int, roi_offset: Tuple[int, int], face_detection_results, image_width: int, image_height: int) -> Tuple[bool, Optional[Tuple[int, int]], Optional[float]]:
        offset_x, offset_y = roi_offset
        nose_x = int(landmarks.landmark[NOSE_TIP].x * roi_width) + offset_x
        nose_y = int(landmarks.landmark[NOSE_TIP].y * roi_height) + offset_y
        
        face_box_size = 150.0
        current_face_center_x, current_face_center_y = None, None
        
        if face_detection_results.detections:
            for detection in face_detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * image_width)
                y = int(bbox.ymin * image_height)
                w = int(bbox.width * image_width)
                h = int(bbox.height * image_height)
                
                if x <= nose_x <= x + w and y <= nose_y <= y + h:
                    current_face_center_x = x + w // 2
                    current_face_center_y = y + h // 2
                    face_box_size = max(w, h)
                    break
        
        if current_face_center_x is None or current_face_center_y is None:
            return True, None, face_box_size
        
        distance = self.calculate_distance((nose_x, nose_y), (current_face_center_x, current_face_center_y))
        adaptive_threshold = self.calculate_adaptive_attention_threshold(face_box_size)
        is_distracted = distance > adaptive_threshold
        
        return is_distracted, (current_face_center_x, current_face_center_y), face_box_size

    def find_person_box_for_point(self, point_info: PointInfo, person_boxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        for bx1, by1, bx2, by2 in person_boxes:
            if bx1 <= point_info.x <= bx2 and by1 <= point_info.y <= by2:
                return (bx1, by1, bx2, by2)
        return None

    def analyze_person_in_roi(self, image: np.ndarray, face_id: int, person_box: Tuple[int, int, int, int], keypoints, face_detection_results) -> Dict[str, any]:
        bx1, by1, bx2, by2 = person_box
        roi = image[by1:by2, bx1:bx2]
        roi_height, roi_width = roi.shape[:2]
        roi_offset = (bx1, by1)
        
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb_roi)
        
        detections = {'EYE': False, 'ATTENTION': False, 'CHIN': False}
        
        if not face_results.multi_face_landmarks:
            detections['ATTENTION'] = True
            return detections
        
        landmarks = face_results.multi_face_landmarks[0]
        
        try:
            # 1. 눈 감지
            right_eye_points = self.get_landmarks_points(landmarks, RIGHT_EYE, roi_width, roi_height, roi_offset)
            left_eye_points = self.get_landmarks_points(landmarks, LEFT_EYE, roi_width, roi_height, roi_offset)
            mean_ear = (self.get_ear(right_eye_points) + self.get_ear(left_eye_points)) / 2
            detections['EYE'] = mean_ear < MIN_EAR
            
            # 2. 주의집중 감지
            height, width = image.shape[:2]
            detections['ATTENTION'], current_face_center, face_box_size = self.detect_attention(face_id, landmarks, roi_width, roi_height, roi_offset, face_detection_results, width, height)
            
            # 3. 턱 좌표
            chin_x = int(landmarks.landmark[CHIN_POINT].x * roi_width) + bx1
            chin_y = int(landmarks.landmark[CHIN_POINT].y * roi_height) + by1
            chin_point = (chin_x, chin_y)
            
            # 4. 턱 괴기 감지
            if keypoints is not None:
                adaptive_chin_threshold = self.calculate_adaptive_chin_threshold(face_box_size)
                left_wrist, right_wrist = keypoints[9], keypoints[10]
                min_dist = min(self.calculate_distance((left_wrist[0], left_wrist[1]), chin_point),
                               self.calculate_distance((right_wrist[0], right_wrist[1]), chin_point))
                detections['CHIN'] = min_dist < adaptive_chin_threshold
                
                cv2.circle(image, chin_point, 5, (0, 0, 255), -1)
                cv2.circle(image, (int(left_wrist[0]), int(left_wrist[1])), 5, (255, 0, 0), -1)
                cv2.circle(image, (int(right_wrist[0]), int(right_wrist[1])), 5, (255, 0, 0), -1)

            if current_face_center:
                nose_x = int(landmarks.landmark[NOSE_TIP].x * roi_width) + bx1
                nose_y = int(landmarks.landmark[NOSE_TIP].y * roi_height) + by1
                current_distance = self.calculate_distance((nose_x, nose_y), current_face_center)
                adaptive_attention_threshold = self.calculate_adaptive_attention_threshold(face_box_size)
                status = "DISTRACTED" if detections['ATTENTION'] else "FOCUSED"
                color = (0, 0, 255) if detections['ATTENTION'] else (0, 255, 0)
                
                cv2.putText(image, f"{status} D:{current_distance:.1f} (T:{adaptive_attention_threshold:.1f})", 
                           (bx1, by1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.line(image, (nose_x, nose_y), current_face_center, color, 2)
            
        except Exception as e:
            print(f"Error in ROI analysis: {e}")
        
        return detections

    def capture_box_image(self, image: np.ndarray, box: Tuple[int, int, int, int], face_id: int, detection_type: DetectionType = None) -> np.ndarray:
        x1, y1, x2, y2 = box
        padding = 20
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
        
        captured_image = image[y1:y2, x1:x2]
        
        if detection_type:
            text = detection_type.value[3]
            # [수정됨] 한글 폰트 대신 영문으로 되돌리거나, 별도 폰트 파일 처리가 필요합니다.
            # 여기서는 OpenCV 기본 폰트를 사용합니다.
            font_scale = captured_image.shape[1] * 0.002
            cv2.putText(captured_image, text, (10, captured_image.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        
        return captured_image

    def move_to_escape_folder(self, face_id: int):
        if face_id in self.captured_images:
            src_path = self.captured_images[face_id]
            if os.path.exists(src_path):
                dst_path = os.path.join('escape_images', os.path.basename(src_path))
                shutil.move(src_path, dst_path)
                del self.captured_images[face_id]

    def cleanup_point(self, face_id: int):
        if face_id in self.center_points: del self.center_points[face_id]
        if face_id in self.captured_images: del self.captured_images[face_id]
        for windows in self.detection_windows.values():
            if face_id in windows: del windows[face_id]

    # [수정됨] 웹(app.py)과 연동되도록 수정
    def close_detection_window(self, face_id, detection_type, detection_key):
        stack_key = detection_type.value[0]
        
        if self.detection_windows[stack_key].get(face_id, False):
            # [제거] cv2.destroyWindow(win_name)
            
            # [추가] Streamlit 대시보드에서 경고를 제거하기 위해 딕셔너리에서 삭제
            if detection_key in self.active_alerts:
                del self.active_alerts[detection_key]
                
            if detection_key in self.detection_images:
                image_path = self.detection_images[detection_key]
                if os.path.exists(image_path):
                    os.remove(image_path) # 파일은 여전히 삭제 (원본 로직 유지)
                del self.detection_images[detection_key]
            self.detection_windows[stack_key][face_id] = False

    # [수정됨] 웹(app.py)과 연동되도록 수정
    def handle_detection_window(self, image, face_id, detection_type, person_box, stack_value):
        stack_key = detection_type.value[0]
        threshold = STACK_THRESHOLD
        detection_key = (face_id, stack_key)
        
        if stack_value >= threshold and not self.detection_windows[stack_key].get(face_id, False):
            captured_img = self.capture_box_image(image, person_box, face_id, detection_type)
            image_path = f"temp_images/capture_{face_id}_{detection_type.value[2]}.jpg"
            cv2.imwrite(image_path, captured_img) # 파일 저장은 원본 로직 유지
            self.detection_images[detection_key] = image_path
            
            # [제거] cv2.imshow(win_name, captured_img)
            self.detection_windows[stack_key][face_id] = True
            
            # [추가] Streamlit 대시보드로 보낼 메시지와 이미지 저장
            message = f"🚨 학생 ID {face_id}: {detection_type.value[3]}"
            self.active_alerts[detection_key] = (message, captured_img)
            
            # [제거] pygame.mixer.music.load("beep.wav")
            # [제거] pygame.mixer.music.play()
            
        elif stack_value < threshold and self.detection_windows[stack_key].get(face_id, False):
            self.close_detection_window(face_id, detection_type, detection_key)

    def process_fixed_points(self, image: np.ndarray, person_boxes: List[Tuple[int, int, int, int]], keypoints_dict: Dict, yolo_world_results, face_detection_results):
        for face_id, point_info in list(self.center_points.items()):
            if not point_info.is_fixed or point_info.stacks is None: continue

            person_box = self.find_person_box_for_point(point_info, person_boxes)
            
            if person_box is None:
                point_info.stacks['S'] += 1
                if point_info.stacks['S'] >= 100:
                    self.move_to_escape_folder(face_id)
                    self.cleanup_point(face_id)
                    continue
                
                for detection_type in DetectionType:
                    stack_key = detection_type.value[0]
                    point_info.stacks[stack_key] = max(point_info.stacks[stack_key] - STACK_DECREASE_RATE, 0)
                    self.close_detection_window(face_id, detection_type, (face_id, stack_key))
                continue
            
            point_info.stacks['S'] = 0
            
            keypoints = keypoints_dict.get(person_box)
            detections = self.analyze_person_in_roi(image, face_id, person_box, keypoints, face_detection_results)
            
            if yolo_world_results and yolo_world_results[0].boxes is not None:
                bx1, by1, bx2, by2 = person_box
                for obj_box, cls_id, conf in zip(yolo_world_results[0].boxes.xyxy, 
                                               yolo_world_results[0].boxes.cls, 
                                               yolo_world_results[0].boxes.conf):
                    cls_id = int(cls_id)
                    class_name = self.yolo_world_model.names[cls_id]
                    detection_type_name = self.classify_yolo_world_detection(class_name)
                    
                    if detection_type_name:
                        ox1, oy1, ox2, oy2 = map(int, obj_box)
                        if (bx1 <= ox1 <= bx2 and bx1 <= ox2 <= bx2 and
                            by1 <= oy1 <= by2 and by1 <= oy2 <= by2):
                            detections[detection_type_name] = True
                            
                            color = (0, 255, 255) if detection_type_name == 'PHONE' else (255, 0, 255)
                            label = f"{class_name}: {conf:.2f}"
                            cv2.rectangle(image, (ox1, oy1), (ox2, oy2), color, 2)
                            cv2.putText(image, label, (ox1, oy1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            for detection_type in DetectionType:
                stack_key = detection_type.value[0]
                detected = detections.get(detection_type.name, False)
                
                if detected:
                    if stack_key == 'A':
                        point_info.stacks[stack_key] = min(point_info.stacks[stack_key] + 0.5, 100)
                    else:
                        point_info.stacks[stack_key] = min(point_info.stacks[stack_key] + 1, 100)
                else:
                    point_info.stacks[stack_key] = max(point_info.stacks[stack_key] - STACK_DECREASE_RATE, 0)
                
                self.handle_detection_window(image, face_id, detection_type, person_box, point_info.stacks[stack_key])

    def update_points_from_faces(self, image: np.ndarray, face_detection_results, yolo_results):
        height, width = image.shape[:2]
        
        detected_faces = []
        if face_detection_results.detections:
            for detection in face_detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                detected_faces.append((x, y, w, h))
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        for face_id, point_info in list(self.center_points.items()):
            if point_info.is_fixed: continue

            point_in_any_face = False
            for face in detected_faces:
                if self.is_point_in_box((point_info.x, point_info.y), face):
                    point_in_any_face = True
                    new_stack = min(point_info.success_stack + 1, 100)
                    
                    if new_stack >= 100 and not point_info.is_fixed:
                        self.center_points[face_id] = PointInfo(
                            point_info.x, point_info.y, new_stack, True, 
                            self.create_new_stacks()
                        )
                        
                        if yolo_results is not None:
                            for box in yolo_results[0].boxes.xyxy:
                                bx1, by1, bx2, by2 = map(int, box)
                                if self.is_point_in_box((point_info.x, point_info.y), 
                                                      (bx1, by1, bx2-bx1, by2-by1)):
                                    captured_img = self.capture_box_image(image, (bx1, by1, bx2, by2), face_id)
                                    image_path = f"temp_images/capture_{face_id}_fixed.jpg"
                                    cv2.imwrite(image_path, captured_img)
                                    self.captured_images[face_id] = image_path
                                    break
                    else:
                        self.center_points[face_id] = PointInfo(
                            point_info.x, point_info.y, new_stack, False, None
                        )
                    break

            if not point_in_any_face:
                new_stack = max(point_info.success_stack - 1, 0)
                if new_stack > 0:
                    self.center_points[face_id] = PointInfo(
                        point_info.x, point_info.y, new_stack, False, None
                    )
                else:
                    self.cleanup_point(face_id)

        for face in detected_faces:
            x, y, w, h = face
            points_in_box = [(face_id, point_info) for face_id, point_info in self.center_points.items()
                            if self.is_point_in_box((point_info.x, point_info.y), face)]
            
            if not points_in_box:
                face_center_x = x + w//2
                face_center_y = y + h//2
                
                should_create_point = True
                if hasattr(self, 'last_person_boxes') and hasattr(self, 'last_person_box_fixed_points'):
                    for person_box in self.last_person_boxes:
                        bx1, by1, bx2, by2 = person_box
                        if self.is_point_in_box((face_center_x, face_center_y), 
                                               (bx1, by1, bx2 - bx1, by2 - by1)):
                            if self.last_person_box_fixed_points[person_box]:
                                should_create_point = False
                                break
                
                if should_create_point:
                    new_id = self.get_next_available_id(set(self.center_points.keys()))
                    self.center_points[new_id] = PointInfo(face_center_x, face_center_y, 0, False, None)
            elif len(points_in_box) > 1:
                points_in_box.sort(key=lambda x: x[1].success_stack, reverse=True)
                for face_id, _ in points_in_box[1:]:
                    self.cleanup_point(face_id)

    def draw_points(self, image: np.ndarray):
        for point_info in self.center_points.values():
            color = (0, 0, 255) if point_info.is_fixed else (0, 255, 0)
            cv2.circle(image, (point_info.x, point_info.y), 3, color, -1)
            line_length = 10
            cv2.line(image, (point_info.x - line_length, point_info.y), (point_info.x + line_length, point_info.y), color, 1)
            cv2.line(image, (point_info.x, point_info.y - line_length), (point_info.x, point_info.y + line_length), color, 1)
            
            if point_info.is_fixed and point_info.stacks:
                stack_items = list(point_info.stacks.items())
                stack_text1 = "FIXED"
                for i in range(4):
                    if i < len(stack_items): stack_text1 += f" {stack_items[i][0]}:{stack_items[i][1]}"
                stack_text2 = ""
                for i in range(4, len(stack_items)):
                    stack_text2 += f" {stack_items[i][0]}:{stack_items[i][1]}"
                
                cv2.putText(image, stack_text1, (point_info.x + 15, point_info.y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if stack_text2:
                    cv2.putText(image, stack_text2, (point_info.x + 15, point_info.y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(image, f"STACK:{point_info.success_stack}", (point_info.x + 15, point_info.y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        start_time = time.time()
        
        yolo_results = self.yolo_model(image, verbose=False)
        
        yolo_world_results = None
        if self.yolo_world_model:
            yolo_world_results = self.yolo_world_model(image, conf=0.15, verbose=False)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_detection_results = self.face_detection.process(rgb_image)
        
        person_boxes = []
        keypoints_dict = {}
        person_box_fixed_points = {}
        
        if yolo_results and yolo_results[0].boxes:
            for box, keypoints in zip(yolo_results[0].boxes.xyxy, yolo_results[0].keypoints.data):
                bx1, by1, bx2, by2 = map(int, box)
                person_box = (bx1, by1, bx2, by2)
                person_boxes.append(person_box)
                keypoints_dict[person_box] = keypoints
                
                cv2.rectangle(image, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                
                has_fixed_point = any(point_info.is_fixed and self.is_point_in_box((point_info.x, point_info.y), (bx1, by1, bx2-bx1, by2-by1))
                                    for point_info in self.center_points.values())
                person_box_fixed_points[person_box] = has_fixed_point
                
                if has_fixed_point:
                    exclude_idx = {0, 1, 2, 3, 4}
                    for idx, keypoint in enumerate(keypoints):
                        if idx not in exclude_idx:
                            x, y, conf = keypoint
                            if conf > 0.5:
                                cv2.circle(image, (int(x), int(y)), 4, (255, 0, 0), -1)
        
        self.last_person_boxes = person_boxes
        self.last_person_box_fixed_points = person_box_fixed_points
        
        self.update_points_from_faces(image, face_detection_results, yolo_results)
        
        self.process_fixed_points(image, person_boxes, keypoints_dict, yolo_world_results, face_detection_results)
        
        self.draw_points(image)
        
        elapsed_time = time.time() - start_time
        if elapsed_time < 0.05:
            time.sleep(0.05 - elapsed_time)
        
        return image

# [제거됨] 님의 원본 main() 함수는 app.py에서 사용하지 않으므로 제거 (있어도 상관은 없음)
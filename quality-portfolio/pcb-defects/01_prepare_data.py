"""
PCB 결함 이미지(PASCAL VOC 주석) 파싱 & 특징 추출
- 데이터: data/PCB_DATASET/{Annotations,images}/<결함유형>/ (693장 이미지, 2,953개 결함 인스턴스)
- 결함당 bbox를 크롭해 64x64로 리사이즈한 뒤, 기하/밝기 특징(9) + HOG 특징(36, 2x2 구역 x 9방향)
  = 45차원 특징으로 변환한다 (WM-811K 프로젝트와 동일하게 원본 이미지를 그대로 넣지 않고
  도메인 기반 특징을 설계하는 방식을 따른다).
"""
import glob
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.filters import sobel
from skimage.transform import resize

DATA_DIR = "data/PCB_DATASET"
CROP_SIZE = 64
EDGE_THRESHOLD = 0.1

LABELS = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
label_to_num = {lbl: i for i, lbl in enumerate(LABELS)}

# ------------------------------------------------------------------
# 1. XML 주석 파싱 -> (이미지경로, bbox, 라벨) 목록 만들기
# ------------------------------------------------------------------
records = []
for xml_path in sorted(glob.glob(f"{DATA_DIR}/Annotations/*/*.xml")):
    root = ET.parse(xml_path).getroot()
    folder = root.find("folder").text
    filename = root.find("filename").text
    img_path = os.path.join(DATA_DIR, "images", folder, filename)
    size_node = root.find("size")
    board_w = int(size_node.find("width").text)
    board_h = int(size_node.find("height").text)

    for obj in root.findall("object"):
        label = obj.find("name").text
        box = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (int(box.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
        records.append({
            "img_path": img_path,
            "label": label,
            "board_w": board_w,
            "board_h": board_h,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        })

meta = pd.DataFrame.from_records(records)
print(f"[1] 이미지 {meta.img_path.nunique()}장, 결함 인스턴스 {len(meta)}개 파싱 완료")
print(meta.label.value_counts().reindex(LABELS))

# ------------------------------------------------------------------
# 2. bbox 크롭 -> 특징 추출 (기하/밝기 9 + HOG 36 = 45차원)
# ------------------------------------------------------------------
FEATURE_NAMES = [
    "bbox_width", "bbox_height", "aspect_ratio", "bbox_area", "area_ratio_to_board",
    "mean_intensity", "std_intensity", "edge_density", "mean_edge_magnitude",
] + [f"hog_q{q}_bin{b}" for q in range(1, 5) for b in range(9)]

_img_cache = {}


def load_image(path):
    if path not in _img_cache:
        _img_cache[path] = np.array(Image.open(path).convert("RGB"))
    return _img_cache[path]


def extract_features(row):
    img = load_image(row.img_path)
    crop = img[row.ymin:row.ymax, row.xmin:row.xmax]
    if crop.size == 0:
        crop = img[max(0, row.ymin - 1):row.ymax + 1, max(0, row.xmin - 1):row.xmax + 1]

    w, h = row.xmax - row.xmin, row.ymax - row.ymin
    aspect_ratio = w / h if h > 0 else 0
    area = w * h
    area_ratio = area / (row.board_w * row.board_h)

    gray = rgb2gray(crop)
    gray_resized = resize(gray, (CROP_SIZE, CROP_SIZE), anti_aliasing=True)

    mean_intensity = gray_resized.mean() * 255
    std_intensity = gray_resized.std() * 255

    edges = sobel(gray_resized)
    edge_density = float((edges > EDGE_THRESHOLD).mean())
    mean_edge_magnitude = float(edges.mean())

    hog_feat = hog(
        gray_resized, orientations=9, pixels_per_cell=(32, 32),
        cells_per_block=(1, 1), feature_vector=True,
    )  # 2x2 공간 구역 x 9방향 = 36차원

    geom = [w, h, aspect_ratio, area, area_ratio, mean_intensity, std_intensity, edge_density, mean_edge_magnitude]
    return np.concatenate([geom, hog_feat])


print(f"[2] {len(meta)}개 결함 인스턴스 특징 추출 중 (bbox 크롭 -> {CROP_SIZE}x{CROP_SIZE} 리사이즈 -> 45차원)...")
X = np.array([extract_features(row) for row in meta.itertuples()])
y = meta.label.map(label_to_num).values.astype(int)
print(f"    최종 특징 행렬: {X.shape}")

np.save("features_X.npy", X)
np.save("labels_y.npy", y)
meta.to_pickle("data/instances_meta.pkl")
with open("data/feature_names.json", "w", encoding="utf-8") as f:
    json.dump(FEATURE_NAMES, f, ensure_ascii=False, indent=2)

print("[3] 저장 완료: features_X.npy, labels_y.npy, data/instances_meta.pkl, data/feature_names.json")

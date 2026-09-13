# 데이터 출처와 실행 준비

대용량 원본 데이터·특징 배열 및 제3자 넷리스트는 재배포하지 않습니다. 아래 원배포처의 이용 조건을 확인한 뒤 해당 파일을 준비합니다. 코드·분석 결과·설명 자료를 포트폴리오에 포함했습니다.

| 프로젝트 | 배포처 | 필요한 파일 |
|---|---|---|
| PCB 결함 유형 분류 | [Kaggle — PCB Defects, akhatova](https://www.kaggle.com/datasets/akhatova/pcb-defects) | `PCB_DATASET/Annotations/<유형>/*.xml`, `PCB_DATASET/images/<유형>/*` |
| ISCAS85 회로 결함진단 | [jpsety — verilog_benchmark_circuits](https://github.com/jpsety/verilog_benchmark_circuits) | `c17.v`, `c432.v`, `c499.v`, `c880.v`, `c1355.v`, `c1908.v` |
| 제조공정 품질분석 | [Kaggle — Predicting Manufacturing Defects Dataset, rabieelkharoua](https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset) | `manufacturing_defect_dataset.csv` |

PCB 파일은 `pcb-defects/data/PCB_DATASET/` 아래에 배치합니다. 배포 압축파일의 추가 상위 폴더가 있다면 위 구조에 맞게 옮깁니다. 전체 배포 이미지 수와 본 분석에서 사용한 주석 결함 이미지 693장은 구분해야 합니다.

ISCAS85는 `circuit-faults/01_generate_data.py`로 다운로드할 수 있습니다. 기존 분석에 사용한 파일과의 비교를 위해 [넷리스트 SHA-256 목록](circuit-faults/results/netlist_sha256.json)을 제공합니다. 상위 저장소 내용이 바뀌면 분석 결과도 달라질 수 있습니다.

제조공정 CSV는 `manufacturing-defects/data/`에 배치합니다. 데이터의 산업·측정 조건은 실제 방산 공정을 의미하지 않습니다.

## 코드와 결과의 변경 이력

2026-09-13에 기존 프로젝트의 분석 코드를 정리했습니다. 계산 로직과 원본 학습 조건은 유지했습니다. 한국어 폰트 선택에 대체 폰트를 추가했고, 회로 결함 집합을 설명하는 주석의 과도한 보장 문구를 수정했습니다.

`results/source_summary.json`은 기존 분석 기록에서 자기소개서용 문구를 제외하고 정리한 결과입니다. 회로 원본의 비표준 JSON 값 `NaN`은 `null`로 정규화했습니다. `reproduced_metrics.json`은 이번 환경에서 재실행하거나 대조한 기록입니다. 새로 작성한 README는 코드에서 확인 가능한 수행 범위와 한계를 기준으로 정리했습니다.

"""
데이터 출처: ISCAS'85 벤치마크 회로 (Verilog 게이트레벨 넷리스트)
1985년 International Symposium on Circuits and Systems에서 VLSI 테스트 연구용으로
공개된 이래 40년 가까이 표준 벤치마크로 쓰이는 조합회로 모음이다. 이 저장소는
Cadence Genus로 합성된 구조적(structural) Verilog 형태로 이를 재배포한다:
https://github.com/jpsety/verilog_benchmark_circuits

선정 회로 6종(게이트 수 6 → 479, 소형~중형 순): c17, c432, c499, c880, c1355, c1908
"""
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/jpsety/verilog_benchmark_circuits/master"
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CIRCUITS = ["c17", "c432", "c499", "c880", "c1355", "c1908"]

for name in CIRCUITS:
    dest = OUT_DIR / f"{name}.v"
    if dest.exists():
        print(f"[skip] {name}.v 이미 존재")
        continue
    url = f"{BASE_URL}/{name}.v"
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, dest)

print(f"\n{len(CIRCUITS)}개 넷리스트 다운로드 완료 → {OUT_DIR}")

"""
6개 ISCAS85 회로에 대해 (1) 랜덤 테스트 벡터를 생성하고 (2) 체크포인트
Stuck-at 결함집합을 시뮬레이션해, 벡터 수에 따른 결함 커버리지 곡선과
끝까지 검출되지 않는 결함(redundant fault 후보)을 구한다.

방법: 각 결함에 대해 "가장 먼저 그 결함을 검출하는 테스트 벡터의 인덱스"를
구해두면, "벡터를 k개까지만 썼을 때의 커버리지"를 재시뮬레이션 없이 바로
계산할 수 있다(첫 검출 인덱스가 k 이하인 결함의 비율). c17은 회로가 작아
5개 입력 전체(2^5=32, exhaustive)로도 시뮬레이션해 랜덤 패턴과 비교한다.
"""
import json

import numpy as np
import pandas as pd

from netlist import checkpoint_faults, parse_verilog, simulate, topo_order

RNG = np.random.default_rng(42)
CIRCUITS = ["c17", "c432", "c499", "c880", "c1355", "c1908"]
N_VECTORS = 1000

results = []
per_fault_rows = []

for name in CIRCUITS:
    circuit = parse_verilog(f"data/{name}.v")
    order = topo_order(circuit["gates"])
    n_gates = len(circuit["gates"])
    n_in = len(circuit["inputs"])

    vectors = {net: RNG.integers(0, 2, N_VECTORS).astype(bool) for net in circuit["inputs"]}
    golden = simulate(circuit, order, vectors)

    faults = checkpoint_faults(circuit)
    first_detect = np.full(len(faults), -1)  # -1 = 미검출

    for fi, (fnet, fval) in enumerate(faults):
        faulty = simulate(circuit, order, vectors, force_net=fnet, force_value=fval)
        diff = np.zeros(N_VECTORS, dtype=bool)
        for o in circuit["outputs"]:
            diff |= (golden[o] != faulty[o])
        detected_idx = np.flatnonzero(diff)
        fd = int(detected_idx[0]) if len(detected_idx) else -1
        first_detect[fi] = fd
        per_fault_rows.append({
            "circuit": name, "fault_net": fnet, "fault_type": "SA1" if fval else "SA0",
            "detected": fd >= 0, "first_detect_vector": fd,
        })

    coverage_curve = [float(np.mean((first_detect >= 0) & (first_detect < k)))
                       for k in range(1, N_VECTORS + 1)]
    final_coverage = float(np.mean(first_detect >= 0))

    results.append({
        "circuit": name, "n_gates": n_gates, "n_inputs": n_in, "n_outputs": len(circuit["outputs"]),
        "n_faults": len(faults), "final_coverage_1000vec": final_coverage,
        "n_undetected": int(np.sum(first_detect < 0)),
        "coverage_curve": coverage_curve,
    })
    print(f"[{name}] 게이트 {n_gates}개, 결함 {len(faults)}개, "
          f"{N_VECTORS}벡터 후 커버리지 {final_coverage*100:.1f}%, 미검출 {int(np.sum(first_detect < 0))}개")

# c17은 회로가 작아 exhaustive(2^5=32) 비교도 함께 수행
circuit = parse_verilog("data/c17.v")
order = topo_order(circuit["gates"])
n_in = len(circuit["inputs"])
bits = np.array([[(v >> k) & 1 for k in range(n_in)] for v in range(2 ** n_in)], dtype=bool)
exh_vectors = {name: bits[:, i] for i, name in enumerate(circuit["inputs"])}
golden = simulate(circuit, order, exh_vectors)
faults = checkpoint_faults(circuit)
exh_detect = np.full(len(faults), -1)
for fi, (fnet, fval) in enumerate(faults):
    faulty = simulate(circuit, order, exh_vectors, force_net=fnet, force_value=fval)
    diff = np.zeros(2 ** n_in, dtype=bool)
    for o in circuit["outputs"]:
        diff |= (golden[o] != faulty[o])
    idx = np.flatnonzero(diff)
    exh_detect[fi] = int(idx[0]) if len(idx) else -1
c17_exhaustive_coverage = [float(np.mean((exh_detect >= 0) & (exh_detect < k))) for k in range(1, 2 ** n_in + 1)]

pd.DataFrame(per_fault_rows).to_csv("data/fault_results.csv", index=False)
with open("data/circuit_summary.json", "w", encoding="utf-8") as f:
    json.dump({"circuits": results, "c17_exhaustive_coverage": c17_exhaustive_coverage}, f, ensure_ascii=False)

print("\n결함 시뮬레이션 완료 → data/fault_results.csv, data/circuit_summary.json")

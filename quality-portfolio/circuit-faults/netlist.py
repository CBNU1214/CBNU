"""
Verilog 게이트레벨 넷리스트 파서 + 비트-병렬(bit-parallel) 논리 시뮬레이터.

ISCAS85 넷리스트는 구조적 Verilog(게이트 프리미티브 인스턴스화)로 되어 있어
정규식으로 충분히 파싱 가능하다. 시뮬레이션은 "모든 테스트 벡터를 numpy
불리언 배열로 한 번에 계산"하는 방식으로 벡터화해, 결함(fault)마다 벡터를
하나씩 도는 순수 파이썬 반복보다 훨씬 빠르게 만든다.
"""
import re
from collections import defaultdict, deque

import numpy as np

GATE_RE = re.compile(
    r"\b(nand|nor|xnor|and|or|xor|not|buf)\s+\w+\s*\(([^;]+)\)\s*;", re.IGNORECASE
)


def parse_verilog(path: str) -> dict:
    text = open(path).read()
    text = re.sub(r"//.*", "", text)  # 주석 제거
    text = re.sub(r"\s+", " ", text)

    mod = re.search(r"module\s+\w+\s*\(([^)]*)\)\s*;", text)
    ports = [p.strip() for p in mod.group(1).split(",")]

    def extract(kind):
        m = re.search(rf"\b{kind}\s+([^;]+);", text)
        return [n.strip() for n in m.group(1).split(",")] if m else []

    inputs = extract("input")
    outputs = extract("output")

    gates = []
    for m in GATE_RE.finditer(text):
        gtype = m.group(1).lower()
        nets = [n.strip() for n in m.group(2).split(",")]
        out_net, in_nets = nets[0], nets[1:]
        gates.append({"type": gtype, "out": out_net, "ins": in_nets})

    return {"inputs": inputs, "outputs": outputs, "gates": gates, "ports": ports}


def topo_order(gates: list) -> list:
    """게이트를 (넷 의존관계 기준) 위상정렬 — 조합회로이므로 사이클이 없다."""
    driven_by = {g["out"]: i for i, g in enumerate(gates)}
    indeg = [0] * len(gates)
    children = defaultdict(list)
    for i, g in enumerate(gates):
        for n in g["ins"]:
            if n in driven_by:
                children[driven_by[n]].append(i)
                indeg[i] += 1
    q = deque(i for i in range(len(gates)) if indeg[i] == 0)
    order = []
    while q:
        i = q.popleft()
        order.append(i)
        for c in children[i]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    assert len(order) == len(gates), "사이클 감지됨 — 조합회로 가정 위반"
    return order


_OPS = {
    "and": lambda a: np.bitwise_and.reduce(a),
    "or": lambda a: np.bitwise_or.reduce(a),
    "nand": lambda a: ~np.bitwise_and.reduce(a),
    "nor": lambda a: ~np.bitwise_or.reduce(a),
    "xor": lambda a: np.bitwise_xor.reduce(a),
    "xnor": lambda a: ~np.bitwise_xor.reduce(a),
    "not": lambda a: ~a[0],
    "buf": lambda a: a[0],
}


def simulate(circuit: dict, order: list, input_vectors: dict,
             force_net: str = None, force_value: bool = None) -> dict:
    """input_vectors: {net_name: np.bool_ array(n_vectors,)}.
    force_net/force_value: stuck-at 결함 주입 — 해당 넷의 값을 상수로 고정한다.
    반환: 모든 넷 이름 → np.bool_ 배열(n_vectors,)."""
    n = len(next(iter(input_vectors.values())))
    nets = dict(input_vectors)
    for i in circuit["inputs"]:
        if i not in nets:
            nets[i] = np.zeros(n, dtype=bool)

    # 1차 입력 결함은 게이트 평가 전에 고정해야 그 이후 모든 게이트에 반영된다
    if force_net is not None and force_net in circuit["inputs"]:
        nets[force_net] = np.full(n, force_value, dtype=bool)

    for idx in order:
        g = circuit["gates"][idx]
        if g["out"] == force_net:
            nets[g["out"]] = np.full(n, force_value, dtype=bool)
            continue
        ins = [nets[x] for x in g["ins"]]
        nets[g["out"]] = _OPS[g["type"]](np.array(ins))

    return nets


def checkpoint_faults(circuit: dict) -> list:
    """1차 입력과 모든 게이트 출력에 SA0/SA1을 주입할 대상 집합.
    팬아웃 분기별 결함은 별도로 모델링하지 않는다. 반환된 대상 집합에 대한
    검출률만 평가하며, 모든 물리적 결함의 검출을 보장하지 않는다.
    """
    checkpoints = list(circuit["inputs"])
    for g in circuit["gates"]:
        checkpoints.append(g["out"])
    faults = []
    for net in checkpoints:
        faults.append((net, False))  # SA0
        faults.append((net, True))   # SA1
    return faults

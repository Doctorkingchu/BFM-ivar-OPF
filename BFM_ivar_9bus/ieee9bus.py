import math
import numpy as np
import pandas as pd
import pandapower as pp

def _apply_circle_qcapability_curve_to_gens(net, smax_gen_mva_by_bus, n_points=6,
                                            curve_style="straightLineYValues"):
    """
    P^2 + Q^2 <= Smax^2 (원형 capability)을 (P, Qmin, Qmax) 테이블로 만들어
    pandapower의 q_capability_curve_table 기반 OPF 제약으로 반영한다.

    - n_points: [0, Pmax] 구간 샘플 개수
    - curve_style: "straightLineYValues" (권장) 또는 "constantYValue"
    """
    # --- pandapower 버전에 따라 util 함수명이 다를 수 있어 방어적으로 처리 ---
    try:
        from pandapower.control import util as ppctrl_util
    except Exception:
        return False

    creator = None
    if hasattr(ppctrl_util, "create_q_capability_curve_characteristics_object"):
        creator = ppctrl_util.create_q_capability_curve_characteristics_object
    elif hasattr(ppctrl_util, "create_q_capability_characteristics_object"):
        creator = ppctrl_util.create_q_capability_characteristics_object

    if creator is None:
        return False  # 이 버전은 OPF capability curve를 못 쓸 가능성이 큼

    # --- 테이블 준비 ---
    if "q_capability_curve_table" not in net:
        net["q_capability_curve_table"] = pd.DataFrame(
            columns=["id_q_capability_curve", "p_mw", "q_min_mvar", "q_max_mvar"]
        )

    qtab = net.q_capability_curve_table
    next_id = 0 if qtab.empty else int(qtab["id_q_capability_curve"].max()) + 1

    # --- gen 테이블에 필요한 컬럼이 없으면 생성(버전 차이 방어) ---
    for col, default in [
        ("reactive_capability_curve", False),
        ("id_q_capability_curve_characteristic", np.nan),
        ("curve_style", None),
    ]:
        if col not in net.gen.columns:
            net.gen[col] = default

    # --- 각 발전기별로 (P, Qmin, Qmax) 점들을 추가 ---
    for gi in net.gen.index:
        bus = int(net.gen.at[gi, "bus"])
        Smax = smax_gen_mva_by_bus.get(bus, None)
        if Smax is None:
            continue

        Pmax = float(net.gen.at[gi, "max_p_mw"])
        if Smax < Pmax:
            raise ValueError(f"Gen@bus{bus}: Smax({Smax}) < Pmax({Pmax}) -> 불가능한 정격")

        p_points = np.linspace(0.0, Pmax, int(n_points))
        rows = []
        for p in p_points:
            qcap = math.sqrt(max(Smax**2 - p**2, 0.0))
            rows.append({
                "id_q_capability_curve": next_id,
                "p_mw": float(p),
                "q_min_mvar": float(-qcap),
                "q_max_mvar": float(qcap),
            })

        qtab = pd.concat([qtab, pd.DataFrame(rows)], ignore_index=True)

        # OPF에서 이 커브를 쓰도록 gen에 플래그/ID 설정
        net.gen.at[gi, "reactive_capability_curve"] = True
        net.gen.at[gi, "id_q_capability_curve_characteristic"] = int(next_id)
        net.gen.at[gi, "curve_style"] = curve_style

        next_id += 1

    net["q_capability_curve_table"] = qtab

    # --- characteristic 객체 생성(문서상 OPF에서 사용되는 연결고리) ---
    creator(net)

    # (선택) 진단 함수가 있으면 sanity check
    if hasattr(ppctrl_util, "q_capability_curve_table_diagnostic"):
        ppctrl_util.q_capability_curve_table_diagnostic(net)

    return True


def busradial9_opf(slack_vm_pu: float = 1.0, line_max_loading_percent: float = 1e6):
    net = pp.create_empty_network(sn_mva=100.0)
    buses = [pp.create_bus(net, vn_kv=10.0, name=f"Bus {b}") for b in range(9)]
    slack_bus = buses[0]

    # Bus voltage limits
    net.bus["min_vm_pu"] = 0.95
    net.bus["max_vm_pu"] = 1.05

    # Slack (ext_grid)  ---- 중요: controllable=True 권장 (limit/버스전압제약 반영)
    eg = pp.create_ext_grid(net, bus=slack_bus, vm_pu=slack_vm_pu, va_degree=0.0, name="Slack")
    if "controllable" not in net.ext_grid.columns:
        net.ext_grid["controllable"] = True
    net.ext_grid.at[eg, "controllable"] = True

    net.ext_grid.loc[eg, ["min_p_mw","max_p_mw"]] = [0.0, 300.0]     # 발전만 허용
    net.ext_grid.loc[eg, ["min_q_mvar","max_q_mvar"]] = [-300.0, 300.0]
    net.ext_grid.loc[eg, ["min_vm_pu","max_vm_pu"]] = [0.95, 1.05]

    pp.create_poly_cost(net, element=eg, et="ext_grid",
                        cp2_eur_per_mw2=0.010, cp1_eur_per_mw=9.0, cp0_eur=0.0)

    # Generators (gen)
    gen_data = [
        (2,   10.0,  0.0,  80.0,  -200.0, 200.0, 0.050,   5.0,  0.0),
        (3,   50.0,  0.0, 100.0,  -200.0, 200.0, 0.025,   8.0,  0.0),
        (5,   60.0,  0.0, 120.0,  -200.0, 200.0, 0.020,   9.0,  0.0),
        (7,   70.0,  0.0, 200.0,  -200.0, 200.0, 0.005,  12.0,  0.0),
    ]
    for bus_id, p0, pmin, pmax, qmin, qmax, c2, c1, c0 in gen_data:
        g = pp.create_gen(net, bus=buses[bus_id], p_mw=p0, vm_pu=1.0,
                          name=f"Gen@{bus_id}", controllable=True)
        net.gen.loc[g, ["min_p_mw","max_p_mw"]] = [pmin, pmax]
        net.gen.loc[g, ["min_q_mvar","max_q_mvar"]] = [qmin, qmax]
        net.gen.loc[g, ["min_vm_pu","max_vm_pu"]] = [0.95, 1.05]
        pp.create_poly_cost(net, element=g, et="gen",
                            cp2_eur_per_mw2=c2, cp1_eur_per_mw=c1, cp0_eur=c0)

    # Loads (고정 수요)
    load_data = [
        (1,  90.0, 32.0),
        (4,  70.0, 24.0),
        (5,  90.0, 28.0),
        (6,  80.0, 22.0),
        (7,  90.0, 22.0),
        (8,  50.0, 18.0),
    ]
    for bus_id, p, q in load_data:
        pp.create_load(net, bus=buses[bus_id], p_mw=p, q_mvar=q, name=f"Load@{bus_id}")

    # Lines
    Z_base = (10.0 ** 2) / net.sn_mva
    line_params = [
        (0, 3, 0.00000, 0.05760),
        (3, 4, 0.01700, 0.09200),
        (4, 5, 0.03900, 0.17000),
        (2, 5, 0.00000, 0.05860),
        (5, 6, 0.01190, 0.10080),
        (6, 7, 0.00850, 0.07200),
        (7, 1, 0.00000, 0.06250),
        (8, 3, 0.01000, 0.08500),
        (8, 7, 0.03200, 0.16100), # mesh면 본 선로 추가
    ]
    for fb, tb, r_pu, x_pu in line_params:
        pp.create_line_from_parameters(
            net, from_bus=buses[fb], to_bus=buses[tb], length_km=1.0,
            r_ohm_per_km=r_pu * Z_base, x_ohm_per_km=x_pu * Z_base,
            c_nf_per_km=0.0, max_i_ka=20.0,
            r0_ohm_per_km=r_pu * Z_base, x0_ohm_per_km=x_pu * Z_base, c0_nf_per_km=0.0,
            name=f"Line {fb}-{tb}"
        )

    # (추가 제약 1) 선로 loading 제약: OPF에서 사용되는 표준 컬럼
    # 너무 빡빡하면 infeasible이 되므로, 일단 크게 두고(예: 1e6%) 점진적으로 낮추는 것을 권장
    net.line["max_loading_percent"] = float(line_max_loading_percent)

    # (추가 제약 2) 발전기 P–Q capability curve: P에 따라 Q 한계가 바뀌도록(권장)
    smax_gen_mva = {2: 90.0, 3: 120.0, 5: 150.0, 7: 230.0}
    ok = _apply_circle_qcapability_curve_to_gens(net, smax_gen_mva, n_points=7,
                                                 curve_style="straightLineYValues")

    if not ok:
        # capability curve 기능이 없는 버전이면, 기존처럼 “상수 Q박스” 근사로라도 보수적으로 제한
        for gi in net.gen.index:
            bus = int(net.gen.at[gi, "bus"])
            Smax = smax_gen_mva.get(bus, None)
            if Smax is None:
                continue
            Pmax = float(net.gen.at[gi, "max_p_mw"])
            qcap = math.sqrt(max(Smax**2 - Pmax**2, 0.0))
            net.gen.at[gi, "min_q_mvar"] = max(float(net.gen.at[gi, "min_q_mvar"]), -qcap)
            net.gen.at[gi, "max_q_mvar"] = min(float(net.gen.at[gi, "max_q_mvar"]),  qcap)

    return net

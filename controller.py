"""
Transition controller + battery simulation.

Rules (per forecast hour):
  * Renewable Storage Mode : demand > 160 kW  OR  hour in 18..22 (inclusive)
  * Main Grid Mode         : otherwise
  * Critical Overload      : demand >= 200 kW (independent of mode)
"""
import pandas as pd

HIGH_DEMAND_KW = 160.0
OVERLOAD_KW = 200.0
EVENING_START, EVENING_END = 18, 22       # inclusive

STORAGE = "Renewable Storage Mode"
GRID = "Main Grid Mode"

# --- battery assumptions (documented in README; change freely) -------------
BATTERY_CAPACITY_KWH = 1500.0
INITIAL_SOC_PCT = 80.0
CHARGE_RATE_KW = 200.0       # max charging power from the grid/renewables
SOLAR_CHARGE_FACTOR = 0.20   # extra kW charged per W/m2 of irradiance in Grid mode
DISCHARGE_SHARE = 0.50       # fraction of load the battery covers in Storage mode
LOW_SOC_WARN_PCT = 20.0


def decide(demand_kw: float, hour: int) -> dict:
    in_window = EVENING_START <= int(hour) <= EVENING_END
    storage = demand_kw > HIGH_DEMAND_KW or in_window
    if demand_kw > HIGH_DEMAND_KW and in_window:
        reason = "High demand (>160 kW) and evening window (18:00-22:00)"
    elif demand_kw > HIGH_DEMAND_KW:
        reason = "Demand above 160 kW"
    elif in_window:
        reason = "Evening window 18:00-22:00"
    else:
        reason = "Demand <= 160 kW and outside 18:00-22:00"
    return {
        "mode": STORAGE if storage else GRID,
        "overload": demand_kw >= OVERLOAD_KW,
        "reason": reason,
    }


def run_controller(rows: list, initial_soc: float = INITIAL_SOC_PCT) -> list:
    """
    rows: list of dicts with timestamp, predicted_demand_kw, solar_irradiance_wm2.
    Adds mode / overload / reason / battery_soc_pct / battery_flag to each row.
    SoC is simulated hour by hour (1h steps):
      Storage mode : battery discharges DISCHARGE_SHARE * demand kWh
      Grid mode    : battery charges CHARGE_RATE_KW + solar bonus
    """
    soc_kwh = BATTERY_CAPACITY_KWH * initial_soc / 100.0
    out = []
    for r in rows:
        demand = float(r["predicted_demand_kw"])
        hour = pd.Timestamp(r["timestamp"]).hour
        d = decide(demand, hour)
        flag = None
        if d["mode"] == STORAGE:
            need = DISCHARGE_SHARE * demand
            if need > soc_kwh:
                flag = "Battery depleted - residual load falls back to grid"
                need = soc_kwh
            soc_kwh -= need
        else:
            gain = CHARGE_RATE_KW + SOLAR_CHARGE_FACTOR * float(r.get("solar_irradiance_wm2", 0) or 0)
            soc_kwh = min(BATTERY_CAPACITY_KWH, soc_kwh + gain)
        soc_pct = round(100.0 * soc_kwh / BATTERY_CAPACITY_KWH, 1)
        if flag is None and soc_pct < LOW_SOC_WARN_PCT:
            flag = "Low battery"
        out.append({**r, **d, "hour": hour, "battery_soc_pct": soc_pct, "battery_flag": flag})
    return out

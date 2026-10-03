"""สถิติและการคาดการณ์การใช้ไฟรายเดือนจากข้อมูลจริง

แทนการสร้างข้อมูลสุ่ม: โมเดลหลายแบบถูกทดสอบย้อนหลัง (walk-forward backtest)
แล้วเลือกแบบที่คลาดเคลื่อนน้อยที่สุดกับข้อมูลของผู้ใช้เอง
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

MIN_POINTS = 3     # ข้อมูลน้อยสุดที่ยอมคาดการณ์
MIN_FOLDS = 4      # โมเดลต้องทดสอบย้อนหลังได้อย่างน้อยกี่รอบจึงนำมาเปรียบเทียบ
MAX_FOLDS = 6


# ---------------------------------------------------------------------------
# สถิติพื้นฐาน
# ---------------------------------------------------------------------------
def calculate_stats(df: pd.DataFrame) -> dict:
    d = df.reset_index(drop=True)
    x = d["kwh"].astype(float)
    n = len(x)
    mean = float(x.mean())
    last = float(x.iloc[-1])
    mom = (last - x.iloc[-2]) / x.iloc[-2] * 100 if n >= 2 and x.iloc[-2] else None
    prev_year = d.loc[d["month"] == d["month"].iloc[-1] - pd.DateOffset(months=12), "kwh"]
    yoy = (last - float(prev_year.iloc[0])) / float(prev_year.iloc[0]) * 100 if len(prev_year) and prev_year.iloc[0] else None
    return {
        "n": n,
        "mean": mean,
        "max": float(x.max()), "max_month": d["month"].iloc[int(x.idxmax())],
        "min": float(x.min()), "min_month": d["month"].iloc[int(x.idxmin())],
        "volatility": float(x.std(ddof=1) / mean * 100) if n >= 2 and mean else 0.0,
        "change_total": float((last - x.iloc[0]) / x.iloc[0] * 100) if x.iloc[0] else 0.0,
        "mom": None if mom is None else float(mom),
        "yoy": None if yoy is None else float(yoy),
        "slope": float(np.polyfit(np.arange(n), x, 1)[0]) if n >= 3 else 0.0,
    }


def temperature_effect(df: pd.DataFrame) -> Optional[dict]:
    """ความสัมพันธ์อุณหภูมิกับหน่วยไฟ (ต้องมีข้อมูลอุณหภูมิอย่างน้อย 6 เดือน)"""
    d = df.dropna(subset=["temp", "kwh"])
    if len(d) < 6 or d["temp"].std() == 0 or d["kwh"].std() == 0:
        return None
    r = float(np.corrcoef(d["temp"], d["kwh"])[0, 1])
    slope, intercept = np.polyfit(d["temp"], d["kwh"], 1)
    if not np.isfinite(r) or not np.isfinite(slope):
        return None
    return {"r": r, "slope": float(slope), "intercept": float(intercept), "n": len(d)}


# ---------------------------------------------------------------------------
# โมเดลคาดการณ์ (ทุกตัวรับ y = หน่วยไฟในอดีต, months = เดือนของ y + เดือนที่จะทำนายต่อท้าย)
# ---------------------------------------------------------------------------
def _pred_mean3(y, months):
    return float(np.mean(y[-3:]))


def _pred_trend(y, months):
    k = min(len(y), 12)
    slope, intercept = np.polyfit(np.arange(k), y[-k:], 1)
    return max(0.0, float(slope * k + intercept))


def _pred_seasonal(y, months):
    return float(y[-12])  # เดือนเดียวกันของปีก่อน


def _rf_features(y, months, i):
    lags = np.asarray(y[i - 3:i], dtype=float)
    base = float(lags.mean()) or 1.0
    angle = 2 * np.pi * (months[i].month - 1) / 12
    return [lags[2] / base, lags[1] / base, lags[0] / base, np.sin(angle), np.cos(angle)], base


def _pred_rf(y, months):
    """Random Forest ทำนายสัดส่วนเทียบค่าเฉลี่ย 3 เดือนก่อนหน้า + ฤดูกาล (ลดปัญหา RF extrapolate ไม่ได้)"""
    n = len(y)
    rows, targets = [], []
    for i in range(3, n):
        feats, base = _rf_features(y, months, i)
        rows.append(feats)
        targets.append(y[i] / base)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=1)
    model.fit(np.array(rows), np.array(targets))
    feats, base = _rf_features(y, months, n)
    return max(0.0, float(model.predict(np.array([feats]))[0]) * base)


@dataclass(frozen=True)
class _Model:
    name: str
    fn: Callable
    min_len: int


MODELS = [
    _Model("ค่าเฉลี่ย 3 เดือนล่าสุด", _pred_mean3, 3),
    _Model("แนวโน้มเส้นตรง", _pred_trend, 4),
    _Model("เดือนเดียวกันของปีก่อน", _pred_seasonal, 12),
    _Model("Random Forest", _pred_rf, 9),
]


@dataclass
class ForecastResult:
    next_month: pd.Timestamp
    kwh: float
    low: float
    high: float
    model_name: str
    reliability: str
    n_points: int
    mape: Optional[float]
    backtest: pd.DataFrame


def _reliability(n: int, mape: Optional[float]) -> str:
    if mape is None:
        return "ต่ำมาก (ข้อมูลน้อยเกินกว่าจะทดสอบย้อนหลัง)"
    if n < 12 or mape > 15:
        return "ต่ำ"
    if n < 24 or mape > 8:
        return "ปานกลาง"
    return "ดี"


def forecast_next_month(df: pd.DataFrame) -> ForecastResult:
    """คาดการณ์หน่วยไฟเดือนถัดไป พร้อมช่วงความไม่แน่นอนและผลทดสอบย้อนหลัง"""
    d = df.sort_values("month").reset_index(drop=True)
    y = d["kwh"].to_numpy(dtype=float)
    n = len(y)
    if n < MIN_POINTS:
        raise ValueError(f"ต้องมีข้อมูลอย่างน้อย {MIN_POINTS} เดือน")
    months = list(pd.to_datetime(d["month"]))
    next_month = months[-1] + pd.DateOffset(months=1)
    months_ext = months + [next_month]

    candidates = [m for m in MODELS if n - m.min_len >= MIN_FOLDS]
    backtest = pd.DataFrame(columns=["วิธี", "คลาดเคลื่อนเฉลี่ย (kWh)", "คลาดเคลื่อนเฉลี่ย (%)", "จำนวนรอบทดสอบ", "ถูกเลือก"])
    rmse = mape = None
    chosen = MODELS[0]
    if candidates:
        folds = min(MAX_FOLDS, min(n - m.min_len for m in candidates))
        stats = []
        for m in candidates:
            errs = np.array([y[i] - m.fn(y[:i], months_ext[:i + 1]) for i in range(n - folds, n)])
            truth = y[n - folds:]
            pct = np.abs(errs[truth > 0]) / truth[truth > 0] * 100
            stats.append((m, float(np.mean(np.abs(errs))), float(np.sqrt(np.mean(errs ** 2))),
                          float(np.mean(pct)) if len(pct) else float("nan")))
        best_mae = min(s[1] for s in stats)
        chosen_stat = next(s for s in stats if s[1] <= best_mae * 1.03)  # เท่ากันโดยประมาณ เลือกวิธีที่ง่ายกว่า
        chosen, rmse, mape = chosen_stat[0], chosen_stat[2], chosen_stat[3]
        backtest = pd.DataFrame({
            "วิธี": [s[0].name for s in stats],
            "คลาดเคลื่อนเฉลี่ย (kWh)": [round(s[1], 1) for s in stats],
            "คลาดเคลื่อนเฉลี่ย (%)": [round(s[3], 1) for s in stats],
            "จำนวนรอบทดสอบ": folds,
            "ถูกเลือก": ["✓" if s[0].name == chosen.name else "" for s in stats],
        })

    pred = max(0.0, float(chosen.fn(y, months_ext)))
    half = 1.28 * rmse if rmse is not None else 0.15 * pred  # ช่วงประมาณ 80%
    return ForecastResult(
        next_month=next_month, kwh=pred, low=max(0.0, pred - half), high=pred + half,
        model_name=chosen.name, reliability=_reliability(n, mape), n_points=n, mape=mape, backtest=backtest,
    )

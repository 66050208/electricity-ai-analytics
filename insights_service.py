"""วิเคราะห์เชิงลึก: จัดกลุ่มรูปแบบการใช้ไฟ (Clustering) และอธิบายโมเดลคาดการณ์ (XAI)

ไม่พึ่ง Streamlit เพื่อให้ทดสอบแยกได้
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from data_service import thai_month_label

MIN_MONTHS_CLUSTER = 6
MIN_MONTHS_XAI = 12
HOT_MONTHS = {3, 4, 5}  # มี.ค.–พ.ค. หน้าร้อนของไทย


# ---------------------------------------------------------------------------
# 1) Clustering: แบ่งเดือนของบ้านนี้เป็นกลุ่ม "ใช้ไฟสูง / ปกติ / ต่ำ"
# ---------------------------------------------------------------------------
def cluster_months(df: pd.DataFrame, k: int = 3) -> Optional[dict]:
    """K-Means บนฟีเจอร์รายเดือน: หน่วยไฟ, ฤดูกาล (sin/cos ของเดือน), อุณหภูมิ (ถ้ามีครบ)

    คืน {table, summary, features, inertia} หรือ None ถ้าข้อมูลน้อยเกิน
    """
    d = df.loc[~df["filled"]].reset_index(drop=True)
    if len(d) < MIN_MONTHS_CLUSTER:
        return None
    k = int(min(k, max(2, len(d) // 3)))
    angle = 2 * np.pi * (d["month"].dt.month - 1) / 12
    feats = pd.DataFrame({"kwh": d["kwh"], "season_sin": np.sin(angle), "season_cos": np.cos(angle)})
    names = ["หน่วยไฟ", "ฤดูกาล"]
    if d["temp"].notna().all():
        feats["temp"] = d["temp"]
        names.append("อุณหภูมิ")
    X = StandardScaler().fit_transform(feats)
    X[:, 0] *= 2.0  # ให้น้ำหนักหน่วยไฟมากกว่าฤดูกาล เพราะเป้าหมายคือแบ่งระดับการใช้ไฟ
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)

    d = d.assign(cluster=km.labels_)
    order = d.groupby("cluster")["kwh"].mean().sort_values().index.tolist()
    label_sets = {2: ["ใช้ไฟต่ำ", "ใช้ไฟสูง"], 3: ["ใช้ไฟต่ำ", "ใช้ไฟปกติ", "ใช้ไฟสูง"]}
    labels = label_sets.get(k, [f"กลุ่ม {i + 1}" for i in range(k)])
    name_of = {c: labels[i] for i, c in enumerate(order)}
    d["กลุ่ม"] = d["cluster"].map(name_of)

    summary = []
    for c in order:
        part = d[d["cluster"] == c]
        months = part["month"].dt.month
        summary.append({
            "กลุ่ม": name_of[c],
            "จำนวนเดือน": len(part),
            "เฉลี่ย (kWh)": round(float(part["kwh"].mean()), 0),
            "ช่วง (kWh)": f"{part['kwh'].min():,.0f}–{part['kwh'].max():,.0f}",
            "หน้าร้อน (%)": round(float(months.isin(HOT_MONTHS).mean() * 100), 0),
            "อุณหภูมิเฉลี่ย (°C)": round(float(part["temp"].mean()), 1) if part["temp"].notna().any() else None,
            "เดือน": ", ".join(thai_month_label(m) for m in part["month"]),
        })
    return {"table": d[["month", "kwh", "temp", "กลุ่ม"]], "summary": pd.DataFrame(summary),
            "features": names, "k": k}


def household_profile(df: pd.DataFrame) -> dict:
    """จัดรูปแบบบ้านจากสถิติ: ใช้ไฟตามฤดูกาล / เพิ่มขึ้นต่อเนื่อง / ลดลง / คงที่ / ไม่สม่ำเสมอ"""
    d = df.loc[~df["filled"]]
    x = d["kwh"].to_numpy(float)
    n = len(x)
    mean = float(x.mean()) if n else 0.0
    cv = float(x.std(ddof=1) / mean * 100) if n >= 2 and mean else 0.0
    slope_pct = float(np.polyfit(np.arange(n), x, 1)[0] / mean * 100) if n >= 4 and mean else 0.0
    hot = d.loc[d["month"].dt.month.isin(HOT_MONTHS), "kwh"]
    cool = d.loc[d["month"].dt.month.isin({11, 12, 1}), "kwh"]
    season_gap = float((hot.mean() - cool.mean()) / mean * 100) if len(hot) and len(cool) and mean else None

    if season_gap is not None and season_gap >= 15:
        key, text = "seasonal", "สายแอร์หน้าร้อน: ใช้ไฟพุ่งช่วง มี.ค.–พ.ค. แอร์น่าจะเป็นตัวหลัก จุดที่ลดได้มากที่สุดคือแอร์ช่วงหน้าร้อน"
    elif slope_pct >= 1.5:
        key, text = "rising", "ใช้ไฟเพิ่มขึ้นต่อเนื่อง: ดูว่ามีเครื่องใช้ไฟฟ้าใหม่ คนอยู่บ้านเพิ่ม หรือเครื่องเก่าที่กินไฟมากขึ้นหรือไม่"
    elif slope_pct <= -1.5:
        key, text = "falling", "ใช้ไฟลดลงต่อเนื่อง: สิ่งที่ทำอยู่ได้ผล รักษาพฤติกรรมนี้ไว้"
    elif cv >= 20:
        key, text = "irregular", "ใช้ไฟไม่สม่ำเสมอ: บางเดือนสูงต่ำต่างกันมาก ลองดูว่าเดือนที่สูงมีเหตุการณ์อะไร"
    else:
        key, text = "steady", "ใช้ไฟคงที่: รูปแบบสม่ำเสมอ ลดได้จากเครื่องที่เปิดตลอด เช่น ตู้เย็น ไฟสแตนด์บาย"
    return {"key": key, "text": text, "cv": round(cv, 1), "trend_pct_per_month": round(slope_pct, 2),
            "season_gap_pct": None if season_gap is None else round(season_gap, 1)}


# ---------------------------------------------------------------------------
# 2) XAI: ปัจจัยไหนมีผลต่อการคาดการณ์ (Random Forest + Permutation Importance)
# ---------------------------------------------------------------------------
def _xai_frame(df: pd.DataFrame, use_temp: bool):
    d = df.sort_values("month").reset_index(drop=True)
    y = d["kwh"].to_numpy(float)
    rows, target, months = [], [], []
    for i in range(3, len(d)):
        base = float(y[i - 3:i].mean()) or 1.0
        angle = 2 * np.pi * (d["month"].iloc[i].month - 1) / 12
        row = {"lag1": y[i - 1] / base, "lag2": y[i - 2] / base, "lag3": y[i - 3] / base,
               "season_sin": np.sin(angle), "season_cos": np.cos(angle)}
        if use_temp:
            row["temp"] = d["temp"].iloc[i]
        rows.append(row)
        target.append(y[i] / base)
        months.append(d["month"].iloc[i])
    return pd.DataFrame(rows), np.array(target), months


GROUPS = {
    "การใช้ไฟเดือนที่แล้ว": ["lag1"],
    "การใช้ไฟ 2–3 เดือนก่อน": ["lag2", "lag3"],
    "ฤดูกาล (เดือนของปี)": ["season_sin", "season_cos"],
    "อุณหภูมิ": ["temp"],
}


def _walk_forward_mae(X: pd.DataFrame, y: np.ndarray, folds: int) -> float:
    errs = []
    for i in range(len(X) - folds, len(X)):
        m = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=1)
        m.fit(X.iloc[:i], y[:i])
        errs.append(abs(m.predict(X.iloc[[i]])[0] - y[i]))
    return float(np.mean(errs))


def explain_forecast(df: pd.DataFrame) -> Optional[dict]:
    """คืน {importance, temp_test} หรือ None ถ้าข้อมูลน้อยกว่า 12 เดือน

    importance: สัดส่วนผลของแต่ละกลุ่มปัจจัย (รวม 100%) จาก permutation importance
    temp_test: เทียบความคลาดเคลื่อนเมื่อใส่/ไม่ใส่อุณหภูมิในโมเดล (ถ้ามีอุณหภูมิ)
    """
    if len(df) < MIN_MONTHS_XAI:
        return None
    has_temp = df["temp"].notna().all()
    X, y, _ = _xai_frame(df, use_temp=has_temp)
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1).fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=30, random_state=42)
    raw = dict(zip(X.columns, np.clip(perm.importances_mean, 0, None)))
    grouped = {g: sum(raw.get(c, 0.0) for c in cols) for g, cols in GROUPS.items() if any(c in raw for c in cols)}
    total = sum(grouped.values()) or 1.0
    importance = pd.DataFrame({"ปัจจัย": list(grouped), "ผลต่อการคาดการณ์ (%)": [v / total * 100 for v in grouped.values()]})
    importance = importance.sort_values("ผลต่อการคาดการณ์ (%)", ascending=False).reset_index(drop=True)

    temp_test = None
    if has_temp:
        folds = int(min(6, len(X) - 6))
        if folds >= 3:
            base_avg = df.sort_values("month")["kwh"].rolling(3).mean().shift(1).iloc[3:].to_numpy()
            with_t = _walk_forward_mae(X, y, folds)
            without_t = _walk_forward_mae(X.drop(columns="temp"), y, folds)
            scale = float(np.nanmean(base_avg[-folds:]))
            temp_test = {"mae_with_temp_kwh": round(with_t * scale, 1), "mae_without_temp_kwh": round(without_t * scale, 1),
                         "folds": folds, "temp_helps": with_t < without_t}
    return {"importance": importance, "temp_test": temp_test, "n": len(X), "has_temp": has_temp}


def describe_importance(importance: pd.DataFrame) -> str:
    top = importance.iloc[0]
    return (f"ปัจจัยที่มีผลต่อการคาดการณ์มากที่สุดคือ “{top['ปัจจัย']}” ({top['ผลต่อการคาดการณ์ (%)']:.0f}%) "
            "ตัวเลขนี้บอกว่าโมเดลพึ่งปัจจัยไหน ไม่ได้แปลว่าเป็นสาเหตุโดยตรง")

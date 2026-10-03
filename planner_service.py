"""จำลองสถานการณ์ (What-if), ความคุ้มค่าการซื้อเครื่องใหม่, พยากรณ์อากาศ 7 วัน และแปลงป้ายเครื่องใช้ไฟฟ้า

สมมติฐานทุกตัวเขียนไว้เป็นค่าคงที่ด้านล่าง เพื่อแสดงในหน้าเว็บและแก้ได้ง่าย
ไม่พึ่ง Streamlit เพื่อให้ทดสอบแยกได้
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from electricity_service import COL_DAYS, COL_DUTY, COL_HOURS, COL_NAME, COL_QTY, COL_WATT, appliance_energy

# ---------------------------------------------------------------------------
# สมมติฐาน (แสดงให้ผู้ใช้เห็นในหน้าเว็บ)
# ---------------------------------------------------------------------------
SHARE_PER_PERSON = 0.40      # สัดส่วนการใช้ไฟที่แปรตามจำนวนคน (อาบน้ำอุ่น ซักผ้า ชาร์จ ไฟห้อง) ที่เหลือคงที่ เช่น ตู้เย็น
AC_SAVING_PER_DEGREE = 0.07  # ตั้งแอร์สูงขึ้น 1°C ประหยัดไฟแอร์ราว 7% (คำแนะนำทั่วไปอยู่ราว 5–10%)
WFH_EXTRA_HOURS = 8          # ทำงานที่บ้าน 1 วัน เปิดแอร์/คอมเพิ่มกี่ชั่วโมง
WFH_COMPUTER_WATT = 60
WEEKS_PER_MONTH = 4.33
DEFAULT_AC_WATT, DEFAULT_AC_DUTY = 1000, 0.7
HOT_DAY_C = 35.0             # อุณหภูมิสูงสุดตั้งแต่นี้ถือว่าร้อนจัด
ASSUMPTIONS = [
    f"การใช้ไฟราว {SHARE_PER_PERSON:.0%} แปรตามจำนวนคนในบ้าน ที่เหลือ (ตู้เย็น ไฟสแตนด์บาย) คงที่",
    f"ตั้งแอร์สูงขึ้น 1°C ลดไฟของแอร์ราว {AC_SAVING_PER_DEGREE:.0%}",
    f"ทำงานที่บ้าน 1 วัน = เปิดแอร์และคอมเพิ่ม {WFH_EXTRA_HOURS} ชั่วโมง",
    "กำลังไฟแอร์ใช้จากตาราง “เครื่องใช้ไฟฟ้า” (รายการที่ชื่อมีคำว่า แอร์) ถ้าไม่มีใช้ 1,000 วัตต์ ทำงานจริง 70%",
]


def _ac_rows(appliances: Optional[pd.DataFrame]) -> pd.DataFrame:
    if appliances is None or len(appliances) == 0:
        return pd.DataFrame()
    names = appliances[COL_NAME].fillna("").astype(str).str.lower()
    return appliances[names.str.contains("แอร์|air|เครื่องปรับอากาศ", regex=True)]


def ac_profile(appliances: Optional[pd.DataFrame]) -> dict:
    """กำลังไฟแอร์ต่อเครื่อง (วัตต์), %ทำงานจริง, kWh/เดือนของแอร์ทั้งหมด"""
    ac = _ac_rows(appliances)
    if ac.empty:
        return {"watt": DEFAULT_AC_WATT, "duty": DEFAULT_AC_DUTY, "kwh": None, "count": 0, "from_table": False}
    watt = pd.to_numeric(ac[COL_WATT], errors="coerce").fillna(0)
    duty = pd.to_numeric(ac[COL_DUTY], errors="coerce").fillna(100) / 100
    qty = pd.to_numeric(ac[COL_QTY], errors="coerce").fillna(1)
    return {"watt": float(np.average(watt, weights=qty.clip(lower=1))) or DEFAULT_AC_WATT,
            "duty": float(np.average(duty, weights=qty.clip(lower=1))) or DEFAULT_AC_DUTY,
            "kwh": float(appliance_energy(ac)["kwh"].sum()), "count": int(qty.sum()), "from_table": True}


# ---------------------------------------------------------------------------
# What-if
# ---------------------------------------------------------------------------
def what_if(base_kwh: float, appliances: Optional[pd.DataFrame] = None, people_now: int = 3, people_after: int = 3,
            wfh_days_per_week: float = 0, ac_temp_up: float = 0, ac_hours_change: float = 0,
            ac_units_used: Optional[int] = None) -> dict:
    """คืน {base_kwh, new_kwh, items:[{เรื่อง, เปลี่ยน_kwh}]} (+ = ใช้ไฟเพิ่ม)"""
    base = max(0.0, float(base_kwh))
    ac = ac_profile(appliances)
    units = ac_units_used if ac_units_used is not None else max(1, ac["count"])
    items = []

    people_now = max(1, int(people_now))
    if people_after != people_now:
        delta = base * SHARE_PER_PERSON * (int(people_after) - people_now) / people_now
        items.append({"เรื่อง": f"คนในบ้าน {people_now} → {int(people_after)} คน", "เปลี่ยน_kwh": delta})
    if wfh_days_per_week:
        per_day = (ac["watt"] / 1000 * ac["duty"] + WFH_COMPUTER_WATT / 1000) * WFH_EXTRA_HOURS
        items.append({"เรื่อง": f"ทำงานที่บ้าน {wfh_days_per_week:g} วัน/สัปดาห์",
                      "เปลี่ยน_kwh": per_day * float(wfh_days_per_week) * WEEKS_PER_MONTH})
    if ac_hours_change:
        items.append({"เรื่อง": f"เปิดแอร์ {'เพิ่ม' if ac_hours_change > 0 else 'ลด'} {abs(ac_hours_change):g} ชม./วัน ({units} เครื่อง)",
                      "เปลี่ยน_kwh": ac["watt"] / 1000 * ac["duty"] * float(ac_hours_change) * 30 * units})
    if ac_temp_up:
        ac_kwh = ac["kwh"] if ac["kwh"] is not None else ac["watt"] / 1000 * ac["duty"] * 8 * 30
        ac_kwh += sum(i["เปลี่ยน_kwh"] for i in items if "แอร์" in i["เรื่อง"])  # รวมชั่วโมงที่ปรับแล้ว
        factor = 1 - (1 - AC_SAVING_PER_DEGREE) ** float(ac_temp_up) if ac_temp_up > 0 else -((1 + AC_SAVING_PER_DEGREE) ** abs(ac_temp_up) - 1)
        items.append({"เรื่อง": f"ตั้งแอร์ {'สูงขึ้น' if ac_temp_up > 0 else 'ต่ำลง'} {abs(ac_temp_up):g}°C",
                      "เปลี่ยน_kwh": -max(0.0, ac_kwh) * factor})

    new = max(0.0, base + sum(i["เปลี่ยน_kwh"] for i in items))
    return {"base_kwh": base, "new_kwh": new, "items": items}


# ---------------------------------------------------------------------------
# คุ้มไหมถ้าซื้อเครื่องใหม่
# ---------------------------------------------------------------------------
def payback(old_watt: float, new_watt: float, hours_per_day: float, price: float, rate: float,
            days_per_month: float = 30, old_duty: float = 100, new_duty: float = 100) -> dict:
    hours = float(np.clip(hours_per_day, 0, 24))
    days = float(np.clip(days_per_month, 0, 31))
    old_kwh = float(old_watt) / 1000 * hours * days * float(old_duty) / 100
    new_kwh = float(new_watt) / 1000 * hours * days * float(new_duty) / 100
    save_kwh = old_kwh - new_kwh
    save_baht = save_kwh * float(rate)
    months = float(price) / save_baht if save_baht > 0 else None
    return {
        "old_kwh": old_kwh, "new_kwh": new_kwh, "save_kwh": save_kwh,
        "save_baht_month": save_baht, "save_baht_year": save_baht * 12,
        "payback_months": months, "payback_years": None if months is None else months / 12,
        "save_10y": save_baht * 120 - float(price),
    }


# ---------------------------------------------------------------------------
# พยากรณ์อากาศ 7 วัน → คำแนะนำ
# ---------------------------------------------------------------------------
def weather_outlook(forecast: pd.DataFrame, temp_effect: Optional[dict], recent_temp: Optional[float],
                    ac_kwh_month: Optional[float] = None) -> dict:
    """forecast: คอลัมน์ date, tmax, tmin, tmean, rain_mm

    คืน {days: ตารางคำแนะนำรายวัน, mean_temp, impact_kwh_month (ถ้าคำนวณได้), summary}
    """
    if forecast is None or forecast.empty:
        return {"days": pd.DataFrame(), "mean_temp": None, "impact_kwh_month": None, "summary": ""}
    f = forecast.copy()

    def advice(row):
        if row["tmax"] >= HOT_DAY_C:
            return "🔥 ร้อนจัด ปิดม่านกันแดดช่วงบ่าย ตั้งแอร์ 26–27°C เปิดพัดลมช่วย"
        if row.get("rain_mm", 0) and row["rain_mm"] >= 10:
            return "🌧️ ฝนตก อากาศเย็นลง ลองใช้พัดลมแทนแอร์ งดใช้เครื่องอบผ้า ตากในร่มแทน"
        if row["tmax"] <= 31:
            return "🌤️ อากาศไม่ร้อนมาก เปิดหน้าต่างระบายอากาศ ลดเวลาเปิดแอร์ได้"
        return "☀️ ร้อนปกติ เปิดแอร์เฉพาะห้องที่ใช้ ปิดเมื่อออกจากห้อง"

    f["คำแนะนำ"] = f.apply(advice, axis=1)
    mean_temp = float(f["tmean"].mean())
    impact = None
    if temp_effect and temp_effect.get("r", 0) >= 0.3 and temp_effect.get("slope", 0) > 0 and recent_temp is not None:
        impact = float(temp_effect["slope"] * (mean_temp - recent_temp))
    hot_days = int((f["tmax"] >= HOT_DAY_C).sum())
    if impact is not None and abs(impact) >= 5:
        direction = "เพิ่มขึ้น" if impact > 0 else "ลดลง"
        summary = (f"7 วันข้างหน้าอุณหภูมิเฉลี่ย {mean_temp:.1f}°C เทียบกับเดือนล่าสุด {recent_temp:.1f}°C "
                   f"จากข้อมูลบ้านคุณ ถ้าอากาศเป็นแบบนี้ทั้งเดือน การใช้ไฟน่าจะ{direction}ราว {abs(impact):,.0f} หน่วย")
    else:
        summary = f"7 วันข้างหน้าอุณหภูมิเฉลี่ย {mean_temp:.1f}°C"
    if hot_days:
        summary += f" มีวันร้อนจัด (≥{HOT_DAY_C:.0f}°C) {hot_days} วัน"
    return {"days": f, "mean_temp": mean_temp, "impact_kwh_month": impact, "hot_days": hot_days, "summary": summary}


# ---------------------------------------------------------------------------
# ป้ายเครื่องใช้ไฟฟ้า (ที่ AI อ่าน) → แถวในตาราง
# ---------------------------------------------------------------------------
TYPE_DEFAULTS = {  # ชั่วโมง/วัน, %ทำงานจริง ค่าเริ่มต้นตามประเภทเครื่อง
    "แอร์": (8, 70), "ตู้เย็น": (24, 40), "ทีวี": (4, 100), "พัดลม": (6, 100), "ไมโครเวฟ": (0.2, 100),
    "หม้อหุงข้าว": (1, 100), "เครื่องซักผ้า": (1, 100), "น้ำอุ่น": (0.3, 100), "เตารีด": (0.3, 100),
    "คอม": (5, 100), "โน้ตบุ๊ก": (5, 100), "หลอดไฟ": (5, 100),
}


def label_to_row(info: dict) -> tuple[dict, list[str]]:
    """แปลงข้อมูลจากป้าย/ฉลากเบอร์ 5 เป็นแถวของตารางเครื่องใช้ไฟฟ้า คืน (แถว, หมายเหตุ)"""
    notes: list[str] = []
    name = (info.get("appliance_type") or "เครื่องใช้ไฟฟ้า").strip()
    if info.get("brand"):
        name = f"{name} {info['brand']}".strip()
    if info.get("cooling_btu"):
        name = f"{name} {info['cooling_btu']:,.0f} BTU"

    watt = info.get("power_watt")
    hours, duty = next((v for k, v in TYPE_DEFAULTS.items() if k in name), (4, 100))
    if not watt and info.get("voltage_v") and info.get("current_a"):
        watt = info["voltage_v"] * info["current_a"]
        notes.append(f"ไม่พบกำลังไฟบนป้าย ประมาณจาก แรงดัน × กระแส = {watt:,.0f} วัตต์ (มักสูงกว่าการใช้จริง)")
    if not watt and info.get("annual_kwh"):
        watt, hours, duty = info["annual_kwh"] * 1000 / 8760, 24, 100
        notes.append(f"ใช้ค่าจากฉลากเบอร์ 5: {info['annual_kwh']:,.0f} หน่วย/ปี (≈ {info['annual_kwh'] / 12:,.0f} หน่วย/เดือน)")
    if not watt:
        notes.append("อ่านกำลังไฟไม่ได้ กรุณากรอกวัตต์เอง")
    if info.get("energy_label_no5"):
        notes.append("มีฉลากประหยัดไฟเบอร์ 5" + (f" ({info['label_stars']} ดาว)" if info.get("label_stars") else ""))
    row = {COL_NAME: name, COL_WATT: round(float(watt), 0) if watt else 0, COL_QTY: 1,
           COL_HOURS: hours, COL_DAYS: 30, COL_DUTY: duty}
    return row, notes

"""อัตราค่าไฟ เครื่องใช้ไฟฟ้า และข้อมูลสภาพอากาศ

ไฟล์นี้ไม่พึ่ง Streamlit เพื่อให้ทดสอบแยกได้ง่าย (ตัวแทนของ gold_service.py เดิม)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# อัตราค่าไฟบ้านอยู่อาศัยแบบก้าวหน้า (บาท/หน่วย) ยังไม่รวมค่าบริการ, Ft, VAT
# ที่มา: ประกาศ กฟน. (MEA) รอบบิลเดือนกันยายน 2569 และค่า Ft งวด ก.ย.–ธ.ค. 2569
# อัตราเปลี่ยนได้ทุกงวด ให้ตรวจกับใบแจ้งค่าไฟจริง หรือแก้ตัวเลขด้านล่างได้เลย
# ---------------------------------------------------------------------------
TARIFF_AS_OF = "รอบบิลเดือนกันยายน 2569 (กฟน./MEA)"

# (จำนวนหน่วยในช่วง, บาท/หน่วย) ช่วงสุดท้ายเป็น None = ไม่จำกัด
TIERS_1_1 = [(15, 2.3488), (10, 2.9882), (175, 3.0000), (200, 4.1584), (None, 4.3583)]  # ใช้ไม่เกิน 150 หน่วย/เดือน
TIERS_1_2 = [(200, 3.0000), (200, 4.1584), (None, 4.3583)]  # ใช้เกิน 150 หน่วย/เดือน

SERVICE_1_1 = 8.19    # ค่าบริการรายเดือน (บาท) ประเภท 1.1 ตรวจสอบกับใบแจ้งหนี้
SERVICE_1_2 = 24.62   # ค่าบริการรายเดือน (บาท) ประเภท 1.2 ตรวจสอบกับใบแจ้งหนี้
FT_DEFAULT_SATANG = 16.23  # ค่า Ft งวด ก.ย.–ธ.ค. 2569 (สตางค์/หน่วย)
VAT_RATE = 0.07


@dataclass(frozen=True)
class TariffConfig:
    """ตั้งค่าวิธีคิดค่าไฟ (frozen เพื่อให้ใช้เป็น argument ของ st.cache_data ได้)"""

    mode: str = "progressive"        # "progressive" = อัตราก้าวหน้า, "flat" = ค่าเฉลี่ยต่อหน่วยที่ผู้ใช้กำหนด
    flat_rate: float = 4.0           # ใช้เมื่อ mode == "flat" (รวมทุกอย่างแล้ว)
    ft_baht: float = FT_DEFAULT_SATANG / 100
    service_1_1: float = SERVICE_1_1
    service_1_2: float = SERVICE_1_2
    vat: float = VAT_RATE


def _energy_charge(kwh: float, tiers) -> float:
    remaining, total = kwh, 0.0
    for size, rate in tiers:
        if remaining <= 0:
            break
        used = remaining if size is None else min(remaining, size)
        total += used * rate
        remaining -= used
    return total


def bill_breakdown(kwh: float, cfg: TariffConfig | None = None) -> dict:
    """แยกส่วนประกอบค่าไฟ: ค่าพลังงาน, Ft, ค่าบริการ, VAT และยอดรวม"""
    cfg = cfg or TariffConfig()
    kwh = max(0.0, float(kwh))
    if cfg.mode == "flat":
        total = kwh * cfg.flat_rate
        return {"plan": "เฉลี่ยต่อหน่วย", "energy": total, "ft": 0.0, "service": 0.0, "vat": 0.0, "total": total}
    plan = "1.1" if kwh <= 150 else "1.2"
    tiers = TIERS_1_1 if plan == "1.1" else TIERS_1_2
    service = cfg.service_1_1 if plan == "1.1" else cfg.service_1_2
    energy = _energy_charge(kwh, tiers)
    ft = kwh * cfg.ft_baht
    subtotal = energy + ft + service
    vat = subtotal * cfg.vat
    return {"plan": plan, "energy": energy, "ft": ft, "service": service, "vat": vat, "total": subtotal + vat}


def estimate_bill(kwh: float, cfg: TariffConfig | None = None) -> float:
    """ค่าไฟโดยประมาณ (บาท) ไม่ใช่ยอดบิลทางการ"""
    return bill_breakdown(kwh, cfg)["total"]


# ---------------------------------------------------------------------------
# เครื่องใช้ไฟฟ้า
# ---------------------------------------------------------------------------
COL_NAME = "เครื่องใช้ไฟฟ้า"
COL_WATT = "กำลังไฟ (วัตต์)"
COL_QTY = "จำนวน (เครื่อง)"
COL_HOURS = "ชั่วโมง/วัน"
COL_DAYS = "วัน/เดือน"
COL_DUTY = "ทำงานจริง (%)"


def default_appliances() -> pd.DataFrame:
    """ค่าเริ่มต้นเป็นค่าประมาณทั่วไป ควรแก้ตามป้ายเครื่องของผู้ใช้"""
    rows = [
        ("แอร์ 12,000 BTU", 1000, 1, 8, 30, 70),
        ("ตู้เย็น", 150, 1, 24, 30, 40),
        ("ทีวี", 100, 1, 4, 30, 100),
        ("พัดลม", 50, 2, 6, 30, 100),
        ("หลอดไฟ LED", 10, 8, 5, 30, 100),
        ("คอมพิวเตอร์/โน้ตบุ๊ก", 60, 1, 5, 30, 100),
        ("เครื่องทำน้ำอุ่น", 3500, 1, 0.3, 30, 100),
        ("เครื่องซักผ้า", 400, 1, 1, 12, 100),
    ]
    return pd.DataFrame(rows, columns=[COL_NAME, COL_WATT, COL_QTY, COL_HOURS, COL_DAYS, COL_DUTY])


def appliance_energy(table: pd.DataFrame) -> pd.DataFrame:
    """คำนวณ kWh/เดือนของแต่ละเครื่อง = วัตต์/1000 × จำนวน × ชม./วัน × วัน × %ทำงานจริง"""
    work = table.copy()
    for col in (COL_WATT, COL_QTY, COL_HOURS, COL_DAYS, COL_DUTY):
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).clip(lower=0)
    work[COL_DAYS] = work[COL_DAYS].clip(upper=31)
    work[COL_HOURS] = work[COL_HOURS].clip(upper=24)
    work[COL_DUTY] = work[COL_DUTY].clip(upper=100)
    names = work[COL_NAME].fillna("").astype(str).str.strip()
    kwh = work[COL_WATT] / 1000 * work[COL_QTY] * work[COL_HOURS] * work[COL_DAYS] * work[COL_DUTY] / 100
    out = pd.DataFrame({"name": names, "kwh": kwh})
    out = out[(out["name"] != "") | (out["kwh"] > 0)].copy()
    out.loc[out["name"] == "", "name"] = "ไม่ระบุชื่อ"
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# สภาพอากาศ (Open-Meteo ฟรี ไม่ต้องใช้ API key)
# ---------------------------------------------------------------------------
def get_weather(latitude: float = 13.7563, longitude: float = 100.5018) -> dict:
    """อุณหภูมิ/ความชื้นปัจจุบัน คืน None เมื่อดึงไม่สำเร็จ"""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": latitude, "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m", "timezone": "auto"},
            timeout=10,
        )
        r.raise_for_status()
        cur = r.json().get("current", {})
        return {"temperature": cur.get("temperature_2m"), "humidity": cur.get("relative_humidity_2m")}
    except Exception:
        return {"temperature": None, "humidity": None}


def get_monthly_temperature(latitude: float, longitude: float, start: str, end: str) -> pd.DataFrame:
    """อุณหภูมิเฉลี่ยรายเดือนย้อนหลังจาก Open-Meteo Archive (คอลัมน์ month, temp)"""
    empty = pd.DataFrame({"month": pd.Series(dtype="datetime64[ns]"), "temp": pd.Series(dtype="float64")})
    today = pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start)
    end_ts = min(pd.Timestamp(end), today - pd.Timedelta(days=5))  # ข้อมูล archive ล่าช้าราว 2–5 วัน
    if start_ts >= end_ts:
        return empty
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": latitude, "longitude": longitude,
                    "start_date": start_ts.strftime("%Y-%m-%d"), "end_date": end_ts.strftime("%Y-%m-%d"),
                    "daily": "temperature_2m_mean", "timezone": "auto"},
            timeout=20,
        )
        r.raise_for_status()
        daily = r.json().get("daily", {})
        t = pd.DataFrame({"date": pd.to_datetime(daily.get("time", [])),
                          "temp": pd.to_numeric(pd.Series(daily.get("temperature_2m_mean", [])), errors="coerce")})
    except Exception:
        return empty
    if t.empty:
        return empty
    t["month"] = t["date"].dt.to_period("M").dt.to_timestamp()
    grouped = t.groupby("month")["temp"].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= 20]  # ตัดเดือนที่มีข้อมูลไม่ครบ
    out = grouped.rename(columns={"mean": "temp"})[["month", "temp"]]
    out["month"] = out["month"].astype("datetime64[ns]")
    return out.reset_index(drop=True)


def get_daily_forecast(latitude: float, longitude: float, days: int = 7) -> pd.DataFrame:
    """พยากรณ์อากาศรายวันจาก Open-Meteo (คอลัมน์ date, tmax, tmin, tmean, rain_mm) คืนตารางว่างเมื่อดึงไม่สำเร็จ"""
    empty = pd.DataFrame(columns=["date", "tmax", "tmin", "tmean", "rain_mm"])
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": latitude, "longitude": longitude, "forecast_days": days, "timezone": "auto",
                    "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum"},
            timeout=10,
        )
        r.raise_for_status()
        daily = r.json().get("daily", {})
        out = pd.DataFrame({
            "date": pd.to_datetime(daily.get("time", [])),
            "tmax": pd.to_numeric(pd.Series(daily.get("temperature_2m_max", [])), errors="coerce"),
            "tmin": pd.to_numeric(pd.Series(daily.get("temperature_2m_min", [])), errors="coerce"),
            "tmean": pd.to_numeric(pd.Series(daily.get("temperature_2m_mean", [])), errors="coerce"),
            "rain_mm": pd.to_numeric(pd.Series(daily.get("precipitation_sum", [])), errors="coerce").fillna(0),
        })
    except Exception:
        return empty
    out["tmean"] = out["tmean"].fillna((out["tmax"] + out["tmin"]) / 2)
    return out.dropna(subset=["tmax"]).reset_index(drop=True)

"""เครื่องมือ (Tools) ที่ AI เรียกใช้ตอนตอบคำถามในหน้า "ถาม AI"

หลักการเดียวกับ Lab 5–7: AI ห้ามเดาตัวเลขเอง ต้องเรียก Tool ที่คำนวณจากข้อมูลจริงของผู้ใช้
ไฟล์นี้ไม่พึ่ง Streamlit และไม่พึ่ง Gemini จึงทดสอบแยกได้
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from data_service import parse_month, thai_month_label
from electricity_service import (
    FT_DEFAULT_SATANG, TARIFF_AS_OF, TariffConfig, appliance_energy, bill_breakdown, estimate_bill,
)
from planner_service import payback, what_if
from stats_service import calculate_stats, forecast_next_month, temperature_effect


@dataclass
class ChatContext:
    """ข้อมูลที่ Tools มองเห็น (ส่งมาจาก app.py)"""

    data: Optional[pd.DataFrame]          # ต้องผ่าน add_bills แล้ว (มี bill_est, bill_best, is_actual)
    tariff: TariffConfig
    calib: float = 1.0                    # ตัวปรับค่าไฟตามบิลจริง
    appliances: Optional[pd.DataFrame] = None
    _forecast: object = field(default=None, repr=False)

    def has_data(self) -> bool:
        return self.data is not None and len(self.data) > 0

    def forecast(self):
        if self._forecast is None:
            self._forecast = forecast_next_month(self.data[["month", "kwh"]])
        return self._forecast


def _r(x, nd: int = 1):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(x), nd)


NO_DATA = {"error": "ผู้ใช้ยังไม่มีข้อมูลค่าไฟ ให้แนะนำไปที่หน้า “ข้อมูลค่าไฟ” เพื่ออัปโหลดไฟล์ ถ่ายรูปบิล หรือกรอกเอง"}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def get_overview(ctx: ChatContext) -> dict:
    if not ctx.has_data():
        return NO_DATA
    d = ctx.data
    s = calculate_stats(d)
    latest = d.iloc[-1]
    eff = temperature_effect(d)
    return {
        "จำนวนเดือน": s["n"],
        "ช่วงข้อมูล": f"{thai_month_label(d['month'].iloc[0])} ถึง {thai_month_label(d['month'].iloc[-1])}",
        "เดือนล่าสุด": thai_month_label(latest["month"]),
        "หน่วยไฟเดือนล่าสุด": _r(latest["kwh"]),
        "ค่าไฟเดือนล่าสุด_บาท": _r(latest["bill_best"], 2),
        "ค่าไฟเดือนล่าสุดเป็นบิลจริง": bool(latest["is_actual"]),
        "เฉลี่ยต่อเดือน_หน่วย": _r(s["mean"]),
        "เฉลี่ยต่อเดือน_บาท": _r(d["bill_best"].mean(), 0),
        "สูงสุด": {"เดือน": thai_month_label(s["max_month"]), "หน่วย": _r(s["max"])},
        "ต่ำสุด": {"เดือน": thai_month_label(s["min_month"]), "หน่วย": _r(s["min"])},
        "เปลี่ยนจากเดือนก่อน_%": _r(s["mom"]),
        "เทียบเดือนเดียวกันปีก่อน_%": _r(s["yoy"]),
        "แนวโน้ม_หน่วยต่อเดือน": _r(s["slope"], 2),
        "ความผันผวน_%": _r(s["volatility"]),
        "ความสัมพันธ์กับอุณหภูมิ": None if eff is None else
            {"r": _r(eff["r"], 2), "หน่วยที่เปลี่ยนต่อ_1C": _r(eff["slope"]), "จำนวนเดือน": eff["n"]},
        "เดือนที่ระบบเติมค่าประมาณ": [thai_month_label(m) for m in d.loc[d["filled"], "month"]],
    }


def _find_month(ctx: ChatContext, month: str):
    ts = parse_month(month)
    if pd.isna(ts):
        return None, {"error": f"อ่านเดือน “{month}” ไม่ได้ ใช้รูปแบบเช่น 2026-09 หรือ ก.ย. 2569"}
    row = ctx.data[ctx.data["month"] == ts]
    if row.empty:
        return None, {"error": f"ไม่มีข้อมูลเดือน {thai_month_label(ts)}",
                      "เดือนที่มี": f"{thai_month_label(ctx.data['month'].iloc[0])} ถึง {thai_month_label(ctx.data['month'].iloc[-1])}"}
    return row.iloc[0], None


def _month_dict(row) -> dict:
    return {
        "เดือน": thai_month_label(row["month"]),
        "หน่วยไฟ": _r(row["kwh"]),
        "ค่าไฟ_บาท": _r(row["bill_best"], 2),
        "เป็นบิลจริง": bool(row["is_actual"]),
        "ระบบเติมค่าประมาณ": bool(row["filled"]),
        "อุณหภูมิเฉลี่ย_C": _r(row["temp"]),
    }


def get_month(ctx: ChatContext, month: str) -> dict:
    if not ctx.has_data():
        return NO_DATA
    row, err = _find_month(ctx, month)
    return err or _month_dict(row)


def compare_months(ctx: ChatContext, month_a: str, month_b: str) -> dict:
    if not ctx.has_data():
        return NO_DATA
    a, err = _find_month(ctx, month_a)
    if err:
        return err
    b, err = _find_month(ctx, month_b)
    if err:
        return err
    diff_kwh = float(b["kwh"] - a["kwh"])
    return {
        "เดือนแรก": _month_dict(a), "เดือนที่สอง": _month_dict(b),
        "ต่างกัน_หน่วย": _r(diff_kwh),
        "ต่างกัน_%": _r(diff_kwh / a["kwh"] * 100) if a["kwh"] else None,
        "ต่างกัน_บาท": _r(b["bill_best"] - a["bill_best"], 2),
    }


def get_forecast(ctx: ChatContext) -> dict:
    if not ctx.has_data():
        return NO_DATA
    if len(ctx.data) < 3:
        return {"error": "ต้องมีข้อมูลอย่างน้อย 3 เดือนจึงคาดการณ์ได้"}
    fc = ctx.forecast()
    return {
        "เดือนที่คาดการณ์": thai_month_label(fc.next_month),
        "หน่วยไฟที่คาด": _r(fc.kwh),
        "ช่วงหน่วยไฟ_80%": [_r(fc.low), _r(fc.high)],
        "ค่าไฟที่คาด_บาท": _r(estimate_bill(fc.kwh, ctx.tariff) * ctx.calib, 0),
        "ช่วงค่าไฟ_บาท": [_r(estimate_bill(fc.low, ctx.tariff) * ctx.calib, 0),
                          _r(estimate_bill(fc.high, ctx.tariff) * ctx.calib, 0)],
        "วิธีที่ใช้": fc.model_name,
        "ความน่าเชื่อถือ": fc.reliability,
        "คลาดเคลื่อนเฉลี่ยตอนทดสอบย้อนหลัง_%": _r(fc.mape),
        "จำนวนเดือนที่ใช้": fc.n_points,
    }


def calculate_bill(ctx: ChatContext, kwh: float) -> dict:
    kwh = float(kwh)
    if kwh < 0 or kwh > 100_000:
        return {"error": "จำนวนหน่วยต้องอยู่ระหว่าง 0–100,000"}
    b = bill_breakdown(kwh, ctx.tariff)
    return {
        "หน่วยไฟ": _r(kwh), "ประเภทผู้ใช้": b["plan"],
        "ค่าพลังงาน": _r(b["energy"], 2), "ค่า_Ft": _r(b["ft"], 2), "ค่าบริการ": _r(b["service"], 2),
        "VAT": _r(b["vat"], 2), "รวม_บาท": _r(b["total"], 2),
        "หมายเหตุ": "ค่าประมาณจากอัตรา กฟน. ไม่ใช่ยอดบิลทางการ",
    }


def simulate_saving(ctx: ChatContext, reduce_percent: float = 0.0, reduce_kwh: float = 0.0) -> dict:
    if not ctx.has_data():
        return NO_DATA
    base = float(ctx.data["kwh"].iloc[-1])
    cut = base * float(reduce_percent) / 100 + float(reduce_kwh)
    cut = float(np.clip(cut, 0, base))
    before = estimate_bill(base, ctx.tariff) * ctx.calib
    after = estimate_bill(base - cut, ctx.tariff) * ctx.calib
    return {
        "อ้างอิงเดือน": thai_month_label(ctx.data["month"].iloc[-1]),
        "หน่วยเดิม": _r(base), "หน่วยหลังลด": _r(base - cut), "ลดลง_หน่วย": _r(cut),
        "ค่าไฟเดิม_บาท": _r(before, 0), "ค่าไฟหลังลด_บาท": _r(after, 0),
        "ประหยัดต่อเดือน_บาท": _r(before - after, 0), "ประหยัดต่อปี_บาท": _r((before - after) * 12, 0),
    }


def _rate_per_kwh(ctx: ChatContext) -> float:
    if ctx.has_data():
        last = ctx.data.iloc[-1]
        if last["kwh"] > 0:
            return float(last["bill_best"] / last["kwh"])
    return estimate_bill(300, ctx.tariff) / 300


def get_appliance_breakdown(ctx: ChatContext) -> dict:
    if ctx.appliances is None or len(ctx.appliances) == 0:
        return {"error": "ยังไม่มีตารางเครื่องใช้ไฟฟ้า ให้ผู้ใช้ไปกรอกที่หน้า “เครื่องใช้ไฟฟ้า”"}
    e = appliance_energy(ctx.appliances).sort_values("kwh", ascending=False)
    rate = _rate_per_kwh(ctx)
    total = float(e["kwh"].sum())
    return {
        "หมายเหตุ": "ค่าประมาณจากตารางที่ผู้ใช้กรอกในหน้าเครื่องใช้ไฟฟ้า",
        "รวม_หน่วยต่อเดือน": _r(total),
        "ค่าไฟเฉลี่ยต่อหน่วย_บาท": _r(rate, 2),
        "รายการ": [{"เครื่อง": r["name"], "หน่วยต่อเดือน": _r(r["kwh"]), "บาทต่อเดือน": _r(r["kwh"] * rate, 0),
                    "สัดส่วน_%": _r(r["kwh"] / total * 100) if total else None} for _, r in e.iterrows()],
    }


def appliance_cost(ctx: ChatContext, watt: float, hours_per_day: float, days_per_month: float = 30,
                   quantity: float = 1, duty_percent: float = 100) -> dict:
    watt, hours, days = float(watt), float(np.clip(hours_per_day, 0, 24)), float(np.clip(days_per_month, 0, 31))
    if watt <= 0 or watt > 50_000:
        return {"error": "กำลังไฟต้องอยู่ระหว่าง 1–50,000 วัตต์"}
    kwh = watt / 1000 * float(quantity) * hours * days * float(np.clip(duty_percent, 0, 100)) / 100
    rate = _rate_per_kwh(ctx)
    return {"หน่วยต่อเดือน": _r(kwh), "ค่าไฟเฉลี่ยต่อหน่วย_บาท": _r(rate, 2),
            "บาทต่อเดือน": _r(kwh * rate, 0), "บาทต่อปี": _r(kwh * rate * 12, 0)}


def get_tariff_info(ctx: ChatContext) -> dict:
    t = ctx.tariff
    return {
        "อ้างอิง": TARIFF_AS_OF,
        "วิธีคิด": "อัตราเฉลี่ยต่อหน่วยที่ผู้ใช้ตั้ง" if t.mode == "flat" else "อัตราก้าวหน้าบ้านอยู่อาศัย",
        "อัตราเฉลี่ย_บาทต่อหน่วย": t.flat_rate if t.mode == "flat" else None,
        "Ft_สตางค์ต่อหน่วย": _r(t.ft_baht * 100, 2), "Ft_ค่าเริ่มต้น": FT_DEFAULT_SATANG,
        "ค่าบริการ_1.1": t.service_1_1, "ค่าบริการ_1.2": t.service_1_2, "VAT_%": t.vat * 100,
        "ตัวปรับตามบิลจริง": _r(ctx.calib, 3),
    }


def simulate_what_if(ctx: ChatContext, people_now: int = 3, people_after: int = 3, wfh_days_per_week: float = 0,
                     ac_temp_up: float = 0, ac_hours_change: float = 0) -> dict:
    if not ctx.has_data():
        return NO_DATA
    base = float(ctx.data["kwh"].iloc[-1])
    r = what_if(base, ctx.appliances, people_now, people_after, wfh_days_per_week, ac_temp_up, ac_hours_change)
    before = estimate_bill(r["base_kwh"], ctx.tariff) * ctx.calib
    after = estimate_bill(r["new_kwh"], ctx.tariff) * ctx.calib
    return {"อ้างอิงเดือน": thai_month_label(ctx.data["month"].iloc[-1]),
            "รายการ": [{"เรื่อง": i["เรื่อง"], "เปลี่ยน_หน่วย": _r(i["เปลี่ยน_kwh"])} for i in r["items"]],
            "หน่วยเดิม": _r(r["base_kwh"]), "หน่วยใหม่": _r(r["new_kwh"]),
            "ค่าไฟเดิม_บาท": _r(before, 0), "ค่าไฟใหม่_บาท": _r(after, 0), "ต่าง_บาทต่อเดือน": _r(after - before, 0),
            "หมายเหตุ": "ประมาณจากสมมติฐานในหน้า “จำลองและความคุ้มค่า”"}


def check_payback(ctx: ChatContext, old_watt: float, new_watt: float, hours_per_day: float, price: float,
                  old_duty: float = 100, new_duty: float = 100) -> dict:
    r = payback(old_watt, new_watt, hours_per_day, price, _rate_per_kwh(ctx), old_duty=old_duty, new_duty=new_duty)
    return {"ประหยัด_หน่วยต่อเดือน": _r(r["save_kwh"]), "ประหยัด_บาทต่อเดือน": _r(r["save_baht_month"], 0),
            "ประหยัด_บาทต่อปี": _r(r["save_baht_year"], 0),
            "คืนทุน_ปี": _r(r["payback_years"], 1) if r["payback_years"] is not None else "ไม่คืนทุน (เครื่องใหม่ไม่ได้ประหยัดกว่า)",
            "กำไรสุทธิใน_10_ปี_บาท": _r(r["save_10y"], 0), "ค่าไฟเฉลี่ยต่อหน่วย_บาท": _r(_rate_per_kwh(ctx), 2)}


# ---------------------------------------------------------------------------
# คำอธิบาย Tools สำหรับ Gemini (function declarations)
# ---------------------------------------------------------------------------
_MONTH = {"type": "STRING", "description": "เดือน เช่น 2026-09 หรือ ก.ย. 2569"}

TOOL_DECLARATIONS = [
    {"name": "get_overview", "description": "ภาพรวมข้อมูลค่าไฟของผู้ใช้: เดือนล่าสุด ค่าเฉลี่ย สูงสุด ต่ำสุด แนวโน้ม เทียบเดือนก่อน/ปีก่อน",
     "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "get_month", "description": "หน่วยไฟและค่าไฟของเดือนที่ระบุ",
     "parameters": {"type": "OBJECT", "properties": {"month": _MONTH}, "required": ["month"]}},
    {"name": "compare_months", "description": "เปรียบเทียบหน่วยไฟและค่าไฟของสองเดือน",
     "parameters": {"type": "OBJECT", "properties": {"month_a": _MONTH, "month_b": _MONTH}, "required": ["month_a", "month_b"]}},
    {"name": "get_forecast", "description": "คาดการณ์หน่วยไฟและค่าไฟเดือนถัดไป พร้อมช่วงความไม่แน่นอนและความน่าเชื่อถือ",
     "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "calculate_bill", "description": "คำนวณค่าไฟจากจำนวนหน่วย แยกค่าพลังงาน Ft ค่าบริการ VAT",
     "parameters": {"type": "OBJECT", "properties": {"kwh": {"type": "NUMBER", "description": "จำนวนหน่วย (kWh)"}}, "required": ["kwh"]}},
    {"name": "simulate_saving", "description": "จำลองว่าถ้าลดการใช้ไฟจากเดือนล่าสุด (เป็น % หรือเป็นหน่วย) จะประหยัดได้กี่บาท",
     "parameters": {"type": "OBJECT", "properties": {
         "reduce_percent": {"type": "NUMBER", "description": "ลดกี่เปอร์เซ็นต์"},
         "reduce_kwh": {"type": "NUMBER", "description": "ลดกี่หน่วย"}}}},
    {"name": "get_appliance_breakdown", "description": "เครื่องใช้ไฟฟ้าแต่ละเครื่องในบ้านกินไฟกี่หน่วยและกี่บาทต่อเดือน (จากตารางที่ผู้ใช้กรอก)",
     "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "appliance_cost", "description": "คำนวณค่าไฟของเครื่องใช้ไฟฟ้าหนึ่งเครื่องจากกำลังไฟและเวลาที่ใช้",
     "parameters": {"type": "OBJECT", "properties": {
         "watt": {"type": "NUMBER", "description": "กำลังไฟ (วัตต์)"},
         "hours_per_day": {"type": "NUMBER", "description": "ชั่วโมงต่อวัน"},
         "days_per_month": {"type": "NUMBER", "description": "วันต่อเดือน (ค่าเริ่มต้น 30)"},
         "quantity": {"type": "NUMBER", "description": "จำนวนเครื่อง (ค่าเริ่มต้น 1)"},
         "duty_percent": {"type": "NUMBER", "description": "% เวลาที่คอมเพรสเซอร์ทำงานจริง แอร์/ตู้เย็นราว 40–70 (ค่าเริ่มต้น 100)"}},
         "required": ["watt", "hours_per_day"]}},
    {"name": "simulate_what_if", "description": "จำลองว่าค่าไฟจะเปลี่ยนเท่าไร ถ้าจำนวนคนในบ้านเปลี่ยน ทำงานที่บ้านกี่วัน ปรับอุณหภูมิแอร์ หรือเปลี่ยนชั่วโมงเปิดแอร์",
     "parameters": {"type": "OBJECT", "properties": {
         "people_now": {"type": "INTEGER", "description": "จำนวนคนตอนนี้"},
         "people_after": {"type": "INTEGER", "description": "จำนวนคนหลังเปลี่ยน"},
         "wfh_days_per_week": {"type": "NUMBER", "description": "ทำงานที่บ้านเพิ่มกี่วันต่อสัปดาห์"},
         "ac_temp_up": {"type": "NUMBER", "description": "ตั้งแอร์สูงขึ้นกี่องศา (ติดลบ = ต่ำลง)"},
         "ac_hours_change": {"type": "NUMBER", "description": "เปิดแอร์เพิ่ม/ลดกี่ชั่วโมงต่อวัน (ติดลบ = ลด)"}}}},
    {"name": "check_payback", "description": "คำนวณว่าซื้อเครื่องใช้ไฟฟ้าใหม่ที่ประหยัดไฟกว่า ประหยัดเดือนละเท่าไร และกี่ปีคืนทุน",
     "parameters": {"type": "OBJECT", "properties": {
         "old_watt": {"type": "NUMBER", "description": "กำลังไฟเครื่องเดิม (วัตต์)"},
         "new_watt": {"type": "NUMBER", "description": "กำลังไฟเครื่องใหม่ (วัตต์)"},
         "hours_per_day": {"type": "NUMBER", "description": "ใช้กี่ชั่วโมงต่อวัน"},
         "price": {"type": "NUMBER", "description": "ราคาเครื่องใหม่ (บาท)"},
         "old_duty": {"type": "NUMBER", "description": "% ทำงานจริงของเครื่องเดิม (แอร์ธรรมดาราว 80–90)"},
         "new_duty": {"type": "NUMBER", "description": "% ทำงานจริงของเครื่องใหม่ (แอร์อินเวอร์เตอร์ราว 50–60)"}},
         "required": ["old_watt", "new_watt", "hours_per_day", "price"]}},
    {"name": "get_tariff_info", "description": "อัตราค่าไฟ ค่า Ft ค่าบริการ และ VAT ที่ระบบใช้คำนวณ",
     "parameters": {"type": "OBJECT", "properties": {}}},
]

_FUNCS = {
    "get_overview": get_overview, "get_month": get_month, "compare_months": compare_months,
    "get_forecast": get_forecast, "calculate_bill": calculate_bill, "simulate_saving": simulate_saving,
    "get_appliance_breakdown": get_appliance_breakdown, "appliance_cost": appliance_cost,
    "get_tariff_info": get_tariff_info, "simulate_what_if": simulate_what_if, "check_payback": check_payback,
}


def execute_tool(ctx: ChatContext, name: str, args: dict) -> dict:
    fn = _FUNCS.get(name)
    if fn is None:
        return {"error": f"ไม่รู้จัก Tool: {name}"}
    try:
        return fn(ctx, **(args or {}))
    except TypeError as err:
        return {"error": f"argument ไม่ถูกต้อง: {err}"}
    except Exception as err:  # ไม่ให้แชทพังเพราะ Tool ตัวเดียว
        return {"error": str(err)}

"""ตรวจความสมเหตุสมผลของข้อมูลที่ AI อ่านจากบิล และรวมเข้ากับข้อมูลเดิม

ไม่พึ่ง Streamlit/Gemini จึงทดสอบแยกได้
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_service import clean_monthly, parse_month, thai_month_label
from electricity_service import TariffConfig, estimate_bill


def bill_month(info: dict):
    """หาเดือนของบิล: billing_month_iso → billing_month → reading_date"""
    for key in ("billing_month_iso", "billing_month", "reading_date"):
        value = info.get(key)
        if value:
            ts = parse_month(value)
            if not pd.isna(ts):
                return ts
    return pd.NaT


def check_bill(info: dict, tariff: TariffConfig) -> dict:
    """คืน {month, kwh, bill, warnings, ok}

    - ถ้าไม่มีจำนวนหน่วย แต่มีเลขอ่านมิเตอร์ ใช้ผลต่างแทน
    - เตือนถ้าหน่วยไม่ตรงกับเลขมิเตอร์ หรือยอดเงินต่างจากที่คำนวณได้มาก
    """
    warnings: list[str] = []
    if info.get("is_electricity_bill") is False:
        return {"month": pd.NaT, "kwh": None, "bill": None, "due": pd.NaT, "ok": False,
                "warnings": ["ไฟล์นี้ดูไม่ใช่บิลค่าไฟ ลองถ่ายใหม่ให้เห็นตัวบิลชัดๆ"]}

    month = bill_month(info)
    kwh = info.get("kwh")
    prev, cur = info.get("meter_previous"), info.get("meter_current")
    if prev is not None and cur is not None and cur >= prev:
        diff = cur - prev
        if kwh is None:
            kwh = diff
            warnings.append(f"ไม่พบจำนวนหน่วยบนบิล จึงใช้ผลต่างเลขมิเตอร์ ({cur:,.0f} − {prev:,.0f} = {diff:,.0f} หน่วย)")
        elif abs(diff - kwh) > max(2, 0.02 * kwh):
            warnings.append(f"จำนวนหน่วย ({kwh:,.0f}) ไม่ตรงกับผลต่างเลขมิเตอร์ ({diff:,.0f}) อาจมีตัวคูณมิเตอร์หรืออ่านผิด ตรวจสอบอีกครั้ง")

    bill = info.get("total_amount")
    if pd.isna(month):
        warnings.append("อ่านเดือนของบิลไม่ได้ กรุณาเลือกเดือนเอง")
    if kwh is None:
        warnings.append("อ่านจำนวนหน่วยไม่ได้ กรุณากรอกเอง")
    elif kwh <= 0 or kwh > 20_000:
        warnings.append(f"จำนวนหน่วย {kwh:,.0f} ดูผิดปกติสำหรับบ้านอยู่อาศัย ตรวจสอบอีกครั้ง")

    if kwh and bill:
        est = estimate_bill(kwh, tariff)
        if est > 0 and abs(bill - est) / est > 0.2:
            warnings.append(f"ยอดเงินบนบิล ฿{bill:,.2f} ต่างจากที่คำนวณจาก {kwh:,.0f} หน่วย (≈ ฿{est:,.0f}) เกิน 20% "
                            "อาจอ่านยอดรวมค้างชำระ หรือเป็นบิลพื้นที่ กฟภ./อัตราอื่น ตรวจสอบอีกครั้ง")
    if info.get("notes"):
        warnings.append(f"AI หมายเหตุ: {info['notes']}")

    due = pd.to_datetime(info.get("due_date"), errors="coerce") if info.get("due_date") else pd.NaT
    return {"month": month, "kwh": kwh, "bill": bill, "due": due, "warnings": warnings,
            "ok": not pd.isna(month) and kwh is not None and kwh > 0}


def upsert_months(existing: pd.DataFrame | None, rows: pd.DataFrame):
    """เพิ่ม/แทนที่เดือนจากบิลเข้ากับข้อมูลเดิม แล้วทำความสะอาดใหม่

    rows: คอลัมน์ month (Timestamp), kwh, bill  — เดือนที่ซ้ำกับของเดิมจะใช้ค่าจากบิลแทน
    คืน (ตารางสะอาด, ข้อความเตือน, จำนวนเดือนที่แทนที่)
    """
    new = rows[["month", "kwh", "bill"]].copy()
    new["temp"] = np.nan
    new["month"] = pd.to_datetime(new["month"]).dt.to_period("M").dt.to_timestamp()
    replaced = 0
    if existing is not None and len(existing):
        old = existing.loc[~existing["filled"], ["month", "kwh", "bill", "temp"]].copy()  # ไม่เอาเดือนที่ระบบเติมเอง
        replaced = int(old["month"].isin(new["month"]).sum())
        temp_map = old.set_index("month")["temp"]
        new["temp"] = new["month"].map(temp_map)
        combined = pd.concat([old[~old["month"].isin(new["month"])], new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates("month", keep="last").sort_values("month")
    clean, issues = clean_monthly(combined, "month", "kwh", "bill", "temp")
    return clean, issues, replaced


def describe_month(ts) -> str:
    return "—" if pd.isna(ts) else thai_month_label(ts)


# ---------------------------------------------------------------------------
# ปฏิทินบิล: สถานะการชำระ + ไฟล์ .ics สำหรับเพิ่มเตือนในปฏิทินมือถือ
# ---------------------------------------------------------------------------
CAL_MONTH, CAL_AMOUNT, CAL_DUE, CAL_PAID = "เดือนบิล", "ยอดเงิน (บาท)", "ครบกำหนดชำระ", "จ่ายแล้ว"


def empty_calendar() -> pd.DataFrame:
    return pd.DataFrame({CAL_MONTH: pd.Series(dtype="str"), CAL_AMOUNT: pd.Series(dtype="float64"),
                         CAL_DUE: pd.Series(dtype="datetime64[ns]"), CAL_PAID: pd.Series(dtype="bool")})


def add_to_calendar(cal: pd.DataFrame, month_label: str, amount, due) -> pd.DataFrame:
    """เพิ่มหรือแทนที่บิลของเดือนเดียวกัน"""
    due_ts = pd.to_datetime(due, errors="coerce") if due else pd.NaT
    row = pd.DataFrame({CAL_MONTH: [month_label], CAL_AMOUNT: [amount], CAL_DUE: [due_ts], CAL_PAID: [False]})
    keep = cal[cal[CAL_MONTH] != month_label] if len(cal) else cal
    paid_before = cal.loc[cal[CAL_MONTH] == month_label, CAL_PAID] if len(cal) else pd.Series(dtype=bool)
    if len(paid_before):
        row[CAL_PAID] = bool(paid_before.iloc[0])
    out = pd.concat([keep, row], ignore_index=True) if len(keep) else row
    out[CAL_DUE] = pd.to_datetime(out[CAL_DUE], errors="coerce")
    return out.sort_values(CAL_DUE, na_position="last").reset_index(drop=True)


def calendar_status(cal: pd.DataFrame, today=None) -> pd.DataFrame:
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    out = cal.copy()
    due = pd.to_datetime(out[CAL_DUE], errors="coerce")
    days = (due - today).dt.days
    paid = out[CAL_PAID].fillna(False).astype(bool)
    out["เหลือ (วัน)"] = days
    out["สถานะ"] = np.select(
        [paid, due.isna(), days < 0, days <= 3],
        ["✅ จ่ายแล้ว", "— ไม่ทราบวันครบกำหนด", "🔴 เลยกำหนด", "🟠 ใกล้ครบกำหนด"], default="🟢 ยังไม่ถึงกำหนด")
    return out


def _ics_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def to_ics(cal: pd.DataFrame, remind_days: int = 2) -> bytes:
    """สร้างไฟล์ปฏิทิน .ics (เปิดได้ใน Google Calendar / iPhone) เฉพาะบิลที่ยังไม่จ่ายและมีวันครบกำหนด"""
    stamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Home Electricity AI//TH", "CALSCALE:GREGORIAN"]
    for i, r in cal.iterrows():
        due = pd.to_datetime(r[CAL_DUE], errors="coerce")
        if pd.isna(due) or bool(r[CAL_PAID]):
            continue
        amount = "" if pd.isna(r[CAL_AMOUNT]) else f" ฿{float(r[CAL_AMOUNT]):,.2f}"
        lines += [
            "BEGIN:VEVENT", f"UID:elec-bill-{due:%Y%m%d}-{i}@home-electricity-ai", f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{due:%Y%m%d}", f"DTEND;VALUE=DATE:{(due + pd.Timedelta(days=1)):%Y%m%d}",
            f"SUMMARY:{_ics_escape('ครบกำหนดจ่ายค่าไฟ ' + str(r[CAL_MONTH]) + amount)}",
            "BEGIN:VALARM", "ACTION:DISPLAY", f"TRIGGER:-P{int(remind_days)}D",
            f"DESCRIPTION:{_ics_escape('อีก ' + str(remind_days) + ' วันครบกำหนดจ่ายค่าไฟ')}", "END:VALARM", "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")

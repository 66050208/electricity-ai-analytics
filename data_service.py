"""นำเข้าและทำความสะอาดข้อมูลค่าไฟรายเดือนจริง (CSV / Excel / กรอกเอง)

คอลัมน์มาตรฐานหลังทำความสะอาด: month (วันที่ 1 ของเดือน), kwh, bill, temp, filled
ไม่พึ่ง Streamlit เพื่อให้ทดสอบแยกได้
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

THAI_MONTH_ABBR = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
THAI_MONTH_FULL = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
                   "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
_EN_FULL = ["january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december"]


def _build_name_lookup() -> list[tuple[str, int]]:
    pairs: dict[str, int] = {}
    for i in range(12):
        for name in (THAI_MONTH_FULL[i], THAI_MONTH_ABBR[i], _EN_FULL[i], _EN_FULL[i][:3]):
            pairs[name.lower().replace(".", "")] = i + 1
    return sorted(pairs.items(), key=lambda kv: len(kv[0]), reverse=True)  # ชื่อยาวก่อน กัน "มีนาคม" ชน "มี"


_NAME_LOOKUP = _build_name_lookup()


def thai_month_label(ts) -> str:
    """เช่น ก.ย. 2569 (พ.ศ.)"""
    ts = pd.Timestamp(ts)
    return f"{THAI_MONTH_ABBR[ts.month - 1]} {ts.year + 543}"


def _fix_year(y: int):
    if y < 100:  # ปีสองหลัก: "26" = 2026, แต่ "68" อ่านเป็น พ.ศ. 2568 (ค.ศ. 2068 ยังไม่ถึง)
        y = 2000 + y if 2000 + y <= pd.Timestamp.today().year + 1 else 2500 + y
    if y > 2400:  # พ.ศ. → ค.ศ.
        y -= 543
    return y if 1990 <= y <= 2100 else None


def _make_month(year, month):
    if year is None or not 1 <= month <= 12:
        return pd.NaT
    return pd.Timestamp(year, month, 1)


def parse_month(value):
    """อ่านเดือนจากรูปแบบที่พบบ่อย: 2026-09, 09/2026, 1/9/2569, ก.ย. 2569, September 2026, วันที่ใน Excel"""
    try:
        if pd.isna(value):
            return pd.NaT
    except (TypeError, ValueError):
        pass
    if not isinstance(value, str) and hasattr(value, "year") and hasattr(value, "month"):
        return _make_month(_fix_year(int(value.year)), int(value.month))
    s = str(value).strip()
    if not s:
        return pd.NaT
    norm = s.lower().replace(".", "")
    for name, num in _NAME_LOOKUP:
        if name in norm:
            m = re.search(r"\d{2,4}", norm.replace(name, " "))
            return _make_month(_fix_year(int(m.group())), num) if m else pd.NaT
    m = re.match(r"^(\d{4})[-/.](\d{1,2})(?:[-/.]\d{1,2})?$", s)  # 2026-09 หรือ 2026-09-15
    if m:
        return _make_month(_fix_year(int(m[1])), int(m[2]))
    m = re.match(r"^(\d{1,2})[-/.](\d{4})$", s)  # 09/2026
    if m:
        return _make_month(_fix_year(int(m[2])), int(m[1]))
    m = re.match(r"^\d{1,2}[-/.](\d{1,2})[-/.](\d{4})$", s)  # 15/09/2026 (วัน/เดือน/ปี)
    if m:
        return _make_month(_fix_year(int(m[2])), int(m[1]))
    return pd.NaT


# ---------------------------------------------------------------------------
# อ่านไฟล์ + ตรวจจับคอลัมน์
# ---------------------------------------------------------------------------
ALIASES = {
    "month": ["month", "date", "period", "เดือน", "วันที่", "รอบบิล", "เดือนปี"],
    "kwh": ["kwh", "unit", "units", "usage", "consumption", "energy", "หน่วย", "หน่วยไฟ", "จำนวนหน่วย", "การใช้ไฟ", "ใช้ไฟ"],
    "bill": ["bill", "baht", "amount", "cost", "ค่าไฟ", "ค่าไฟฟ้า", "บาท", "ยอดเงิน", "จำนวนเงิน", "ยอดชำระ"],
    "temp": ["temp", "temperature", "อุณหภูมิ", "อุณหภูมิเฉลี่ย"],
}


def _norm(s) -> str:
    return re.sub(r"[\s_\-()/\[\]:.]", "", str(s).lower())


def detect_columns(df: pd.DataFrame) -> dict:
    """เดาว่าคอลัมน์ไหนคือ เดือน / หน่วยไฟ / ค่าไฟ / อุณหภูมิ (ผู้ใช้แก้เองได้ในหน้าแอป)"""
    cols = list(df.columns)
    used: set = set()
    result: dict = {}
    for key in ("month", "kwh", "bill", "temp"):
        aliases = [_norm(a) for a in ALIASES[key]]
        found = next((c for c in cols if c not in used and _norm(c) in aliases), None)
        if found is None:
            found = next((c for c in cols if c not in used and any(a in _norm(c) for a in aliases)), None)
        result[key] = found
        if found is not None:
            used.add(found)
    return result


def read_table(uploaded) -> pd.DataFrame:
    """อ่าน CSV (UTF-8 / Windows-874) หรือ Excel (.xlsx) จากไฟล์ที่อัปโหลด"""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(io.BytesIO(data))
    for enc in ("utf-8-sig", "utf-8", "cp874"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
    raise ValueError("อ่านตัวอักษรในไฟล์ไม่ได้ ลองบันทึกเป็น CSV (UTF-8) หรือ Excel (.xlsx)")


def _to_number(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(",", "", regex=False).str.replace("฿", "", regex=False).str.strip()
    return pd.to_numeric(text, errors="coerce")


# ---------------------------------------------------------------------------
# ทำความสะอาด
# ---------------------------------------------------------------------------
def clean_monthly(raw: pd.DataFrame, month_col, kwh_col, bill_col=None, temp_col=None):
    """คืน (ตารางสะอาด, รายการข้อความเตือน) หรือ raise ValueError เมื่อใช้ข้อมูลไม่ได้เลย"""
    if month_col is None or kwh_col is None:
        raise ValueError("ต้องเลือกคอลัมน์ “เดือน” และ “หน่วยไฟ (kWh)” อย่างน้อย")
    issues: list[str] = []

    work = pd.DataFrame({
        "month": raw[month_col].map(parse_month),
        "kwh": _to_number(raw[kwh_col]),
        "bill": _to_number(raw[bill_col]) if bill_col is not None else np.nan,
        "temp": _to_number(raw[temp_col]) if temp_col is not None else np.nan,
    })
    kwh_blank = raw[kwh_col].isna() | raw[kwh_col].astype(str).str.strip().eq("")
    work = work[~kwh_blank].copy()  # แถวที่ยังไม่ได้กรอกหน่วยไฟ ข้ามโดยไม่เตือน
    month_raw = raw.loc[work.index, month_col]

    invalid = work["month"].isna() | work["kwh"].isna() | (work["kwh"] < 0)
    if invalid.any():
        examples = ", ".join(f"“{v}”" for v in month_raw[invalid].astype(str).head(3))
        issues.append(f"ข้าม {int(invalid.sum())} แถวที่อ่านเดือนหรือจำนวนหน่วยไม่ได้ (เช่น {examples})")
        work = work[~invalid]
    if work.empty:
        raise ValueError("ไม่พบแถวข้อมูลที่ใช้ได้ ตรวจสอบรูปแบบเดือนและตัวเลขหน่วยไฟ")

    dup = int(work["month"].duplicated().sum())
    if dup:
        issues.append(f"พบเดือนซ้ำ {dup} แถว จึงรวมหน่วยไฟและค่าไฟของเดือนเดียวกันเข้าด้วยกัน (เช่น มิเตอร์หลายเครื่อง)")
    work = work.groupby("month", as_index=False).agg(
        kwh=("kwh", "sum"),
        bill=("bill", lambda s: s.sum(min_count=1)),
        temp=("temp", "mean"),
    )

    full = pd.date_range(work["month"].min(), work["month"].max(), freq="MS")
    work = work.set_index("month").reindex(full)
    work.index.name = "month"
    filled = work["kwh"].isna()
    if filled.any():
        labels = ", ".join(thai_month_label(m) for m in work.index[filled][:4])
        more = "…" if filled.sum() > 4 else ""
        issues.append(f"ขาดข้อมูล {int(filled.sum())} เดือน ({labels}{more}) ระบบเติมค่าประมาณจากเดือนข้างเคียงเพื่อให้คาดการณ์ได้")
        work["kwh"] = work["kwh"].interpolate(limit_area="inside")
    work["filled"] = filled
    out = work.reset_index()
    out["month"] = out["month"].astype("datetime64[ns]")

    real = out.loc[~out["filled"], "kwh"]
    if len(real) >= 8:
        med = float(np.median(real))
        mad = float(np.median(np.abs(real - med)))
        if mad > 0:
            z = 0.6745 * (out["kwh"] - med) / mad
            odd = out[(z.abs() > 3.5) & ~out["filled"]]
            for _, row in odd.head(3).iterrows():
                issues.append(f"{thai_month_label(row['month'])} ใช้ {row['kwh']:,.0f} หน่วย สูง/ต่ำกว่าปกติมาก ตรวจสอบว่าพิมพ์ถูกต้องหรือไม่")
    return out[["month", "kwh", "bill", "temp", "filled"]], issues


# ---------------------------------------------------------------------------
# ข้อมูลตัวอย่างและเทมเพลต
# ---------------------------------------------------------------------------
def _last_full_month() -> pd.Timestamp:
    return pd.Timestamp.today().to_period("M").to_timestamp() - pd.DateOffset(months=1)


def demo_data(months: int = 18) -> pd.DataFrame:
    """ข้อมูลสมมติ (ไม่ใช่ข้อมูลจริง) ที่ร้อนจัด มี.ค.–พ.ค. ใช้ไฟสูง ใช้ลองระบบเท่านั้น"""
    seasonal = {1: 0.92, 2: 0.95, 3: 1.08, 4: 1.22, 5: 1.25, 6: 1.15, 7: 1.08, 8: 1.05, 9: 1.02, 10: 1.0, 11: 0.93, 12: 0.90}
    offsets = [3, -5, 8, -2, 6, -7, 4, 0, -3, 5, -6, 2, 7, -4, 1, -8, 5, -1, 2, -3, 4, -2, 1, 0]
    end = _last_full_month()
    idx = pd.date_range(end=end, periods=months, freq="MS")
    kwh = [round(340 * seasonal[m.month] * (1 + 0.004 * i) + offsets[i % len(offsets)]) for i, m in enumerate(idx)]
    return pd.DataFrame({"month": idx.astype("datetime64[ns]"), "kwh": np.array(kwh, dtype=float),
                         "bill": np.nan, "temp": np.nan, "filled": False})


def template_csv() -> bytes:
    """เทมเพลต CSV ให้เปิดใน Excel แล้วกรอก (มี BOM เพื่อให้ภาษาไทยไม่เพี้ยน)"""
    idx = pd.date_range(end=_last_full_month(), periods=12, freq="MS")
    lines = ["เดือน,หน่วยไฟ (kWh),ค่าไฟ (บาท)"] + [f"{m:%Y-%m},," for m in idx]
    return ("\ufeff" + "\n".join(lines) + "\n").encode("utf-8")


def export_csv(df: pd.DataFrame) -> bytes:
    """ส่งออกข้อมูลที่ทำความสะอาดแล้ว ในรูปแบบที่อัปโหลดกลับเข้ามาได้"""
    out = pd.DataFrame({"เดือน": df["month"].dt.strftime("%Y-%m"), "หน่วยไฟ (kWh)": df["kwh"].round(2),
                        "ค่าไฟ (บาท)": df["bill"].round(2)})
    if df["temp"].notna().any():
        out["อุณหภูมิเฉลี่ย (°C)"] = df["temp"].round(2)
    return ("\ufeff" + out.to_csv(index=False)).encode("utf-8")

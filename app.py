from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_service import SAVING_TIPS, generate_insights
from bill_service import (
    CAL_AMOUNT, CAL_DUE, CAL_MONTH, CAL_PAID, add_to_calendar, calendar_status, check_bill, empty_calendar, to_ics,
    upsert_months,
)
from chat_tools import ChatContext
from data_service import (
    clean_monthly, demo_data, detect_columns, export_csv, parse_month, read_table, template_csv, thai_month_label,
)
from electricity_service import (
    COL_DAYS, COL_DUTY, COL_HOURS, COL_NAME, COL_QTY, COL_WATT, FT_DEFAULT_SATANG, SERVICE_1_1, SERVICE_1_2,
    TARIFF_AS_OF, TIERS_1_1, TIERS_1_2, TariffConfig, appliance_energy, bill_breakdown, default_appliances,
    estimate_bill, get_daily_forecast, get_monthly_temperature, get_weather,
)
from insights_service import cluster_months, describe_importance, explain_forecast, household_profile
from planner_service import ASSUMPTIONS, ac_profile, label_to_row, payback, weather_outlook, what_if
from llm_service import (
    BILL_TYPES, friendly_error, has_api_key as has_gemini_key, mime_for, read_appliance_label, read_bill, run_chat,
)
from firebase_auth import (
    guest_allowed, is_logged_in, login_as_guest, login_form, logout, reset_form, signup_form,
)
from stats_service import calculate_stats, forecast_next_month, temperature_effect
from theme import COLORS, PLOT_CONFIG, apply_theme, reliability_chip, stretch, style_fig

st.set_page_config(page_title="Home Electricity AI", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
apply_theme()

PAGE_DATA, PAGE_OVERVIEW, PAGE_FORECAST, PAGE_INSIGHT, PAGE_PLANNER, PAGE_CHAT, PAGE_APPLIANCE, PAGE_CALENDAR, PAGE_HELP = (
    "ข้อมูลค่าไฟ", "ภาพรวม", "คาดการณ์และคำแนะนำ", "วิเคราะห์เชิงลึก", "จำลองและความคุ้มค่า", "ถาม AI",
    "เครื่องใช้ไฟฟ้า", "ปฏิทินบิล", "วิธีทำงาน")
PAGES = [PAGE_DATA, PAGE_OVERVIEW, PAGE_FORECAST, PAGE_INSIGHT, PAGE_PLANNER, PAGE_CHAT, PAGE_APPLIANCE, PAGE_CALENDAR, PAGE_HELP]
COL_B_MONTH, COL_B_KWH, COL_B_BILL, COL_B_DUE, COL_B_FILE = "เดือน", "หน่วยไฟ (kWh)", "ค่าไฟ (บาท)", "ครบกำหนดชำระ", "ไฟล์"
CLUSTER_COLORS = {"ใช้ไฟต่ำ": COLORS["green"], "ใช้ไฟปกติ": COLORS["blue"], "ใช้ไฟสูง": COLORS["red"]}
SUGGESTED_QUESTIONS = [
    "เดือนล่าสุดใช้ไฟเท่าไร สูงกว่าปกติไหม",
    "เดือนหน้าค่าไฟน่าจะประมาณเท่าไร",
    "ถ้าลดการใช้ไฟ 15% จะประหยัดได้ปีละกี่บาท",
    "เครื่องใช้ไฟฟ้าตัวไหนกินไฟมากที่สุด ควรเริ่มลดตรงไหน",
    "แอร์ 1,200 วัตต์ เปิดวันละ 8 ชั่วโมง เสียค่าไฟเดือนละเท่าไร",
    "ถ้าตั้งแอร์สูงขึ้น 2 องศา และทำงานที่บ้าน 2 วันต่อสัปดาห์ ค่าไฟจะเป็นเท่าไร",
    "เปลี่ยนแอร์ 1,200 วัตต์เป็นอินเวอร์เตอร์ 900 วัตต์ ราคา 18,000 บาท กี่ปีคืนทุน",
]
TARIFF_PROGRESSIVE = "อัตราก้าวหน้าบ้านอยู่อาศัย"
TARIFF_FLAT = "ค่าเฉลี่ยต่อหน่วย (กำหนดเอง)"
SOURCE_SAMPLE = "ข้อมูลตัวอย่าง"
NONE_OPTION = "— ไม่มี —"


# ---------------------------------------------------------------------------
# แคช: ไม่ให้เรียก API / เทรนโมเดลซ้ำทุกครั้งที่กดปุ่ม
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def cached_weather(lat: float, lon: float) -> dict:
    return get_weather(lat, lon)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_monthly_temp(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    return get_monthly_temperature(lat, lon, start, end)


@st.cache_data(show_spinner="กำลังคำนวณคาดการณ์...")
def cached_forecast(df: pd.DataFrame):
    return forecast_next_month(df)


# ---------------------------------------------------------------------------
# ตัวช่วย
# ---------------------------------------------------------------------------
def go_to(page: str) -> None:
    st.session_state.page = page


def baht(x: float) -> str:
    return f"฿{x:,.0f}"


def current_df():
    df = st.session_state.get("data")
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    temp_df = st.session_state.get("temp_df")
    if temp_df is not None and df["temp"].isna().all():
        df = df.drop(columns="temp").merge(temp_df[["month", "temp"]], on="month", how="left")
    return df


def add_bills(df: pd.DataFrame, tariff: TariffConfig) -> pd.DataFrame:
    out = df.copy()
    out["bill_est"] = out["kwh"].map(lambda k: estimate_bill(k, tariff))
    out["is_actual"] = out["bill"].notna()
    out["bill_best"] = out["bill"].fillna(out["bill_est"])
    return out


def calibration(d: pd.DataFrame) -> float:
    """ถ้าผู้ใช้ใส่ค่าไฟจริงอย่างน้อย 3 เดือน ใช้สัดส่วนบิลจริง/บิลที่คำนวณ ปรับค่าที่ประมาณให้ใกล้ของจริง"""
    ok = d[d["is_actual"] & (d["bill_est"] > 0)]
    if len(ok) < 3:
        return 1.0
    return float(np.clip((ok["bill"] / ok["bill_est"]).median(), 0.8, 1.25))


def require_data():
    df = current_df()
    if df is None:
        st.info("ยังไม่มีข้อมูลค่าไฟ เริ่มจากอัปโหลดไฟล์ กรอกเอง หรือลองใช้ข้อมูลตัวอย่าง")
        st.button("ไปที่หน้า “ข้อมูลค่าไฟ”", on_click=go_to, args=(PAGE_DATA,), type="primary")
        st.stop()
    if st.session_state.get("data_source") == SOURCE_SAMPLE:
        st.warning("ตอนนี้ใช้ข้อมูลตัวอย่าง ตัวเลขไม่ใช่ข้อมูลของคุณ")
    return df


def chat_context(tariff: TariffConfig) -> ChatContext:
    df = current_df()
    data = add_bills(df, tariff) if df is not None else None
    return ChatContext(
        data=data, tariff=tariff, calib=calibration(data) if data is not None else 1.0,
        appliances=st.session_state.get("appliance_table", default_appliances()),
    )


def ask_question(text: str) -> None:
    st.session_state.pending_question = text


def clear_chat() -> None:
    st.session_state.chat = []


def gemini_missing_notice() -> None:
    st.info("ฟีเจอร์นี้ใช้ Gemini ต้องตั้งค่า `GEMINI_API_KEY` ใน `.env` (รันในเครื่อง) หรือ Streamlit Secrets (บน Cloud) "
            "ขอ key ฟรีได้ที่ Google AI Studio ดูขั้นตอนใน README")


def read_bills_with_ai(files: list, tariff: TariffConfig) -> None:
    """ส่งบิลแต่ละไฟล์ให้ AI อ่าน (จำผลไว้ ไม่อ่านไฟล์เดิมซ้ำ) แล้วเตรียมตารางให้ผู้ใช้ตรวจ"""
    cache = st.session_state.setdefault("bill_cache", {})
    rows, notes = [], []
    progress = st.progress(0.0, text="AI กำลังอ่านบิล...")
    for i, (name, data, mime) in enumerate(files):
        digest = hashlib.sha1(data).hexdigest()
        if digest not in cache:
            try:
                cache[digest] = read_bill(data, mime)
            except Exception as err:
                notes.append((name, [friendly_error(err) if not isinstance(err, ValueError) else str(err)]))
                progress.progress((i + 1) / len(files), text=f"อ่านแล้ว {i + 1}/{len(files)}")
                continue
        checked = check_bill(cache[digest], tariff)
        notes.append((name, checked["warnings"]))
        if checked["kwh"] is not None or not pd.isna(checked["month"]):
            rows.append({COL_B_MONTH: "" if pd.isna(checked["month"]) else f"{checked['month']:%Y-%m}",
                         COL_B_KWH: checked["kwh"], COL_B_BILL: checked["bill"],
                         COL_B_DUE: None if pd.isna(checked["due"]) else checked["due"].date(), COL_B_FILE: name})
        progress.progress((i + 1) / len(files), text=f"อ่านแล้ว {i + 1}/{len(files)}")
    progress.empty()
    st.session_state.bill_version = st.session_state.get("bill_version", 0) + 1  # รีเซ็ตตารางแก้ไขเมื่ออ่านชุดใหม่
    st.session_state.bill_rows = pd.DataFrame(rows, columns=[COL_B_MONTH, COL_B_KWH, COL_B_BILL, COL_B_DUE, COL_B_FILE])
    st.session_state.bill_notes = notes


def save_bill_rows(edited: pd.DataFrame) -> None:
    work = edited.copy()
    work["month"] = work[COL_B_MONTH].map(parse_month)
    work["kwh"] = pd.to_numeric(work[COL_B_KWH], errors="coerce")
    work["bill"] = pd.to_numeric(work[COL_B_BILL], errors="coerce")
    bad = work["month"].isna() | work["kwh"].isna() | (work["kwh"] <= 0)
    if bad.all():
        st.error("ยังไม่มีแถวที่มีทั้งเดือนและหน่วยไฟ กรุณากรอกให้ครบก่อน")
        return
    if bad.any():
        st.warning(f"ข้าม {int(bad.sum())} แถวที่ยังไม่มีเดือนหรือหน่วยไฟ")
    existing = st.session_state.get("data")
    if st.session_state.get("data_source") == SOURCE_SAMPLE:
        existing = None  # ไม่ปนกับข้อมูลตัวอย่าง
    try:
        clean, issues, replaced = upsert_months(existing, work[~bad])
    except ValueError as err:
        st.error(str(err))
        return
    st.session_state.data = clean
    st.session_state.data_source = "บิลที่ AI อ่าน" if existing is None else "ข้อมูลเดิม + บิลที่ AI อ่าน"
    cal = calendar_df()
    for _, r in work[~bad].iterrows():  # บิลที่มีวันครบกำหนด → ปฏิทินบิล
        due = r.get(COL_B_DUE)
        if due is not None and not pd.isna(due):
            cal = add_to_calendar(cal, thai_month_label(r["month"]), None if pd.isna(r["bill"]) else float(r["bill"]), due)
    st.session_state.bill_calendar = cal
    st.session_state.temp_df = None
    st.session_state.bill_rows = None
    st.session_state.bill_notes = []
    st.success(f"เพิ่มข้อมูลจากบิล {int((~bad).sum())} เดือนแล้ว" + (f" (แทนที่เดือนเดิม {replaced} เดือน)" if replaced else "")
               + f" ตอนนี้มีข้อมูลทั้งหมด {len(clean)} เดือน")
    for text in issues:
        st.warning(text)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_daily_forecast(lat: float, lon: float) -> pd.DataFrame:
    return get_daily_forecast(lat, lon)


@st.cache_data(show_spinner="กำลังวิเคราะห์...")
def cached_clusters(df: pd.DataFrame):
    return cluster_months(df)


@st.cache_data(show_spinner="กำลังเทรนโมเดลเพื่ออธิบายผล...")
def cached_explain(df: pd.DataFrame):
    return explain_forecast(df)


def current_rate(tariff: TariffConfig) -> float:
    """ค่าไฟเฉลี่ยต่อหน่วยจากบิลล่าสุด ถ้าไม่มีข้อมูลใช้ 300 หน่วยเป็นฐาน"""
    df = current_df()
    if df is not None:
        last = add_bills(df, tariff).iloc[-1]
        if last["kwh"] > 0:
            return float(last["bill_best"] / last["kwh"])
    return estimate_bill(300, tariff) / 300


def appliance_table() -> pd.DataFrame:
    return st.session_state.get("appliance_table", st.session_state.get("appliances", default_appliances()))


def add_appliance_row(row: dict) -> None:
    st.session_state.appliances = pd.concat([appliance_table(), pd.DataFrame([row])], ignore_index=True)
    st.session_state.appliance_table = st.session_state.appliances
    st.session_state.appliance_version = st.session_state.get("appliance_version", 0) + 1
    st.session_state.label_result = None


def calendar_df() -> pd.DataFrame:
    return st.session_state.setdefault("bill_calendar", empty_calendar())


def save_dataset(raw, month_col, kwh_col, bill_col, temp_col, source: str) -> None:
    try:
        clean, issues = clean_monthly(raw, month_col, kwh_col, bill_col, temp_col)
    except ValueError as err:
        st.error(str(err))
        return
    st.session_state.data = clean
    st.session_state.data_source = source
    st.session_state.temp_df = None
    st.success(f"บันทึกข้อมูล {len(clean)} เดือนแล้ว ไปดูผลที่หน้า “ภาพรวม” ได้เลย")
    for text in issues:
        st.warning(text)


def load_sample() -> None:
    st.session_state.data = demo_data()
    st.session_state.data_source = SOURCE_SAMPLE
    st.session_state.temp_df = None


def clear_data() -> None:
    st.session_state.data = None
    st.session_state.data_source = ""
    st.session_state.temp_df = None


# ---------------------------------------------------------------------------
# ล็อกอิน
# ---------------------------------------------------------------------------
if not is_logged_in():
    st.title("⚡ Home Electricity AI")
    st.write("ดูว่าเดือนนี้ใช้ไฟเท่าไร ค่าไฟเท่าไร และเดือนหน้าคาดว่าจะเป็นอย่างไร จากข้อมูลบิลของคุณเอง")
    form_col, _ = st.columns([1, 1])
    with form_col:
        tab_login, tab_signup = st.tabs(["เข้าสู่ระบบ", "สมัครสมาชิก"])
        with tab_login:
            login_form()
            with st.expander("ลืมรหัสผ่าน?"):
                reset_form()
        with tab_signup:
            signup_form()
        if guest_allowed():
            st.divider()
            st.button("ลองใช้โดยไม่ล็อกอิน (โหมดทดสอบ)", on_click=login_as_guest)
    st.stop()

st.session_state.setdefault("data", None)
st.session_state.setdefault("data_source", "")
st.session_state.setdefault("page", PAGE_DATA)

# ---------------------------------------------------------------------------
# แถบข้างและการตั้งค่า
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Home Electricity AI")
    st.caption(st.session_state.get("user_email", ""))
    st.radio("เมนู", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    with st.expander("ตั้งค่าอัตราค่าไฟและที่ตั้ง"):
        tariff_mode = st.radio("วิธีคิดค่าไฟ", [TARIFF_PROGRESSIVE, TARIFF_FLAT], key="tariff_mode")
        ft_satang = st.number_input("ค่า Ft (สตางค์/หน่วย)", 0.0, 200.0, FT_DEFAULT_SATANG, 0.01, format="%.2f",
                                    help="ดูได้จากใบแจ้งค่าไฟ งวดนี้ (ก.ย.–ธ.ค. 2569) คือ 16.23 สตางค์/หน่วย")
        if tariff_mode == TARIFF_FLAT:
            flat_rate = st.number_input("ค่าไฟเฉลี่ย (บาท/หน่วย รวมทุกอย่าง)", 1.0, 10.0, 4.0, 0.05)
            service_11, service_12 = SERVICE_1_1, SERVICE_1_2
        else:
            flat_rate = 4.0
            service_11 = st.number_input("ค่าบริการรายเดือน ประเภท 1.1 (บาท)", 0.0, 100.0, SERVICE_1_1, 0.01, format="%.2f")
            service_12 = st.number_input("ค่าบริการรายเดือน ประเภท 1.2 (บาท)", 0.0, 100.0, SERVICE_1_2, 0.01, format="%.2f")
        latitude = st.number_input("ละติจูด", -90.0, 90.0, 13.7563, 0.0001, format="%.4f")
        longitude = st.number_input("ลองจิจูด", -180.0, 180.0, 100.5018, 0.0001, format="%.4f")
        st.caption(f"อัตราอ้างอิง {TARIFF_AS_OF} ตรวจสอบกับใบแจ้งค่าไฟของคุณอีกครั้ง")
    st.button("ออกจากระบบ", on_click=logout, **stretch("button"))

tariff = TariffConfig(
    mode="flat" if tariff_mode == TARIFF_FLAT else "progressive",
    flat_rate=float(flat_rate), ft_baht=float(ft_satang) / 100,
    service_1_1=float(service_11), service_1_2=float(service_12),
)
page = st.session_state.page

# ---------------------------------------------------------------------------
# หน้า: ข้อมูลค่าไฟ
# ---------------------------------------------------------------------------
if page == PAGE_DATA:
    st.title("ข้อมูลค่าไฟของคุณ")
    st.write("ใส่หน่วยไฟ (kWh) รายเดือนจากใบแจ้งค่าไฟ ยิ่งย้อนหลังนานยิ่งคาดการณ์แม่น แนะนำ 12 เดือนขึ้นไป")

    tab_bill, tab_upload, tab_manual, tab_sample = st.tabs(
        ["📷 ถ่ายรูป/อัปโหลดบิล", "อัปโหลดไฟล์ CSV/Excel", "กรอกเอง", "ใช้ข้อมูลตัวอย่าง"])

    with tab_bill:
        st.write("ถ่ายรูปหรืออัปโหลดใบแจ้งค่าไฟ/ใบเสร็จ (รูปภาพหรือ PDF) ได้หลายใบพร้อมกัน AI จะอ่านเดือน จำนวนหน่วย และยอดเงินให้ "
                 "คุณตรวจและแก้ได้ก่อนบันทึก")
        if not has_gemini_key():
            gemini_missing_notice()
        else:
            bill_files = st.file_uploader("อัปโหลดรูปบิลหรือ PDF", type=BILL_TYPES, accept_multiple_files=True, key="bill_files",
                                          help="รองรับ JPG, PNG, WEBP, HEIC และ PDF ถ่ายให้เห็นทั้งใบ ตัวเลขชัด ไม่สะท้อนแสง")
            use_camera = st.toggle("ถ่ายจากกล้อง", key="bill_camera_on")
            shot = st.camera_input("ถ่ายรูปบิล", key="bill_camera") if use_camera else None
            files = [(f.name, f.getvalue(), mime_for(f.name, f.type)) for f in (bill_files or [])]
            if shot is not None:
                files.append(("รูปจากกล้อง.jpg", shot.getvalue(), "image/jpeg"))
            st.caption("🔒 รูปบิลจะถูกส่งให้ Google Gemini อ่านเฉพาะตัวเลขค่าไฟ ระบบสั่งไม่ให้ดึงชื่อ ที่อยู่ หรือเลขผู้ใช้ไฟ "
                       "และไม่เก็บรูปไว้ ถ้าไม่สบายใจ ปิดชื่อและที่อยู่ก่อนถ่ายได้")
            if st.button(f"ให้ AI อ่านบิล ({len(files)} ไฟล์)", type="primary", disabled=not files, key="read_bills"):
                read_bills_with_ai(files, tariff)

            for name, warns in st.session_state.get("bill_notes", []):
                for w in warns:
                    st.warning(f"**{name}:** {w}")
            bill_rows = st.session_state.get("bill_rows")
            if bill_rows is not None:
                if bill_rows.empty:
                    st.error("ยังอ่านข้อมูลจากบิลไม่ได้ ลองถ่ายใหม่ให้ชัดขึ้น หรือกรอกเองที่แท็บ “กรอกเอง”")
                else:
                    st.markdown("**ตรวจข้อมูลที่ AI อ่านได้** (แก้ในตารางได้เลย) แล้วกดบันทึก")
                    edited_bills = st.data_editor(
                        bill_rows, num_rows="dynamic", hide_index=True, key=f"bill_editor_{st.session_state.get('bill_version', 0)}",
                        column_config={
                            COL_B_MONTH: st.column_config.TextColumn(COL_B_MONTH, help="เช่น 2026-09 หรือ ก.ย. 2569", required=True),
                            COL_B_KWH: st.column_config.NumberColumn(COL_B_KWH, min_value=0.0, format="%.0f"),
                            COL_B_BILL: st.column_config.NumberColumn(COL_B_BILL, min_value=0.0, format="%.2f"),
                            COL_B_DUE: st.column_config.DateColumn(COL_B_DUE, format="DD/MM/YYYY",
                                                                   help="ถ้ามี ระบบจะเพิ่มเข้า “ปฏิทินบิล” ให้"),
                            COL_B_FILE: st.column_config.TextColumn(COL_B_FILE, disabled=True),
                        },
                        **stretch("data_editor"),
                    )
                    has_data = st.session_state.get("data") is not None and st.session_state.get("data_source") != SOURCE_SAMPLE
                    label = "เพิ่มเข้าข้อมูลเดิมของฉัน" if has_data else "บันทึกเป็นข้อมูลของฉัน"
                    if st.button(label, type="primary", key="save_bills"):
                        save_bill_rows(edited_bills)
                    if has_data:
                        st.caption("เดือนที่ซ้ำกับข้อมูลเดิมจะใช้ค่าจากบิลแทน")

    with tab_upload:
        st.download_button("ดาวน์โหลดเทมเพลต CSV", template_csv(), "electricity_template.csv", "text/csv")
        upload = st.file_uploader("อัปโหลดไฟล์ CSV หรือ Excel (.xlsx)", type=["csv", "xlsx"])
        if upload is not None:
            raw = None
            try:
                raw = read_table(upload)
                raw.columns = [str(c) for c in raw.columns]
            except Exception as err:  # ไฟล์เสีย/รูปแบบไม่รองรับ
                st.error(f"อ่านไฟล์ไม่ได้: {err}")
            if raw is not None and raw.empty:
                st.error("ไฟล์นี้ไม่มีข้อมูล")
            elif raw is not None:
                st.caption(f"พบ {len(raw)} แถว (แสดง 5 แถวแรก)")
                st.dataframe(raw.head(5), hide_index=True, **stretch("dataframe"))
                guess = detect_columns(raw)
                cols = list(raw.columns)

                def pick(label: str, key: str, required: bool):
                    options = cols if required else [NONE_OPTION] + cols
                    default = guess.get(key)
                    idx = options.index(default) if default in options else 0
                    return st.selectbox(label, options, index=idx, key=f"map_{key}_{upload.name}")

                st.markdown("**เลือกคอลัมน์ที่ตรงกับข้อมูล** (ระบบเดาให้แล้ว แก้ได้)")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    month_col = pick("เดือน", "month", True)
                with c2:
                    kwh_col = pick("หน่วยไฟ (kWh)", "kwh", True)
                with c3:
                    bill_col = pick("ค่าไฟจริง (บาท)", "bill", False)
                with c4:
                    temp_col = pick("อุณหภูมิ (ถ้ามี)", "temp", False)
                if st.button("ใช้ข้อมูลนี้", type="primary", key="use_upload"):
                    save_dataset(raw, month_col, kwh_col,
                                 None if bill_col == NONE_OPTION else bill_col,
                                 None if temp_col == NONE_OPTION else temp_col,
                                 f"ไฟล์ {upload.name}")

    with tab_manual:
        st.caption("พิมพ์เดือนได้หลายแบบ เช่น 2026-09, 09/2026, ก.ย. 2569 ส่วนค่าไฟจริงใส่หรือไม่ใส่ก็ได้")
        if "manual_df" not in st.session_state:
            months = pd.date_range(end=pd.Timestamp.today().to_period("M").to_timestamp() - pd.DateOffset(months=1),
                                   periods=6, freq="MS")
            st.session_state.manual_df = pd.DataFrame({
                "เดือน": [f"{m:%Y-%m}" for m in months],
                "หน่วยไฟ (kWh)": pd.Series([None] * 6, dtype="float64"),
                "ค่าไฟ (บาท)": pd.Series([None] * 6, dtype="float64"),
            })
        edited = st.data_editor(
            st.session_state.manual_df, num_rows="dynamic", key="manual_editor", hide_index=True,
            column_config={
                "เดือน": st.column_config.TextColumn("เดือน", required=True),
                "หน่วยไฟ (kWh)": st.column_config.NumberColumn("หน่วยไฟ (kWh)", min_value=0.0, format="%.0f"),
                "ค่าไฟ (บาท)": st.column_config.NumberColumn("ค่าไฟ (บาท)", min_value=0.0, format="%.2f"),
            },
            **stretch("data_editor"),
        )
        if st.button("บันทึกข้อมูลที่กรอก", type="primary", key="save_manual"):
            save_dataset(edited, "เดือน", "หน่วยไฟ (kWh)", "ค่าไฟ (บาท)", None, "กรอกเอง")

    with tab_sample:
        st.write("ข้อมูลสมมติ 18 เดือน ใช้ลองดูว่าแอปทำงานอย่างไร ไม่ใช่ข้อมูลจริง")
        st.button("โหลดข้อมูลตัวอย่าง", on_click=load_sample, key="load_sample")

    df = current_df()
    if df is not None:
        st.divider()
        st.subheader("ข้อมูลที่ใช้อยู่")
        st.caption(f"{len(df)} เดือน ({thai_month_label(df['month'].iloc[0])} – {thai_month_label(df['month'].iloc[-1])}) "
                   f"แหล่งข้อมูล: {st.session_state.get('data_source') or '-'}")
        if len(df) < 12:
            st.warning("ข้อมูลยังไม่ถึง 12 เดือน คาดการณ์ได้แต่ความน่าเชื่อถือต่ำ")
        view = pd.DataFrame({
            "เดือน": [thai_month_label(m) for m in df["month"]],
            "หน่วยไฟ (kWh)": df["kwh"].round(1),
            "ค่าไฟจริง (บาท)": df["bill"],
            "หมายเหตุ": np.where(df["filled"], "ระบบเติมค่าประมาณ", ""),
        })
        if df["temp"].notna().any():
            view["อุณหภูมิเฉลี่ย (°C)"] = df["temp"].round(1)
        st.dataframe(view, hide_index=True, **stretch("dataframe"))
        left, right = st.columns(2)
        with left:
            st.download_button("ดาวน์โหลดข้อมูลนี้ (CSV)", export_csv(df), "my_electricity.csv", "text/csv", **stretch("download_button"))
        with right:
            st.button("ล้างข้อมูล", on_click=clear_data, **stretch("button"))
        st.caption("ข้อมูลเก็บในเบราว์เซอร์ระหว่างใช้งานเท่านั้น ถ้ารีเฟรชหรือออกจากระบบจะหายไป ดาวน์โหลดไว้เพื่ออัปโหลดครั้งต่อไป")

# ---------------------------------------------------------------------------
# หน้า: ภาพรวม
# ---------------------------------------------------------------------------
elif page == PAGE_OVERVIEW:
    st.title("ภาพรวมการใช้ไฟ")
    d = add_bills(require_data(), tariff)
    stats = calculate_stats(d)
    latest = d.iloc[-1]
    labels = [thai_month_label(m) for m in d["month"]]
    latest_label = labels[-1]

    cols = st.columns(4)
    with cols[0]:
        with st.container(border=True):
            st.metric(f"ใช้ไฟ {latest_label}", f"{latest['kwh']:,.0f} หน่วย",
                      delta=None if stats["mom"] is None else f"{stats['mom']:+.1f}% จากเดือนก่อน", delta_color="inverse")
    with cols[1]:
        with st.container(border=True):
            tag = "บิลจริง" if latest["is_actual"] else "ประมาณ"
            st.metric(f"ค่าไฟ {latest_label} ({tag})", baht(latest["bill_best"]),
                      delta=f"≈ ฿{latest['bill_best'] / latest['kwh']:.2f} ต่อหน่วย" if latest["kwh"] > 0 else None, delta_color="off")
    with cols[2]:
        with st.container(border=True):
            st.metric("เฉลี่ยต่อเดือน", f"{stats['mean']:,.0f} หน่วย", delta=f"≈ {baht(d['bill_best'].mean())} ต่อเดือน", delta_color="off")
    with cols[3]:
        with st.container(border=True):
            st.metric("เทียบเดือนเดียวกันปีก่อน", "—" if stats["yoy"] is None else f"{stats['yoy']:+.1f}%",
                      help="ต้องมีข้อมูลย้อนหลังครบ 13 เดือนจึงเทียบได้")

    tab_kwh, tab_bill, tab_temp = st.tabs(["หน่วยไฟ (kWh)", "ค่าไฟ (บาท)", "อุณหภูมิกับการใช้ไฟ"])
    with tab_kwh:
        fig = go.Figure()
        fig.add_bar(x=labels, y=d["kwh"], name="หน่วยไฟ",
                    marker_color=[COLORS["blue_light"]] * (len(d) - 1) + [COLORS["blue"]],
                    hovertemplate="%{y:,.0f} หน่วย<extra></extra>")
        fig.add_scatter(x=labels, y=d["kwh"].rolling(3, min_periods=3).mean(), mode="lines", name="เฉลี่ย 3 เดือน",
                        line=dict(color=COLORS["amber"], width=3), hovertemplate="%{y:,.0f} หน่วย<extra></extra>")
        fig.add_hline(y=stats["mean"], line_dash="dot", line_color=COLORS["muted"],
                      annotation_text="ค่าเฉลี่ยทั้งหมด", annotation_position="top left")
        fig.update_yaxes(title_text="หน่วย (kWh)")
        st.plotly_chart(style_fig(fig), config=PLOT_CONFIG, **stretch("plotly_chart"))
        st.caption("แท่งสีเข้มคือเดือนล่าสุด เส้นสีส้มคือค่าเฉลี่ย 3 เดือนที่ช่วยดูแนวโน้ม")

    with tab_bill:
        fig = go.Figure()
        fig.add_bar(x=labels, y=d["bill_best"], name="ค่าไฟ",
                    marker_color=[COLORS["amber"] if a else COLORS["amber_light"] for a in d["is_actual"]],
                    hovertemplate="฿%{y:,.0f}<extra></extra>")
        fig.update_yaxes(title_text="บาท")
        st.plotly_chart(style_fig(fig), config=PLOT_CONFIG, **stretch("plotly_chart"))
        if (~d["is_actual"]).any():
            st.caption("สีเข้มคือบิลจริงที่คุณกรอก สีอ่อนคือประมาณจากอัตราค่าไฟ (ไม่ใช่ยอดบิลทางการ)")
        else:
            st.caption("ค่าไฟทุกเดือนมาจากบิลจริงที่คุณกรอก")

    with tab_temp:
        eff = temperature_effect(d)
        if d["temp"].notna().sum() >= 3:
            sub = d.dropna(subset=["temp"])
            fig = go.Figure()
            fig.add_scatter(x=sub["temp"], y=sub["kwh"], mode="markers", name="รายเดือน",
                            text=[thai_month_label(m) for m in sub["month"]], marker=dict(size=11, color=COLORS["blue"]),
                            hovertemplate="%{text}<br>%{x:.1f}°C, %{y:,.0f} หน่วย<extra></extra>")
            if eff:
                xs = np.array([sub["temp"].min(), sub["temp"].max()])
                fig.add_scatter(x=xs, y=eff["slope"] * xs + eff["intercept"], mode="lines", name="เส้นแนวโน้ม",
                                line=dict(color=COLORS["amber"], width=3, dash="dot"))
            style_fig(fig)
            fig.update_xaxes(type="linear", title_text="อุณหภูมิเฉลี่ยรายเดือน (°C)", tickangle=0)
            fig.update_yaxes(title_text="หน่วยไฟ (kWh)")
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, config=PLOT_CONFIG, **stretch("plotly_chart"))
            if eff:
                st.caption(f"อุณหภูมิสูงขึ้น 1°C สัมพันธ์กับการใช้ไฟเปลี่ยนไปราว {eff['slope']:+.1f} หน่วย (r = {eff['r']:.2f}) "
                           "ความสัมพันธ์ไม่ได้แปลว่าเป็นสาเหตุโดยตรง")
        else:
            st.write("ยังไม่มีข้อมูลอุณหภูมิ ดึงอุณหภูมิเฉลี่ยรายเดือนย้อนหลังตามที่ตั้งที่ตั้งค่าไว้ได้จาก Open-Meteo (ฟรี)")
            if st.button("ดึงอุณหภูมิรายเดือน", key="fetch_temp"):
                with st.spinner("กำลังดึงข้อมูลอุณหภูมิ..."):
                    t = cached_monthly_temp(float(latitude), float(longitude),
                                            d["month"].min().strftime("%Y-%m-%d"),
                                            (d["month"].max() + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"))
                if t.empty:
                    st.error("ดึงข้อมูลอุณหภูมิไม่สำเร็จ ตรวจสอบอินเทอร์เน็ตแล้วลองใหม่ หรือใส่คอลัมน์อุณหภูมิในไฟล์ของคุณเอง")
                else:
                    st.session_state.temp_df = t
                    st.rerun()

    st.subheader("สถิติสำคัญ")
    st.table(pd.DataFrame({
        "ตัวชี้วัด": ["ใช้ไฟสูงสุด", "ใช้ไฟต่ำสุด", "เฉลี่ยต่อเดือน", "ความผันผวน", "แนวโน้ม"],
        "ค่า": [
            f"{stats['max']:,.0f} หน่วย ({thai_month_label(stats['max_month'])})",
            f"{stats['min']:,.0f} หน่วย ({thai_month_label(stats['min_month'])})",
            f"{stats['mean']:,.0f} หน่วย",
            f"{stats['volatility']:.1f}% (ยิ่งสูงยิ่งใช้ไฟไม่สม่ำเสมอ)",
            f"{stats['slope']:+.1f} หน่วยต่อเดือน",
        ],
    }).set_index("ตัวชี้วัด"))

    weather = cached_weather(float(latitude), float(longitude))
    if weather.get("temperature") is not None:
        st.caption(f"อากาศตอนนี้ตามที่ตั้งที่ตั้งค่าไว้: {weather['temperature']}°C ความชื้น {weather.get('humidity', '-')}% (Open-Meteo)")

# ---------------------------------------------------------------------------
# หน้า: คาดการณ์และคำแนะนำ
# ---------------------------------------------------------------------------
elif page == PAGE_FORECAST:
    st.title("คาดการณ์เดือนหน้า")
    d = add_bills(require_data(), tariff)
    if len(d) < 3:
        st.warning("ต้องมีข้อมูลอย่างน้อย 3 เดือนจึงจะคาดการณ์ได้")
        st.stop()
    stats = calculate_stats(d)
    fc = cached_forecast(d[["month", "kwh"]])
    calib = calibration(d)
    pred_bill = estimate_bill(fc.kwh, tariff) * calib
    low_bill = estimate_bill(fc.low, tariff) * calib
    high_bill = estimate_bill(fc.high, tariff) * calib
    latest = d.iloc[-1]
    labels = [thai_month_label(m) for m in d["month"]]
    next_label = thai_month_label(fc.next_month)

    st.subheader(f"คาดว่าเดือน {next_label}")
    cols = st.columns(3)
    with cols[0]:
        with st.container(border=True):
            chg = (fc.kwh - latest["kwh"]) / latest["kwh"] * 100 if latest["kwh"] else None
            st.metric("หน่วยไฟที่คาด", f"{fc.kwh:,.0f} หน่วย",
                      delta=None if chg is None else f"{chg:+.1f}% จากเดือนล่าสุด", delta_color="inverse")
    with cols[1]:
        with st.container(border=True):
            st.metric("ค่าไฟที่คาด", baht(pred_bill), delta=f"{pred_bill - latest['bill_best']:+,.0f} บาท จากเดือนล่าสุด", delta_color="inverse")
    with cols[2]:
        with st.container(border=True):
            st.metric("ช่วงที่น่าจะเป็น (~80%)", f"{baht(low_bill)} – {baht(high_bill)}")
    st.markdown(f"ความน่าเชื่อถือ: {reliability_chip(fc.reliability)}", unsafe_allow_html=True)
    st.caption(f"วิธีที่ใช้: {fc.model_name} • ข้อมูล {fc.n_points} เดือน"
               + (f" • ค่าไฟปรับตามบิลจริงของคุณ ×{calib:.2f}" if abs(calib - 1) > 0.03 else ""))

    fig = go.Figure()
    fig.add_scatter(x=labels, y=d["kwh"], mode="lines+markers", name="ใช้จริง",
                    line=dict(color=COLORS["blue"], width=3), hovertemplate="%{y:,.0f} หน่วย<extra></extra>")
    fig.add_scatter(x=[labels[-1], next_label], y=[latest["kwh"], fc.kwh], mode="lines", showlegend=False,
                    line=dict(color=COLORS["amber"], width=2, dash="dot"), hoverinfo="skip")
    fig.add_scatter(x=[next_label], y=[fc.kwh], mode="markers", name="คาดการณ์",
                    marker=dict(size=13, color=COLORS["amber"]),
                    error_y=dict(type="data", symmetric=False, array=[fc.high - fc.kwh], arrayminus=[fc.kwh - fc.low],
                                 color=COLORS["amber"], thickness=2),
                    hovertemplate="%{y:,.0f} หน่วย<extra></extra>")
    fig.update_yaxes(title_text="หน่วย (kWh)")
    st.plotly_chart(style_fig(fig), config=PLOT_CONFIG, **stretch("plotly_chart"))

    st.subheader("ข้อสังเกตจากข้อมูลของคุณ")
    saving = estimate_bill(latest["kwh"], tariff) - estimate_bill(latest["kwh"] * 0.9, tariff)
    insights = generate_insights({
        "latest_kwh": float(latest["kwh"]), "latest_bill": float(latest["bill_best"]),
        "latest_label": labels[-1], "pred_kwh": fc.kwh, "pred_bill": pred_bill,
        "mean_kwh": stats["mean"], "slope": stats["slope"], "n_months": stats["n"],
        "reliability": fc.reliability, "temp_effect": temperature_effect(d), "saving_10pct": saving * calib,
    })
    renderers = {"success": st.success, "warning": st.warning, "info": st.info}
    for item in insights:
        renderers[item["level"]](item["text"])

    st.subheader("อากาศ 7 วันข้างหน้า")
    wx = cached_daily_forecast(float(latitude), float(longitude))
    if wx.empty:
        st.caption("ดึงพยากรณ์อากาศไม่สำเร็จ (Open-Meteo) ลองใหม่ภายหลัง")
    else:
        recent_temp = d["temp"].dropna().iloc[-1] if d["temp"].notna().any() else None
        outlook = weather_outlook(wx, temperature_effect(d), None if recent_temp is None else float(recent_temp),
                                  ac_profile(appliance_table())["kwh"])
        st.info(outlook["summary"] + ("" if recent_temp is not None else
                " (ดึงอุณหภูมิรายเดือนที่หน้า “ภาพรวม” ก่อน ระบบจะประมาณผลต่อค่าไฟให้ได้)"))
        fig = go.Figure()
        day_labels = [f"{t:%d/%m}" for t in wx["date"]]
        fig.add_scatter(x=day_labels, y=wx["tmax"], mode="lines+markers", name="สูงสุด",
                        line=dict(color=COLORS["red"], width=3), hovertemplate="%{y:.1f}°C<extra></extra>")
        fig.add_scatter(x=day_labels, y=wx["tmin"], mode="lines+markers", name="ต่ำสุด",
                        line=dict(color=COLORS["blue"], width=3), hovertemplate="%{y:.1f}°C<extra></extra>")
        fig.add_hline(y=35, line_dash="dot", line_color=COLORS["muted"], annotation_text="ร้อนจัด 35°C", annotation_position="top left")
        fig.update_yaxes(title_text="°C")
        st.plotly_chart(style_fig(fig, height=300), config=PLOT_CONFIG, **stretch("plotly_chart"))
        st.dataframe(pd.DataFrame({
            "วันที่": [f"{t.day} {thai_month_label(t)}" for t in outlook["days"]["date"]],
            "สูงสุด/ต่ำสุด (°C)": [f"{a:.0f} / {b:.0f}" for a, b in zip(outlook["days"]["tmax"], outlook["days"]["tmin"])],
            "ฝน (มม.)": outlook["days"]["rain_mm"].round(1),
            "คำแนะนำ": outlook["days"]["คำแนะนำ"],
        }), hide_index=True, **stretch("dataframe"))

    with st.expander("วิธีลดค่าไฟที่ทำได้ทันที"):
        for title, detail in SAVING_TIPS:
            st.markdown(f"- **{title}** {detail}")

    with st.expander("ดูว่าการคาดการณ์แม่นแค่ไหน"):
        if fc.backtest.empty:
            st.write("ข้อมูลยังน้อยเกินกว่าจะทดสอบย้อนหลัง จึงใช้ค่าเฉลี่ย 3 เดือนล่าสุด ยิ่งมีข้อมูล 12 เดือนขึ้นไประบบยิ่งเลือกวิธีที่เหมาะกับบ้านคุณได้")
        else:
            st.write("ระบบลองทำนายเดือนที่ผ่านมาโดยปิดข้อมูลเดือนนั้นไว้ แล้วเทียบกับค่าจริง ยิ่งคลาดเคลื่อนน้อยยิ่งดี และเลือกวิธีที่ดีที่สุดให้อัตโนมัติ")
            st.dataframe(fc.backtest, hide_index=True, **stretch("dataframe"))
        st.caption("ผลคาดการณ์เป็นการประมาณ ไม่ใช่ค่าไฟจริง อาจคลาดเคลื่อนถ้าพฤติกรรมการใช้ไฟเปลี่ยน เช่น ซื้อแอร์ใหม่ หรือมีคนอยู่บ้านเพิ่ม")

# ---------------------------------------------------------------------------
# หน้า: วิเคราะห์เชิงลึก (Clustering + XAI)
# ---------------------------------------------------------------------------
elif page == PAGE_INSIGHT:
    st.title("วิเคราะห์เชิงลึก")
    d = require_data()

    prof = household_profile(d)
    st.subheader("รูปแบบการใช้ไฟของบ้านคุณ")
    st.success(prof["text"])
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.metric("หน้าร้อนใช้ไฟมากกว่าหน้าหนาว", "—" if prof["season_gap_pct"] is None else f"{prof['season_gap_pct']:+.0f}%",
                      help="เทียบค่าเฉลี่ย มี.ค.–พ.ค. กับ พ.ย.–ม.ค.")
    with c2:
        with st.container(border=True):
            st.metric("แนวโน้มต่อเดือน", f"{prof['trend_pct_per_month']:+.1f}%")
    with c3:
        with st.container(border=True):
            st.metric("ความผันผวน", f"{prof['cv']:.0f}%")

    st.subheader("จัดกลุ่มเดือนด้วย K-Means")
    cl = cached_clusters(d)
    if cl is None:
        st.info("ต้องมีข้อมูลจริงอย่างน้อย 6 เดือนจึงจัดกลุ่มได้")
    else:
        t = cl["table"]
        fig = go.Figure()
        for name in cl["summary"]["กลุ่ม"]:
            part = t[t["กลุ่ม"] == name]
            fig.add_bar(x=[thai_month_label(m) for m in part["month"]], y=part["kwh"], name=name,
                        marker_color=CLUSTER_COLORS.get(name, COLORS["muted"]), hovertemplate="%{x}: %{y:,.0f} หน่วย<extra></extra>")
        style_fig(fig)
        fig.update_xaxes(categoryorder="array", categoryarray=[thai_month_label(m) for m in t["month"]])
        fig.update_yaxes(title_text="หน่วย (kWh)")
        st.plotly_chart(fig, config=PLOT_CONFIG, **stretch("plotly_chart"))
        st.dataframe(cl["summary"], hide_index=True, **stretch("dataframe"))
        st.caption(f"K-Means แบ่ง {cl['k']} กลุ่มจากฟีเจอร์: {', '.join(cl['features'])} (ปรับมาตรฐานก่อน และให้น้ำหนักหน่วยไฟ ×2) "
                   "ใช้หาว่าเดือนแบบไหนที่ทำให้ค่าไฟสูง")
        high = cl["summary"][cl["summary"]["กลุ่ม"] == "ใช้ไฟสูง"]
        if len(high) and high["หน้าร้อน (%)"].iloc[0] >= 50:
            st.info("เดือนในกลุ่ม “ใช้ไฟสูง” ส่วนใหญ่เป็นหน้าร้อน แอร์น่าจะเป็นตัวหลัก ลองดูหน้า “จำลองและความคุ้มค่า” ว่าปรับแอร์แล้วประหยัดได้เท่าไร")

    st.subheader("AI อธิบายการคาดการณ์ (Explainable AI)")
    ex = cached_explain(d)
    if ex is None:
        st.info("ต้องมีข้อมูลอย่างน้อย 12 เดือนจึงอธิบายโมเดลได้ (ต้องครบฤดูกาล)")
    else:
        imp = ex["importance"].sort_values("ผลต่อการคาดการณ์ (%)")
        fig = go.Figure(go.Bar(x=imp["ผลต่อการคาดการณ์ (%)"], y=imp["ปัจจัย"], orientation="h", marker_color=COLORS["amber"],
                               hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
        style_fig(fig, height=max(240, 60 * len(imp) + 60))
        fig.update_xaxes(type="linear", title_text="ผลต่อการคาดการณ์ (%)", tickangle=0)
        fig.update_yaxes(type="category")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, config=PLOT_CONFIG, **stretch("plotly_chart"))
        st.write(describe_importance(ex["importance"]))
        st.caption(f"วิธี: Random Forest ทำนายหน่วยไฟจากเดือนก่อนหน้า ฤดูกาล{' และอุณหภูมิ' if ex['has_temp'] else ''} ({ex['n']} ตัวอย่าง) "
                   "แล้ววัดความสำคัญด้วย Permutation Importance คือสลับค่าปัจจัยนั้นแบบสุ่ม ดูว่าโมเดลแย่ลงเท่าไร")
        if ex["temp_test"]:
            tt = ex["temp_test"]
            verdict = "ช่วยให้แม่นขึ้น" if tt["temp_helps"] else "ไม่ได้ช่วยให้แม่นขึ้น"
            st.markdown(f"**ทดลองใส่อุณหภูมิในโมเดล:** คลาดเคลื่อนเฉลี่ย {tt['mae_with_temp_kwh']:,.1f} หน่วย (ใส่) เทียบ "
                        f"{tt['mae_without_temp_kwh']:,.1f} หน่วย (ไม่ใส่) จากการทดสอบย้อนหลัง {tt['folds']} เดือน → อุณหภูมิ{verdict}")
        elif not ex["has_temp"]:
            st.caption("ยังไม่มีอุณหภูมิครบทุกเดือน ดึงได้ที่หน้า “ภาพรวม” แท็บอุณหภูมิ แล้วกลับมาดูว่าอุณหภูมิช่วยให้แม่นขึ้นไหม")

# ---------------------------------------------------------------------------
# หน้า: จำลองและความคุ้มค่า (What-if + Payback)
# ---------------------------------------------------------------------------
elif page == PAGE_PLANNER:
    st.title("จำลองและความคุ้มค่า")
    tab_whatif, tab_payback = st.tabs(["ถ้า...จะเป็นอย่างไร (What-if)", "คุ้มไหมถ้าซื้อเครื่องใหม่"])

    with tab_whatif:
        d = current_df()
        if d is None:
            st.info("What-if ต้องใช้ข้อมูลค่าไฟของคุณเป็นจุดเริ่มต้น เพิ่มข้อมูลที่หน้า “ข้อมูลค่าไฟ” ก่อน (แท็บความคุ้มค่าใช้ได้เลย)")
            st.button("ไปที่หน้า “ข้อมูลค่าไฟ”", on_click=go_to, args=(PAGE_DATA,), key="wi_go_data")
        else:
            base_kwh = float(d["kwh"].iloc[-1])
            calib = calibration(add_bills(d, tariff))
            ac = ac_profile(appliance_table())
            st.caption(f"เริ่มจากการใช้ไฟเดือนล่าสุด ({thai_month_label(d['month'].iloc[-1])}) {base_kwh:,.0f} หน่วย ปรับแถบด้านล่างแล้วดูผลทันที")
            c1, c2 = st.columns(2)
            with c1:
                people_now = st.number_input("ตอนนี้มีคนในบ้าน (คน)", 1, 20, 3, key="wi_people_now")
                people_after = st.slider("ถ้าคนในบ้านเป็น (คน)", 1, 20, int(people_now), key="wi_people_after")
                wfh = st.slider("ทำงาน/เรียนที่บ้านเพิ่ม (วัน/สัปดาห์)", 0, 7, 0, key="wi_wfh")
            with c2:
                ac_temp = st.slider("ปรับอุณหภูมิแอร์ (°C)", -3, 4, 0, key="wi_ac_temp", help="+ คือตั้งสูงขึ้น (ประหยัด) − คือตั้งต่ำลง")
                ac_hours = st.slider("เปิดแอร์เพิ่ม/ลด (ชม./วัน)", -8, 8, 0, key="wi_ac_hours")
                units = st.number_input("จำนวนแอร์ที่ได้รับผล (เครื่อง)", 1, 10, max(1, ac["count"]), key="wi_units")
            r = what_if(base_kwh, appliance_table(), int(people_now), int(people_after), wfh, ac_temp, ac_hours, int(units))
            before, after = estimate_bill(r["base_kwh"], tariff) * calib, estimate_bill(r["new_kwh"], tariff) * calib
            m1, m2, m3 = st.columns(3)
            with m1:
                with st.container(border=True):
                    st.metric("หน่วยไฟต่อเดือน", f"{r['new_kwh']:,.0f} หน่วย", delta=f"{r['new_kwh'] - r['base_kwh']:+,.0f} หน่วย", delta_color="inverse")
            with m2:
                with st.container(border=True):
                    st.metric("ค่าไฟต่อเดือน", baht(after), delta=f"{after - before:+,.0f} บาท", delta_color="inverse")
            with m3:
                with st.container(border=True):
                    st.metric("ต่อปี", f"{(after - before) * 12:+,.0f} บาท", delta_color="off")
            if r["items"]:
                fig = go.Figure(go.Waterfall(
                    x=["เดิม"] + [i["เรื่อง"] for i in r["items"]] + ["ใหม่"],
                    measure=["absolute"] + ["relative"] * len(r["items"]) + ["total"],
                    y=[r["base_kwh"]] + [i["เปลี่ยน_kwh"] for i in r["items"]] + [0],
                    increasing=dict(marker_color=COLORS["red"]), decreasing=dict(marker_color=COLORS["green"]),
                    totals=dict(marker_color=COLORS["blue"]), hovertemplate="%{x}: %{y:+,.0f} หน่วย<extra></extra>"))
                style_fig(fig, height=360)
                fig.update_xaxes(tickangle=0)
                fig.update_yaxes(title_text="หน่วย (kWh)")
                st.plotly_chart(fig, config=PLOT_CONFIG, **stretch("plotly_chart"))
            with st.expander("สมมติฐานที่ใช้คำนวณ"):
                for a in ASSUMPTIONS:
                    st.markdown(f"- {a}")
                st.caption("แอร์: " + (f"จากตารางเครื่องใช้ไฟฟ้า {ac['watt']:,.0f} วัตต์ ทำงานจริง {ac['duty']:.0%}" if ac["from_table"]
                                       else "ไม่พบในตาราง ใช้ค่าเริ่มต้น"))

    with tab_payback:
        rate = current_rate(tariff)
        st.caption(f"เทียบเครื่องเดิมกับเครื่องใหม่ที่ประหยัดไฟกว่า คิดค่าไฟเฉลี่ย ฿{rate:.2f} ต่อหน่วย (จากบิลล่าสุดของคุณ)")
        presets = {
            "กำหนดเอง": (1200, 900, 8, 18000, 100, 100),
            "แอร์ธรรมดา → แอร์อินเวอร์เตอร์ 12,000 BTU": (1200, 1000, 8, 15000, 85, 55),
            "ตู้เย็นเก่า 10 ปี → ตู้เย็นเบอร์ 5 ใหม่": (180, 90, 24, 9000, 45, 40),
            "หลอดตะเกียบ 10 ดวง → หลอด LED": (18 * 10, 9 * 10, 5, 800, 100, 100),
        }
        preset = st.selectbox("ตัวอย่าง", list(presets), key="pb_preset")
        ow, nw, hrs, price, od, nd = presets[preset]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**เครื่องเดิม**")
            old_w = st.number_input("กำลังไฟ (วัตต์)", 1, 20000, ow, key=f"pb_old_w_{preset}")
            old_d = st.slider("ทำงานจริง (%)", 10, 100, od, 5, key=f"pb_old_d_{preset}")
        with c2:
            st.markdown("**เครื่องใหม่**")
            new_w = st.number_input("กำลังไฟ (วัตต์) ", 1, 20000, nw, key=f"pb_new_w_{preset}")
            new_d = st.slider("ทำงานจริง (%) ", 10, 100, nd, 5, key=f"pb_new_d_{preset}",
                              help="แอร์อินเวอร์เตอร์ปรับรอบได้ จึงทำงานเต็มกำลังน้อยกว่าแอร์ธรรมดา")
        c3, c4 = st.columns(2)
        with c3:
            hours_pb = st.number_input("ใช้วันละ (ชั่วโมง)", 0.0, 24.0, float(hrs), 0.5, key=f"pb_hours_{preset}")
        with c4:
            price_pb = st.number_input("ราคาเครื่องใหม่ (บาท)", 0, 500000, price, 500, key=f"pb_price_{preset}")
        pb = payback(old_w, new_w, hours_pb, price_pb, rate, old_duty=old_d, new_duty=new_d)
        m1, m2, m3 = st.columns(3)
        with m1:
            with st.container(border=True):
                st.metric("ประหยัดต่อเดือน", baht(max(pb["save_baht_month"], 0)), delta=f"{pb['save_kwh']:,.0f} หน่วย", delta_color="off")
        with m2:
            with st.container(border=True):
                st.metric("คืนทุนใน", "ไม่คืนทุน" if pb["payback_years"] is None else
                          (f"{pb['payback_months']:.0f} เดือน" if pb["payback_months"] < 24 else f"{pb['payback_years']:.1f} ปี"))
        with m3:
            with st.container(border=True):
                st.metric("ผลรวมใน 10 ปี", f"{pb['save_10y']:+,.0f} บาท", help="เงินที่ประหยัดได้ 10 ปี ลบราคาเครื่อง")
        if pb["payback_months"] is not None:
            months_axis = list(range(0, 121, 6))
            fig = go.Figure()
            fig.add_scatter(x=months_axis, y=[pb["save_baht_month"] * m - price_pb for m in months_axis], mode="lines",
                            line=dict(color=COLORS["green"], width=3), name="เงินที่ได้คืนสะสม",
                            hovertemplate="เดือนที่ %{x}: %{y:+,.0f} บาท<extra></extra>")
            fig.add_hline(y=0, line_color=COLORS["muted"])
            style_fig(fig, height=300)
            fig.update_xaxes(type="linear", title_text="เดือน", tickangle=0)
            fig.update_yaxes(title_text="บาท")
            st.plotly_chart(fig, config=PLOT_CONFIG, **stretch("plotly_chart"))
            verdict = "คุ้ม" if pb["payback_years"] <= 5 else "คุ้มระยะยาว" if pb["payback_years"] <= 10 else "ไม่ค่อยคุ้มด้านค่าไฟ"
            st.info(f"สรุป: {verdict} เครื่องใช้ไฟฟ้าส่วนใหญ่ใช้งานราว 8–10 ปี ถ้าคืนทุนเกิน 10 ปี ควรซื้อเมื่อเครื่องเดิมเสียมากกว่า")
        else:
            st.warning("เครื่องใหม่ใช้ไฟเท่าหรือมากกว่าเครื่องเดิม จึงไม่คืนทุนจากค่าไฟ")

# ---------------------------------------------------------------------------
# หน้า: ถาม AI
# ---------------------------------------------------------------------------
elif page == PAGE_CHAT:
    head, btn = st.columns([4, 1])
    with head:
        st.title("ถาม AI เรื่องค่าไฟ")
    with btn:
        st.button("ล้างแชท", on_click=clear_chat, **stretch("button"))
    st.caption("AI ตอบจากข้อมูลค่าไฟของคุณ โดยเรียกเครื่องมือคำนวณในระบบ (ไม่เดาตัวเลขเอง) ค่าไฟที่ได้เป็นค่าประมาณ")
    if not has_gemini_key():
        gemini_missing_notice()
        st.stop()
    if current_df() is None:
        st.info("ยังไม่มีข้อมูลค่าไฟ ถามเรื่องทั่วไปได้ แต่ถ้าอยากให้ AI วิเคราะห์บ้านคุณ ให้เพิ่มข้อมูลก่อน")
    elif st.session_state.get("data_source") == SOURCE_SAMPLE:
        st.warning("ตอนนี้ใช้ข้อมูลตัวอย่าง คำตอบจะอ้างอิงตัวเลขสมมติ")

    chat = st.session_state.setdefault("chat", [])
    if not chat:
        st.markdown("**ลองถาม:**")
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            st.button(q, key=f"suggest_{i}", on_click=ask_question, args=(q,))

    for msg in chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                with st.expander(f"ข้อมูลที่ AI ใช้ ({len(msg['trace'])} ขั้นตอน)"):
                    for step in msg["trace"]:
                        st.markdown(f"`{step['tool']}` {json.dumps(step['arguments'], ensure_ascii=False)}")
                        st.json(step["result"], expanded=False)

    question = st.chat_input("พิมพ์คำถาม เช่น ทำไมเดือนนี้ค่าไฟแพง")
    question = question or st.session_state.pop("pending_question", None)
    if question:
        history = [{"role": m["role"], "content": m["content"]} for m in chat]
        chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("กำลังคิด..."):
                try:
                    result = run_chat(question, history, chat_context(tariff))
                    answer, trace = result["answer"], result["trace"]
                except Exception as err:
                    answer, trace = friendly_error(err), []
            st.markdown(answer)
        chat.append({"role": "assistant", "content": answer, "trace": trace})
        st.rerun()

# ---------------------------------------------------------------------------
# หน้า: เครื่องใช้ไฟฟ้า
# ---------------------------------------------------------------------------
elif page == PAGE_APPLIANCE:
    st.title("เครื่องใช้ไฟฟ้าตัวไหนกินไฟมากที่สุด")
    st.write("แก้ตารางได้เลย ค่าเริ่มต้นเป็นค่าประมาณทั่วไป ควรดูกำลังไฟ (วัตต์) จากป้ายเครื่องของคุณ")
    if "appliances" not in st.session_state:
        st.session_state.appliances = default_appliances()
    with st.expander("📷 ถ่ายรูปป้ายสเปก / ฉลากเบอร์ 5 ให้ AI เพิ่มเครื่องให้"):
        if not has_gemini_key():
            gemini_missing_notice()
        else:
            st.caption("ถ่ายป้ายข้างเครื่อง (มีค่า W, V, A) หรือฉลากประหยัดไฟเบอร์ 5 (มีหน่วยต่อปี) ให้ตัวเลขชัด")
            label_file = st.file_uploader("รูปป้าย", type=["jpg", "jpeg", "png", "webp", "heic"], key="label_file")
            label_cam = st.camera_input("หรือถ่ายจากกล้อง", key="label_cam") if st.toggle("ใช้กล้อง", key="label_cam_on") else None
            src = label_cam or label_file
            if st.button("ให้ AI อ่านป้าย", type="primary", disabled=src is None, key="read_label"):
                with st.spinner("AI กำลังอ่านป้าย..."):
                    try:
                        info = read_appliance_label(src.getvalue(), mime_for(getattr(src, "name", "photo.jpg"), src.type))
                        st.session_state.label_result = info
                    except Exception as err:
                        st.session_state.label_result = None
                        st.error(friendly_error(err) if not isinstance(err, ValueError) else str(err))
            info = st.session_state.get("label_result")
            if info:
                if info.get("is_appliance_label") is False:
                    st.warning("รูปนี้ดูไม่ใช่ป้ายเครื่องใช้ไฟฟ้า ลองถ่ายใหม่")
                else:
                    row, notes = label_to_row(info)
                    for n in notes + ([f"AI หมายเหตุ: {info['notes']}"] if info.get("notes") else []):
                        st.caption(f"• {n}")
                    row_df = st.data_editor(pd.DataFrame([row]), hide_index=True, key="label_row_editor", **stretch("data_editor"))
                    st.button("เพิ่มเข้าตาราง", type="primary", on_click=add_appliance_row,
                              args=(row_df.iloc[0].to_dict(),), key="add_label_row")

    table = st.data_editor(
        st.session_state.appliances, num_rows="dynamic", key=f"appliance_editor_{st.session_state.get('appliance_version', 0)}",
        hide_index=True,
        column_config={
            COL_NAME: st.column_config.TextColumn(COL_NAME),
            COL_WATT: st.column_config.NumberColumn(COL_WATT, min_value=0, format="%.0f"),
            COL_QTY: st.column_config.NumberColumn(COL_QTY, min_value=0, step=1, format="%d"),
            COL_HOURS: st.column_config.NumberColumn(COL_HOURS, min_value=0.0, max_value=24.0, format="%.1f"),
            COL_DAYS: st.column_config.NumberColumn(COL_DAYS, min_value=0, max_value=31, step=1, format="%d"),
            COL_DUTY: st.column_config.NumberColumn(COL_DUTY, min_value=0, max_value=100, step=5, format="%d",
                                                    help="แอร์/ตู้เย็นคอมเพรสเซอร์ไม่ได้ทำงานตลอดเวลา ใส่ 40–70%"),
        },
        **stretch("data_editor"),
    )
    st.session_state.appliance_table = table  # ให้หน้า "ถาม AI" เห็นตารางล่าสุด
    energy = appliance_energy(table)
    total_kwh = float(energy["kwh"].sum())

    df = current_df()
    if df is not None:
        last = add_bills(df, tariff).iloc[-1]
        rate = float(last["bill_best"] / last["kwh"]) if last["kwh"] > 0 else 0.0
    else:
        last, rate = None, (estimate_bill(total_kwh, tariff) / total_kwh if total_kwh > 0 else 0.0)
    energy["cost"] = energy["kwh"] * rate

    cols = st.columns(3)
    with cols[0]:
        with st.container(border=True):
            st.metric("รวมต่อเดือน", f"{total_kwh:,.0f} หน่วย")
    with cols[1]:
        with st.container(border=True):
            st.metric("ค่าไฟโดยประมาณ", baht(energy["cost"].sum()), delta=f"≈ ฿{rate:.2f} ต่อหน่วย", delta_color="off")
    with cols[2]:
        with st.container(border=True):
            if last is not None and last["kwh"] > 0:
                st.metric("เทียบกับยอดใช้จริงเดือนล่าสุด", f"{total_kwh / last['kwh'] * 100:.0f}%",
                          help="ถ้าต่ำกว่า 100% มาก แปลว่ายังใส่เครื่องไม่ครบ หรือค่าประมาณต่ำเกินไป ถ้าเกิน 100% แปลว่าประมาณสูงเกินไป")
            else:
                st.metric("เทียบกับยอดใช้จริงเดือนล่าสุด", "—", help="เพิ่มข้อมูลค่าไฟที่หน้า “ข้อมูลค่าไฟ” เพื่อเทียบ")

    if total_kwh > 0:
        ranked = energy.sort_values("kwh")
        fig = go.Figure(go.Bar(x=ranked["kwh"], y=ranked["name"], orientation="h", marker_color=COLORS["blue"],
                               hovertemplate="%{y}: %{x:,.1f} หน่วย<extra></extra>"))
        style_fig(fig, height=max(280, 44 * len(ranked) + 80))
        fig.update_xaxes(type="linear", title_text="หน่วยไฟต่อเดือน (kWh)", tickangle=0)
        fig.update_yaxes(type="category")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, config=PLOT_CONFIG, **stretch("plotly_chart"))
        st.caption("สูตร: วัตต์ ÷ 1,000 × จำนวน × ชั่วโมง/วัน × วัน/เดือน × %ทำงานจริง • ค่าไฟคิดตามค่าเฉลี่ยต่อหน่วยจากบิลล่าสุดของคุณ"
                   if last is not None else "สูตร: วัตต์ ÷ 1,000 × จำนวน × ชั่วโมง/วัน × วัน/เดือน × %ทำงานจริง")

# ---------------------------------------------------------------------------
# หน้า: ปฏิทินบิล
# ---------------------------------------------------------------------------
elif page == PAGE_CALENDAR:
    st.title("ปฏิทินบิลค่าไฟ")
    st.write("ระบบเพิ่มบิลให้อัตโนมัติเมื่อ AI อ่านวันครบกำหนดจากรูปบิล หรือเพิ่มเองด้านล่าง ติ๊ก “จ่ายแล้ว” เมื่อชำระ")
    cal = calendar_df()
    with st.form("add_bill_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            f_month = st.text_input("เดือนบิล", placeholder="ก.ย. 2569")
        with c2:
            f_amount = st.number_input("ยอดเงิน (บาท)", 0.0, 100000.0, 0.0, 10.0)
        with c3:
            f_due = st.date_input("ครบกำหนดชำระ", value=pd.Timestamp.today().date() + pd.Timedelta(days=7))
        if st.form_submit_button("เพิ่มบิล"):
            ts = parse_month(f_month)
            label = thai_month_label(ts) if not pd.isna(ts) else f_month.strip()
            if not label:
                st.error("กรุณาใส่เดือนบิล")
            else:
                st.session_state.bill_calendar = cal = add_to_calendar(cal, label, f_amount or None, f_due)
                st.success(f"เพิ่มบิล {label} แล้ว")
    if len(cal) == 0:
        st.info("ยังไม่มีบิลในปฏิทิน อัปโหลดรูปบิลที่หน้า “ข้อมูลค่าไฟ” หรือเพิ่มเองด้านบน")
    else:
        status = calendar_status(cal)
        overdue = status[status["สถานะ"].str.startswith("🔴")]
        soon = status[status["สถานะ"].str.startswith("🟠")]
        if len(overdue):
            st.error(f"มีบิลเลยกำหนด {len(overdue)} ใบ: " + ", ".join(overdue[CAL_MONTH]))
        if len(soon):
            st.warning("ใกล้ครบกำหนด: " + ", ".join(f"{r[CAL_MONTH]} (อีก {int(r['เหลือ (วัน)'])} วัน)" for _, r in soon.iterrows()))
        unpaid = status.loc[~status[CAL_PAID].astype(bool), CAL_AMOUNT].sum()
        st.metric("ยอดค้างชำระรวม", baht(unpaid))
        edited_cal = st.data_editor(
            status[[CAL_MONTH, CAL_AMOUNT, CAL_DUE, "เหลือ (วัน)", "สถานะ", CAL_PAID]], hide_index=True, num_rows="dynamic",
            key=f"cal_editor_{len(cal)}",
            column_config={
                CAL_AMOUNT: st.column_config.NumberColumn(CAL_AMOUNT, format="%.2f"),
                CAL_DUE: st.column_config.DateColumn(CAL_DUE, format="DD/MM/YYYY"),
                "เหลือ (วัน)": st.column_config.NumberColumn("เหลือ (วัน)", disabled=True),
                "สถานะ": st.column_config.TextColumn("สถานะ", disabled=True),
                CAL_PAID: st.column_config.CheckboxColumn(CAL_PAID),
            },
            **stretch("data_editor"),
        )
        def _norm_cal(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame[[CAL_MONTH, CAL_AMOUNT, CAL_DUE, CAL_PAID]].copy().reset_index(drop=True)
            out[CAL_MONTH] = out[CAL_MONTH].fillna("").astype(str)
            out[CAL_AMOUNT] = pd.to_numeric(out[CAL_AMOUNT], errors="coerce").astype("float64")
            out[CAL_DUE] = pd.to_datetime(out[CAL_DUE], errors="coerce").astype("datetime64[ns]")
            out[CAL_PAID] = out[CAL_PAID].fillna(False).astype(bool)
            return out

        new_cal = _norm_cal(edited_cal)
        if not new_cal.equals(_norm_cal(cal)):  # ผู้ใช้แก้ตาราง (ติ๊กจ่ายแล้ว/ลบแถว) → บันทึกแล้ววาดใหม่
            st.session_state.bill_calendar = new_cal
            st.rerun()
        st.download_button("📅 เพิ่มเตือนลงปฏิทินมือถือ (.ics)", to_ics(cal), "electricity_bills.ics", "text/calendar",
                           help="เปิดไฟล์บน iPhone หรือ Google Calendar จะมีเตือนก่อนครบกำหนด 2 วัน (เฉพาะบิลที่ยังไม่จ่าย)")
        st.caption("ปฏิทินเก็บในเบราว์เซอร์ระหว่างใช้งานเหมือนข้อมูลค่าไฟ ดาวน์โหลด .ics ไว้เพื่อให้เตือนได้แม้ปิดเว็บ")

# ---------------------------------------------------------------------------
# หน้า: วิธีทำงาน
# ---------------------------------------------------------------------------
else:
    st.title("วิธีทำงานและข้อจำกัด")
    st.subheader("ขั้นตอน")
    st.markdown(
        "1. **ข้อมูล** คุณอัปโหลดหรือกรอกหน่วยไฟรายเดือน ระบบตรวจรูปแบบเดือน (รองรับ พ.ศ.) เติมเดือนที่ขาด และเตือนค่าที่ผิดปกติ\n"
        "2. **สถิติ** ค่าเฉลี่ย สูงสุด ต่ำสุด ความผันผวน แนวโน้ม และเทียบเดือนเดียวกันปีก่อน\n"
        "3. **คาดการณ์** ระบบลองหลายวิธี (ค่าเฉลี่ย 3 เดือน, แนวโน้มเส้นตรง, เดือนเดียวกันปีก่อน, Random Forest) "
        "ทดสอบย้อนหลังกับข้อมูลของคุณ แล้วใช้วิธีที่คลาดเคลื่อนน้อยที่สุด\n"
        "4. **ค่าไฟ** คำนวณจากอัตราก้าวหน้าบ้านอยู่อาศัย + ค่าบริการ + Ft + VAT 7% หรือปรับตามบิลจริงถ้าคุณกรอกไว้\n"
        "5. **คำแนะนำ** ข้อความสรุปในหน้าคาดการณ์สร้างจากกฎที่ตั้งไว้ (ไม่เรียก AI ภายนอก จึงตอบเหมือนเดิมทุกครั้ง)\n"
        "6. **อ่านบิลด้วย AI** ส่งรูป/PDF ให้ Gemini อ่านเป็น JSON ตาม schema แล้วระบบตรวจซ้ำ เช่น หน่วยต้องตรงกับผลต่างเลขมิเตอร์ "
        "ยอดเงินต้องใกล้กับที่คำนวณได้ จากนั้นให้ผู้ใช้ยืนยันก่อนบันทึก\n"
        "7. **ถาม AI** Gemini ทำงานแบบ Agent เลือกเรียกเครื่องมือคำนวณ (ภาพรวม, เดือนที่ระบุ, คาดการณ์, คำนวณบิล, จำลองการประหยัด, "
        "เครื่องใช้ไฟฟ้า, What-if, ความคุ้มค่า) แล้วตอบจากผลลัพธ์จริง กดดู “ข้อมูลที่ AI ใช้” ใต้คำตอบได้\n"
        "8. **วิเคราะห์เชิงลึก** K-Means จัดกลุ่มเดือนเป็น ใช้ไฟต่ำ/ปกติ/สูง และ Random Forest + Permutation Importance "
        "บอกว่าปัจจัยไหนมีผลต่อการคาดการณ์ พร้อมทดสอบว่าใส่อุณหภูมิแล้วแม่นขึ้นไหม\n"
        "9. **จำลองและความคุ้มค่า** What-if จากสมมติฐานที่แสดงไว้ และคำนวณระยะคืนทุนการเปลี่ยนเครื่องใช้ไฟฟ้า\n"
        "10. **อากาศ 7 วัน** พยากรณ์จาก Open-Meteo ร่วมกับความสัมพันธ์อุณหภูมิ–หน่วยไฟของบ้านคุณ\n"
        "11. **อ่านป้ายเครื่องใช้ไฟฟ้า** AI อ่านวัตต์/แรงดัน/กระแส/หน่วยต่อปีจากรูป แล้วเพิ่มเข้าตาราง\n"
        "12. **ปฏิทินบิล** เก็บวันครบกำหนดจากบิล แสดงสถานะ และส่งออก .ics ให้ปฏิทินมือถือเตือน"
    )
    st.subheader("อัตราค่าไฟที่ใช้")
    st.write(f"อ้างอิง {TARIFF_AS_OF} (บาทต่อหน่วย ยังไม่รวมค่าบริการ Ft และ VAT)")
    st.table(pd.DataFrame({
        "ช่วงหน่วย": ["1–15", "16–25", "26–200", "201–400", "401 ขึ้นไป"],
        "ประเภท 1.1 (ไม่เกิน 150 หน่วย/เดือน)": [f"{TIERS_1_1[0][1]:.4f}", f"{TIERS_1_1[1][1]:.4f}", f"{TIERS_1_1[2][1]:.4f}",
                                                 f"{TIERS_1_1[3][1]:.4f}", f"{TIERS_1_1[4][1]:.4f}"],
        "ประเภท 1.2 (เกิน 150 หน่วย/เดือน)": ["3.0000", "3.0000", "3.0000", f"{TIERS_1_2[1][1]:.4f}", f"{TIERS_1_2[2][1]:.4f}"],
    }).set_index("ช่วงหน่วย"))
    example = bill_breakdown(350, tariff)
    st.caption(f"ตัวอย่างใช้ 350 หน่วย: ค่าพลังงาน ฿{example['energy']:,.0f} + Ft ฿{example['ft']:,.0f} + ค่าบริการ ฿{example['service']:,.0f} "
               f"+ VAT ฿{example['vat']:,.0f} = ฿{example['total']:,.0f} (ประมาณ) ค่า Ft และค่าบริการเปลี่ยนได้ในแถบตั้งค่าด้านซ้าย "
               "อัตรานี้เป็นของ กฟน. (กรุงเทพฯ นนทบุรี สมุทรปราการ) พื้นที่ กฟภ. อาจต่างกัน")
    st.subheader("ข้อจำกัด")
    st.markdown(
        "- ผลคาดการณ์เป็นการประมาณจากข้อมูลที่คุณให้ ข้อมูลน้อยกว่า 12 เดือนจะไม่ครอบคลุมฤดูกาล\n"
        "- ยอดบิลจริงอาจต่างเล็กน้อยจากที่คำนวณ เช่น รอบจดมิเตอร์ ส่วนลดชั่วคราว หรือค่าบริการที่เปลี่ยน\n"
        "- ข้อมูลเก็บชั่วคราวในเบราว์เซอร์ ไม่ได้บันทึกบนเซิร์ฟเวอร์ ดาวน์โหลดไฟล์ไว้ก่อนออกจากระบบ\n"
        "- อุณหภูมิรายเดือนมาจาก Open-Meteo ตามพิกัดที่ตั้งค่าไว้ ไม่ใช่อุณหภูมิในบ้าน\n"
        "- AI อ่านบิลผิดได้ โดยเฉพาะรูปเบลอหรือสะท้อนแสง ตรวจตัวเลขก่อนบันทึกทุกครั้ง\n"
        "- คำตอบของ AI อาจคลาดเคลื่อน ตัวเลขที่อ้างอิงข้อมูลของคุณตรวจได้จาก “ข้อมูลที่ AI ใช้”"
    )
    st.subheader("โครงสร้างไฟล์")
    st.code("""electricity-ai-analytics/
├── app.py                  หน้าแอปหลัก
├── theme.py                ธีม CSS และสไตล์กราฟ
├── firebase_auth.py        ล็อกอิน/สมัครสมาชิก
├── data_service.py         นำเข้าและทำความสะอาดข้อมูลจริง
├── electricity_service.py  อัตราค่าไฟ เครื่องใช้ไฟฟ้า สภาพอากาศ
├── stats_service.py        สถิติและโมเดลคาดการณ์
├── ai_service.py           ข้อความแนะนำ (กฎ)
├── llm_service.py          เชื่อม Gemini: แชท Agent + อ่านบิล
├── chat_tools.py           เครื่องมือที่ AI เรียกใช้
├── bill_service.py         ตรวจค่าที่อ่านจากบิล + รวมข้อมูล + ปฏิทินบิล
├── insights_service.py     K-Means และ Explainable AI
├── planner_service.py      What-if, ความคุ้มค่า, อากาศ 7 วัน, ป้ายเครื่องใช้ไฟฟ้า
├── tests/                  ทดสอบอัตโนมัติ (pytest)
├── sample_data/            ไฟล์ CSV ตัวอย่าง
└── .streamlit/config.toml  สีและฟอนต์หลัก""")

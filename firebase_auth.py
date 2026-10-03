"""ล็อกอิน/สมัครสมาชิกด้วย Firebase Authentication (Email + Password) ผ่าน REST API"""
from __future__ import annotations

import os
import re

import requests
import streamlit as st

from theme import stretch

try:  # อ่านค่าจากไฟล์ .env ถ้ามี (ใช้ตอนรันในเครื่อง)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

BASE_URL = "https://identitytoolkit.googleapis.com/v1/accounts"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

ERROR_MESSAGES = {
    "EMAIL_EXISTS": "อีเมลนี้สมัครไว้แล้ว ลองเข้าสู่ระบบแทน",
    "INVALID_LOGIN_CREDENTIALS": "อีเมลหรือรหัสผ่านไม่ถูกต้อง",
    "INVALID_PASSWORD": "อีเมลหรือรหัสผ่านไม่ถูกต้อง",
    "EMAIL_NOT_FOUND": "ไม่พบอีเมลนี้ในระบบ",
    "INVALID_EMAIL": "รูปแบบอีเมลไม่ถูกต้อง",
    "WEAK_PASSWORD": "รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "ลองหลายครั้งเกินไป กรุณารอสักครู่แล้วลองใหม่",
    "USER_DISABLED": "บัญชีนี้ถูกระงับการใช้งาน",
    "OPERATION_NOT_ALLOWED": "ยังไม่ได้เปิด Email/Password ใน Firebase Authentication",
}


def _api_key() -> str:
    """อ่านจาก Streamlit Secrets ก่อน แล้วค่อย environment/.env"""
    try:
        key = st.secrets["FIREBASE_API_KEY"]
    except Exception:
        key = None
    return str(key or os.getenv("FIREBASE_API_KEY", "")).strip()


def has_api_key() -> bool:
    key = _api_key()
    return bool(key) and not key.startswith("YOUR_")


def guest_allowed() -> bool:
    return os.getenv("ALLOW_GUEST", "").strip().lower() in {"1", "true", "yes"}


def _call(endpoint: str, payload: dict):
    if not has_api_key():
        st.error("ยังไม่ได้ตั้งค่า FIREBASE_API_KEY (ดูวิธีตั้งค่าใน README)")
        return None
    try:
        r = requests.post(f"{BASE_URL}:{endpoint}", params={"key": _api_key()}, json=payload, timeout=15)
    except requests.RequestException:
        st.error("เชื่อมต่อ Firebase ไม่ได้ ตรวจสอบอินเทอร์เน็ตแล้วลองใหม่")
        return None
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.ok:
        return data
    code = str(data.get("error", {}).get("message", ""))
    for prefix, message in ERROR_MESSAGES.items():
        if code.startswith(prefix):  # Firebase ต่อท้ายรายละเอียด เช่น "WEAK_PASSWORD : ..." จึงเช็กด้วยคำนำหน้า
            st.error(message)
            return None
    st.error(f"ดำเนินการไม่สำเร็จ ({code or r.status_code})")
    return None


def _valid_email(email: str) -> bool:
    if not EMAIL_RE.match(email):
        st.error("รูปแบบอีเมลไม่ถูกต้อง")
        return False
    return True


def _start_session(email: str) -> None:
    st.session_state.logged_in = True
    st.session_state.user_email = email


def login_form() -> None:
    with st.form("login"):
        email = st.text_input("อีเมล")
        password = st.text_input("รหัสผ่าน", type="password")
        if st.form_submit_button("เข้าสู่ระบบ", type="primary", **stretch("form_submit_button")):
            email = email.strip()
            if _valid_email(email) and password:
                data = _call("signInWithPassword", {"email": email, "password": password, "returnSecureToken": True})
                if data:
                    _start_session(data.get("email", email))
                    st.rerun()
            elif not password:
                st.error("กรุณากรอกรหัสผ่าน")


def signup_form() -> None:
    with st.form("signup"):
        email = st.text_input("อีเมล", key="signup_email")
        password = st.text_input("รหัสผ่าน (อย่างน้อย 6 ตัวอักษร)", type="password", key="signup_password")
        confirm = st.text_input("ยืนยันรหัสผ่าน", type="password", key="signup_confirm")
        if st.form_submit_button("สมัครสมาชิก", type="primary", **stretch("form_submit_button")):
            email = email.strip()
            if not _valid_email(email):
                return
            if password != confirm:
                st.error("รหัสผ่านสองช่องไม่ตรงกัน")
                return
            data = _call("signUp", {"email": email, "password": password, "returnSecureToken": True})
            if data:
                _start_session(data.get("email", email))
                st.rerun()


def reset_form() -> None:
    with st.form("reset"):
        email = st.text_input("อีเมลที่ใช้สมัคร", key="reset_email")
        if st.form_submit_button("ส่งอีเมลตั้งรหัสผ่านใหม่", **stretch("form_submit_button")):
            email = email.strip()
            if _valid_email(email) and _call("sendOobCode", {"requestType": "PASSWORD_RESET", "email": email}):
                st.success("ส่งลิงก์ตั้งรหัสผ่านใหม่ไปที่อีเมลแล้ว (ตรวจสอบกล่องจดหมายขยะด้วย)")


def login_as_guest() -> None:
    _start_session("ผู้ใช้ทดลอง")


def is_logged_in() -> bool:
    return bool(st.session_state.get("logged_in", False))


def logout() -> None:
    """ใช้เป็น on_click callback (Streamlit รันซ้ำเองหลัง callback จึงไม่ต้อง st.rerun)"""
    for key in ("logged_in", "user_email", "data", "data_source", "temp_df", "manual_df", "appliances"):
        st.session_state.pop(key, None)

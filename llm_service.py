"""เชื่อมต่อ Gemini: (1) แชทถาม-ตอบแบบ Agent เรียก Tools  (2) อ่านบิลค่าไฟจากรูป/PDF

ตั้งค่า GEMINI_API_KEY (และ GEMINI_MODEL ถ้าต้องการ) ใน .env หรือ Streamlit Secrets
"""
from __future__ import annotations

import json
import os
import re
import time

from chat_tools import TOOL_DECLARATIONS, ChatContext, execute_tool

DEFAULT_MODEL = "gemini-3.6-flash"
MAX_STEPS = 6
MAX_HISTORY_TURNS = 10  # ส่งประวัติแชทย้อนหลังกี่รอบ (กันข้อความยาวเกินและเปลือง token)

BILL_MIME = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp",
    "heic": "image/heic", "heif": "image/heif", "pdf": "application/pdf",
}
BILL_TYPES = list(BILL_MIME)
MAX_BILL_BYTES = 15 * 1024 * 1024


def _secret(name: str, default: str = "") -> str:
    try:
        import streamlit as st
        value = st.secrets[name]
    except Exception:
        value = None
    return str(value or os.getenv(name, default)).strip()


def api_key() -> str:
    return _secret("GEMINI_API_KEY")


def model_name() -> str:
    return _secret("GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def has_api_key() -> bool:
    key = api_key()
    return bool(key) and not key.startswith("YOUR_")


def _client():
    if not has_api_key():
        raise RuntimeError("ยังไม่ได้ตั้งค่า GEMINI_API_KEY (ดูวิธีตั้งค่าใน README)")
    from google import genai
    return genai.Client(api_key=api_key())


def _generate(client, **kwargs):
    """เรียก Gemini พร้อมลองใหม่เมื่อเซิร์ฟเวอร์ไม่ว่าง (503/429)"""
    last = None
    for delay in (0, 2, 5):
        if delay:
            time.sleep(delay)
        try:
            return client.models.generate_content(model=model_name(), **kwargs)
        except Exception as err:
            last = err
            text = str(err)
            if not any(code in text for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")):
                raise
    raise last


def friendly_error(err: Exception) -> str:
    text = str(err)
    if "API_KEY_INVALID" in text or "API key not valid" in text:
        return "GEMINI_API_KEY ไม่ถูกต้อง ตรวจสอบ key อีกครั้ง"
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "ใช้ Gemini เกินโควต้าชั่วคราว รอสักครู่แล้วลองใหม่"
    if "503" in text or "UNAVAILABLE" in text:
        return "เซิร์ฟเวอร์ Gemini ไม่ว่าง ลองใหม่อีกครั้ง"
    if "NOT_FOUND" in text and "model" in text.lower():
        return f"ไม่พบโมเดล {model_name()} ลองตั้ง GEMINI_MODEL เป็นรุ่นอื่น"
    return f"เรียก AI ไม่สำเร็จ: {text[:300]}"


# ---------------------------------------------------------------------------
# 1) แชทถาม-ตอบ (Agent + Tools)
# ---------------------------------------------------------------------------
SYSTEM_CHAT = """คุณคือ "ผู้ช่วยค่าไฟบ้าน" ในเว็บ Home Electricity AI ตอบเป็นภาษาไทย สุภาพ กระชับ เข้าใจง่าย

กติกา:
- ตัวเลขทุกตัวที่เกี่ยวกับข้อมูลของผู้ใช้ (หน่วยไฟ ค่าไฟ คาดการณ์ เครื่องใช้ไฟฟ้า) ต้องมาจากผลของ Tool เท่านั้น ห้ามเดา
- ถ้า Tool ตอบ error ให้บอกผู้ใช้ตรงๆ และแนะนำสิ่งที่ต้องทำ
- ค่าไฟที่คำนวณเป็นค่าประมาณ ไม่ใช่ยอดบิลทางการ ระบุเมื่อเกี่ยวข้อง
- คาดการณ์ต้องบอกช่วงและความน่าเชื่อถือด้วย ไม่รับประกันผล
- คำถามความรู้ทั่วไปเรื่องไฟฟ้า/การประหยัดไฟ ตอบจากความรู้ได้ แต่ถ้าต้องการกำลังไฟของเครื่องเฉพาะรุ่น ให้บอกว่าเป็นค่าประมาณและแนะนำดูป้ายเครื่อง
- คำถามนอกเรื่องไฟฟ้า/ค่าไฟ/พลังงานในบ้าน ให้ปฏิเสธสุภาพและบอกว่าช่วยเรื่องอะไรได้
- ตอบสั้น ใช้ bullet เมื่อมีหลายประเด็น ปิดท้ายด้วยคำแนะนำที่ทำได้จริง 1–2 ข้อเมื่อเหมาะสม
"""


def run_chat(question: str, history: list[dict], ctx: ChatContext) -> dict:
    """คืน {"answer": str, "trace": [ {tool, arguments, result} ]}

    history: [{"role": "user"|"assistant", "content": str}] ไม่รวมคำถามปัจจุบัน
    """
    from google.genai import types

    client = _client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_CHAT,
        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
    )
    contents = []
    for msg in history[-MAX_HISTORY_TURNS * 2:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

    trace: list[dict] = []
    seen: set = set()
    for _ in range(MAX_STEPS):
        resp = _generate(client, contents=contents, config=config)
        calls = resp.function_calls or []
        if not calls:
            return {"answer": (resp.text or "").strip() or "ขออภัย ตอบไม่ได้ในตอนนี้ ลองถามใหม่อีกครั้ง", "trace": trace}
        contents.append(resp.candidates[0].content)
        parts = []
        for call in calls:
            args = dict(call.args or {})
            sig = (call.name, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str))
            if sig in seen:
                result = {"error": "เรียก Tool เดิมด้วยค่าเดิมซ้ำ ใช้ผลก่อนหน้าได้เลย"}
            else:
                seen.add(sig)
                result = execute_tool(ctx, call.name, args)
            trace.append({"tool": call.name, "arguments": args, "result": result})
            parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))
    return {"answer": "คำถามนี้ต้องคำนวณหลายขั้นเกินไป ลองแยกถามทีละเรื่อง", "trace": trace}


# ---------------------------------------------------------------------------
# 2) อ่านบิลค่าไฟจากรูปภาพ / PDF (Gemini Vision + JSON schema)
# ---------------------------------------------------------------------------
BILL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_electricity_bill": {"type": "BOOLEAN", "description": "เป็นใบแจ้งค่าไฟฟ้าหรือใบเสร็จค่าไฟหรือไม่"},
        "provider": {"type": "STRING", "description": "MEA (กฟน.) / PEA (กฟภ.) / อื่นๆ", "nullable": True},
        "billing_month": {"type": "STRING", "description": "เดือน/ปีของค่าไฟ ตามที่พิมพ์บนบิล เช่น ก.ย. 2569", "nullable": True},
        "billing_month_iso": {"type": "STRING", "description": "เดือนเดียวกันในรูปแบบ YYYY-MM เป็นปี ค.ศ.", "nullable": True},
        "reading_date": {"type": "STRING", "description": "วันที่จดเลขอ่าน YYYY-MM-DD ปี ค.ศ.", "nullable": True},
        "kwh": {"type": "NUMBER", "description": "จำนวนหน่วยที่ใช้ (kWh) ของรอบนี้", "nullable": True},
        "meter_previous": {"type": "NUMBER", "description": "เลขอ่านครั้งก่อน", "nullable": True},
        "meter_current": {"type": "NUMBER", "description": "เลขอ่านครั้งนี้", "nullable": True},
        "energy_charge": {"type": "NUMBER", "description": "ค่าพลังงานไฟฟ้า (บาท)", "nullable": True},
        "ft_rate_satang": {"type": "NUMBER", "description": "อัตรา Ft สตางค์ต่อหน่วย", "nullable": True},
        "ft_amount": {"type": "NUMBER", "description": "ค่า Ft รวม (บาท)", "nullable": True},
        "service_charge": {"type": "NUMBER", "description": "ค่าบริการรายเดือน (บาท)", "nullable": True},
        "vat": {"type": "NUMBER", "description": "ภาษีมูลค่าเพิ่ม (บาท)", "nullable": True},
        "total_amount": {"type": "NUMBER", "description": "ยอดค่าไฟฟ้ารวมของเดือนนี้ (บาท) ไม่รวมยอดค้างชำระ", "nullable": True},
        "tariff_type": {"type": "STRING", "description": "ประเภทผู้ใช้ไฟ เช่น 1.1 หรือ 1.2", "nullable": True},
        "due_date": {"type": "STRING", "description": "วันครบกำหนดชำระ YYYY-MM-DD ปี ค.ศ.", "nullable": True},
        "notes": {"type": "STRING", "description": "สิ่งที่อ่านไม่ชัดหรือไม่แน่ใจ (ภาษาไทยสั้นๆ)", "nullable": True},
    },
    "required": ["is_electricity_bill"],
}

PROMPT_BILL = """อ่านใบแจ้งค่าไฟฟ้า/ใบเสร็จค่าไฟฟ้าของไทยในไฟล์นี้ แล้วกรอกข้อมูลตาม schema
- อ่านเฉพาะตัวเลขที่เห็นจริงบนบิล ถ้าไม่เห็นหรือไม่แน่ใจให้เป็น null ห้ามเดา
- ปี พ.ศ. ให้แปลงเป็น ค.ศ. (ลบ 543) ใน billing_month_iso, reading_date และ due_date
- total_amount คือค่าไฟของเดือนนี้ ไม่รวมยอดค้างชำระหรือค่าปรับ
- ห้ามดึงข้อมูลส่วนตัว เช่น ชื่อ ที่อยู่ เลขผู้ใช้ไฟฟ้า เลขบัญชี
- ถ้าไม่ใช่บิลค่าไฟ ให้ is_electricity_bill = false"""


def _to_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d[\d,]*\.?\d*", str(v))
    return float(m.group().replace(",", "")) if m else None


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("AI ไม่ได้ส่งข้อมูลกลับในรูปแบบที่อ่านได้")
        return json.loads(m.group())


def mime_for(filename: str, fallback: str = "") -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return BILL_MIME.get(ext) or fallback or "image/jpeg"


def read_bill(data: bytes, mime_type: str) -> dict:
    """ส่งรูป/PDF ให้ Gemini อ่าน คืน dict ตาม BILL_SCHEMA (ตัวเลขแปลงเป็น float แล้ว)"""
    from google.genai import types

    if len(data) > MAX_BILL_BYTES:
        raise ValueError("ไฟล์ใหญ่เกิน 15 MB ลองถ่ายใหม่หรือย่อรูปก่อน")
    resp = _generate(
        _client(),
        contents=[types.Part.from_bytes(data=data, mime_type=mime_type), PROMPT_BILL],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=BILL_SCHEMA, temperature=0,
        ),
    )
    out = _parse_json(resp.text)
    for key in ("kwh", "meter_previous", "meter_current", "energy_charge", "ft_rate_satang", "ft_amount",
                "service_charge", "vat", "total_amount"):
        out[key] = _to_float(out.get(key))
    return out


# ---------------------------------------------------------------------------
# 3) อ่านป้ายสเปก / ฉลากประหยัดไฟเบอร์ 5 ของเครื่องใช้ไฟฟ้า
# ---------------------------------------------------------------------------
LABEL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_appliance_label": {"type": "BOOLEAN", "description": "เป็นป้ายสเปกหรือฉลากพลังงานของเครื่องใช้ไฟฟ้าหรือไม่"},
        "appliance_type": {"type": "STRING", "description": "ชนิดเครื่องเป็นภาษาไทยสั้นๆ เช่น แอร์ ตู้เย็น ทีวี พัดลม ไมโครเวฟ", "nullable": True},
        "brand": {"type": "STRING", "description": "ยี่ห้อ (ถ้าเห็น)", "nullable": True},
        "power_watt": {"type": "NUMBER", "description": "กำลังไฟฟ้า (วัตต์) ถ้าเป็น kW ให้แปลงเป็นวัตต์ ถ้าแอร์มีทั้ง cooling/heating ใช้ cooling", "nullable": True},
        "voltage_v": {"type": "NUMBER", "description": "แรงดันไฟฟ้า (V)", "nullable": True},
        "current_a": {"type": "NUMBER", "description": "กระแสไฟฟ้า (A)", "nullable": True},
        "cooling_btu": {"type": "NUMBER", "description": "ขนาดความเย็น BTU/h (แอร์)", "nullable": True},
        "annual_kwh": {"type": "NUMBER", "description": "พลังงานไฟฟ้าที่ใช้ต่อปี (kWh/ปี) จากฉลากเบอร์ 5", "nullable": True},
        "energy_label_no5": {"type": "BOOLEAN", "description": "มีฉลากประหยัดไฟฟ้าเบอร์ 5 หรือไม่", "nullable": True},
        "label_stars": {"type": "INTEGER", "description": "จำนวนดาวบนฉลากเบอร์ 5 (ถ้ามี)", "nullable": True},
        "notes": {"type": "STRING", "description": "สิ่งที่อ่านไม่ชัด (ภาษาไทยสั้นๆ)", "nullable": True},
    },
    "required": ["is_appliance_label"],
}

PROMPT_LABEL = """อ่านป้ายสเปก (nameplate) หรือฉลากประหยัดไฟฟ้าเบอร์ 5 ของเครื่องใช้ไฟฟ้าในรูปนี้ แล้วกรอกตาม schema
- อ่านเฉพาะตัวเลขที่เห็นจริง ถ้าไม่เห็นให้เป็น null ห้ามเดาจากความรู้ทั่วไป
- ถ้าไม่ใช่ป้ายเครื่องใช้ไฟฟ้า ให้ is_appliance_label = false"""


def read_appliance_label(data: bytes, mime_type: str) -> dict:
    from google.genai import types

    if len(data) > MAX_BILL_BYTES:
        raise ValueError("ไฟล์ใหญ่เกิน 15 MB ลองถ่ายใหม่หรือย่อรูปก่อน")
    resp = _generate(
        _client(),
        contents=[types.Part.from_bytes(data=data, mime_type=mime_type), PROMPT_LABEL],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=LABEL_SCHEMA, temperature=0),
    )
    out = _parse_json(resp.text)
    for key in ("power_watt", "voltage_v", "current_a", "cooling_btu", "annual_kwh"):
        out[key] = _to_float(out.get(key))
    return out

"""ธีมและตัวช่วยจัดหน้า

หลักการ: พื้นหลังขาว ตัวอักษรเข้มคอนทราสต์สูง ใช้สีน้ำเงินแทน "หน่วยไฟ" และสีเหลืองอำพันแทน "เงิน/คาดการณ์"
ใช้ component มาตรฐานของ Streamlit เป็นหลัก CSS ปรับเฉพาะขนาดตัวอักษร ระยะห่าง และเส้นขอบ
"""
from __future__ import annotations

import inspect

import streamlit as st

COLORS = {
    "ink": "#14213D",
    "muted": "#4B5A6B",
    "line": "#D9E0E8",
    "blue": "#1D4ED8",
    "blue_light": "#9DB7F5",
    "amber": "#D97706",
    "amber_light": "#F6C77E",
    "green": "#047857",
    "red": "#B91C1C",
}

FONT_STACK = "'IBM Plex Sans Thai', 'Noto Sans Thai', 'Sarabun', sans-serif"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600&display=swap');

html, body, .stApp, h1, h2, h3, h4, h5, p, li, label, button, input, textarea,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"],
[data-testid="stTable"] {{
    font-family: {FONT_STACK};
}}

.block-container {{ max-width: 1100px; padding-top: 2.2rem; padding-bottom: 4rem; }}
p, li, label {{ font-size: 1.02rem; line-height: 1.65; }}
h1 {{ font-size: 1.9rem; font-weight: 600; color: {COLORS['ink']}; }}
h2, h3 {{ font-weight: 600; color: {COLORS['ink']}; }}
h3 {{ font-size: 1.2rem; margin-top: .4rem; }}
[data-testid="stCaptionContainer"] {{ color: {COLORS['muted']}; font-size: .95rem; }}

/* การ์ดตัวเลข */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {COLORS['line']} !important; border-radius: 12px; }}
[data-testid="stMetricLabel"] {{ color: {COLORS['muted']}; font-size: .98rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.85rem; font-weight: 600; font-variant-numeric: tabular-nums; color: {COLORS['ink']}; }}
[data-testid="stMetricDelta"] {{ font-size: .95rem; }}

/* แท็บและเมนูให้อ่านง่าย */
button[role="tab"] {{ font-size: 1.02rem; padding: .6rem 1rem; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: .35rem 0; }}
[data-testid="stSidebar"] [role="radiogroup"] p {{ font-size: 1.05rem; }}

/* ตารางสรุป */
[data-testid="stTable"] td, [data-testid="stTable"] th {{ font-size: 1rem; padding: .55rem .8rem; }}

/* ป้ายสถานะ */
.chip {{ display: inline-block; padding: 2px 12px; border-radius: 999px; font-size: .9rem; font-weight: 500; }}
.chip-good {{ background: #D1FAE5; color: #065F46; }}
.chip-mid {{ background: #FEF3C7; color: #92400E; }}
.chip-low {{ background: #FEE2E2; color: #991B1B; }}

@media (max-width: 640px) {{
    .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    h1 {{ font-size: 1.5rem; }}
    [data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
}}
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def stretch(func_name: str) -> dict:
    """ขยายเต็มความกว้างให้ได้ทั้ง Streamlit รุ่นใหม่ (width="stretch") และรุ่นเก่า (use_container_width)"""
    try:
        params = inspect.signature(getattr(st, func_name)).parameters
        if "width" in params:
            return {"width": "stretch"}
    except (TypeError, ValueError, AttributeError):
        pass
    return {"use_container_width": True}


def reliability_chip(text: str) -> str:
    cls = "chip-good" if text.startswith("ดี") else "chip-mid" if text.startswith("ปานกลาง") else "chip-low"
    return f'<span class="chip {cls}">{text}</span>'


def style_fig(fig, height: int = 380):
    """รูปแบบกราฟ Plotly ให้เหมือนกันทุกกราฟ: ตัวอักษรใหญ่ เส้นกริดจาง ไม่มีแถบเครื่องมือ"""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        font=dict(family=FONT_STACK, size=14, color=COLORS["ink"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(type="category", showgrid=False, tickangle=-45)
    fig.update_yaxes(gridcolor=COLORS["line"], zeroline=False)
    return fig


PLOT_CONFIG = {"displayModeBar": False}

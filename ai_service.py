"""สร้างข้อความแนะนำภาษาไทยจากตัวเลขที่คำนวณได้ (ใช้กฎ ไม่ได้เรียก AI API ภายนอก)"""
from __future__ import annotations

SAVING_TIPS = [
    ("แอร์", "ตั้งอุณหภูมิ 26–27°C และล้างแผ่นกรองทุก 1–2 เดือน แอร์มักเป็นเครื่องที่กินไฟมากที่สุดในบ้าน"),
    ("ตู้เย็น", "อย่าเปิดค้างนาน และเว้นช่องว่างด้านหลังให้ระบายความร้อน"),
    ("ไฟสแตนด์บาย", "ถอดปลั๊กอุปกรณ์ที่ไม่ได้ใช้ หรือใช้ปลั๊กพ่วงที่มีสวิตช์"),
    ("เครื่องทำน้ำอุ่น", "เปิดเฉพาะตอนอาบน้ำ ตั้งอุณหภูมิไม่สูงเกินจำเป็น"),
    ("หลอดไฟ", "เปลี่ยนเป็น LED และปิดไฟเมื่อไม่มีคนอยู่ในห้อง"),
]


def generate_insights(ctx: dict) -> list[dict]:
    """คืนรายการ {level, text}; level = success | info | warning

    ctx: latest_kwh, latest_bill, latest_label, pred_kwh, pred_bill, mean_kwh, slope,
         n_months, reliability, temp_effect (dict|None), saving_10pct (float|None)
    """
    out: list[dict] = []
    latest, pred = ctx["latest_kwh"], ctx["pred_kwh"]

    if latest > 0:
        change = (pred - latest) / latest * 100
        money = f"ค่าไฟประมาณ ฿{ctx['pred_bill']:,.0f}"
        if change > 8:
            out.append({"level": "warning", "text": f"เดือนหน้าคาดว่าใช้ไฟประมาณ {pred:,.0f} หน่วย ({money}) เพิ่มขึ้นราว {change:.0f}% จากเดือนล่าสุด"})
        elif change < -8:
            out.append({"level": "success", "text": f"เดือนหน้าคาดว่าใช้ไฟประมาณ {pred:,.0f} หน่วย ({money}) ลดลงราว {abs(change):.0f}% จากเดือนล่าสุด"})
        else:
            out.append({"level": "info", "text": f"เดือนหน้าคาดว่าใช้ไฟประมาณ {pred:,.0f} หน่วย ({money}) ใกล้เคียงเดือนล่าสุด"})

    mean = ctx["mean_kwh"]
    if mean > 0:
        dev = (latest - mean) / mean * 100
        if dev >= 10:
            out.append({"level": "warning", "text": f"{ctx['latest_label']} ใช้ไฟสูงกว่าค่าเฉลี่ยของคุณ {dev:.0f}% ลองดูว่ามีเครื่องที่เปิดนานกว่าปกติหรือไม่"})
        elif dev <= -10:
            out.append({"level": "success", "text": f"{ctx['latest_label']} ใช้ไฟต่ำกว่าค่าเฉลี่ยของคุณ {abs(dev):.0f}%"})

    slope = ctx["slope"]
    if ctx["n_months"] >= 6 and mean > 0 and abs(slope) / mean > 0.015:
        direction = "เพิ่มขึ้น" if slope > 0 else "ลดลง"
        out.append({"level": "info", "text": f"แนวโน้มระยะยาว {direction}ประมาณ {abs(slope):.1f} หน่วยต่อเดือน"})

    eff = ctx.get("temp_effect")
    if eff and eff["r"] >= 0.5 and eff["slope"] > 0:
        out.append({"level": "info", "text": f"ในข้อมูลของคุณ เดือนที่อากาศร้อนขึ้น 1°C ใช้ไฟเพิ่มราว {eff['slope']:.1f} หน่วย (ความสัมพันธ์ r = {eff['r']:.2f} จาก {eff['n']} เดือน) น่าจะมาจากการใช้แอร์"})

    saving = ctx.get("saving_10pct")
    if saving and saving >= 1:
        out.append({"level": "info", "text": f"ถ้าลดการใช้ไฟลง 10% ({latest * 0.1:,.0f} หน่วย) จะลดค่าไฟได้ประมาณ ฿{saving:,.0f} ต่อเดือน"})

    if ctx["n_months"] < 12:
        out.append({"level": "warning", "text": f"ตอนนี้มีข้อมูล {ctx['n_months']} เดือน ยังไม่ครอบคลุมฤดูกาลครบปี ผลคาดการณ์จึงเป็นเพียงแนวทาง ยิ่งมีข้อมูลย้อนหลัง 12–24 เดือนยิ่งแม่นขึ้น"})
    return out

import os
import re
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

def read_orders() -> Dict[str, Any]:
    try:
        with open(Path(__file__).parent.parent / "data" / "orders.json", "r", encoding="utf-8") as f:
            orders = json.load(f)
            return {str(o.get("order_id")): o for o in orders}
    except Exception:
        return {}

ORDERS: Dict[str, Any] = read_orders()


def _format_order_response(order: dict) -> str:
    status = order.get("status", "unknown")
    order_id = order.get("order_id")
    tracking = order.get("tracking_number")
    parts = [f"Заказ {order_id}: статус — {status}."]
    if tracking:
        parts.append(f"Трек-номер: {tracking}.")
    if order.get("delivery_method"):
        parts.append(f"Способ доставки: {order['delivery_method']}.")
    if order.get("cancel_reason"):
        parts.append(f"Причина отмены: {order['cancel_reason']}.")
    return " ".join(parts)


def handle_order_query(prompt: str) -> Optional[str]:
    # Ищем номер заказа в тексте
    m = re.search(r"Заказ\s*(\d+)", prompt, re.IGNORECASE)
    if not m:
        m = re.search(r"(?:order|заказ)\s*(\d+)", prompt, re.IGNORECASE)
    if m:
        oid = m.group(1)
        order = ORDERS.get(oid)
        if order:
            return _format_order_response(order)
        else:
            return f"Заказ {oid} не найден. Проверьте номер и попробуйте снова."
    return None



BASE = Path(__file__).parent.parent


def read_style_config():
    with open(BASE / "data" / "style_guide.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    current = os.getenv("PROMPT_VERSION", "v1")
    versions = data.get("versions", {})
    style = versions.get(current, {})
    if not style or current not in versions:
        raise ValueError(f"No style configuration found for PROMPT_VERSION={current}")
    return style

def read_brand_config():
    with open(BASE / "data" / "style_guide.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    brand = data.get("brand", {})
    return brand

STYLE = read_style_config()
BRAND = read_brand_config()


def ask(prompt: str) -> dict:
    local = handle_order_query(prompt)
    if local is not None:
        return {
            "answer": local,
            "actions": ["Проверьте страницу заказа в личном кабинете"],
            "tone": STYLE.get("tone", {})
        }

    llm = ChatOpenAI(temperature=0.7)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"Ты — ассистент, который пишет в стиле бренда {BRAND}. В ответе использует не более {STYLE["tone"]['sentences_max']} предложений."
                   f"Тон: {STYLE['tone']['persona']}. Избегай: {', '.join(STYLE['tone']['avoid'])}. "
                   f"Обязательно: {', '.join(STYLE['tone']['must_include'])}."
                   f"В случае отсутствия данных по вопросу используй: {STYLE["fallback"]["no_data"]}."
                   f"В случае неопределённости используй: {STYLE['fallback']['unclear_query']}."
                   f"В случае ошибки используй: {STYLE['fallback']['error']}."),
        ("human", "{prompt}")
    ])

    chain = prompt_template | llm
    response = chain.invoke({"prompt": prompt})
    # Normalize answer to plain text (use .content when available)
    answer_text = getattr(response, "content", str(response))
    if hasattr(answer_text, "strip"):
        answer_text = answer_text.strip()

    return {
        "answer": answer_text,
        "actions": STYLE.get("actions", []),
        "tone": STYLE.get("tone", {})
    }

def parse_chat_logs(path: Path) -> list[dict]:
    pairs = []
    last_user = None

    user_re = re.compile(r"\[INFO\].*User:\s*(.*)")
    bot_re = re.compile(r"\[INFO\].*Bot:\s*(.*)")

    for line in path.read_text(encoding="utf-8").splitlines():
        u = user_re.search(line)
        if u:
            last_user = u.group(1).strip()
            continue

        b = bot_re.search(line)
        if b and last_user:
            pairs.append({
                "prompt": last_user,
                "answer": b.group(1).strip()
            })
            last_user = None

    return pairs
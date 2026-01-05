import os, json, pathlib, re, statistics
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from brand_chain import ask, STYLE, BASE, BRAND, parse_chat_logs

load_dotenv(BASE / '.env', override=True)
REPORTS = BASE / 'reports'
REPORTS.mkdir(exist_ok=True)

def rule_checks(text: str) -> int:
    score = 100
    # 1) Без эмодзи
    if re.search(r"[\U0001F300-\U0001FAFF]", text):
        score -= 20
    # 2) Без крика!!!
    if "!!!" in text:
        score -= 10
    # 3) Количество предложений
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) > STYLE['tone']['sentences_max']:
        score -= 10
    # 4) Личное мнение
    if re.search(r"\b(я считаю|по моему мнению|на мой взгляд|я думаю)\b", text, re.IGNORECASE):
        score -= 10
    if re.search(r"\b(лучше|хуже|самый|хороший|плохой)\b", text, re.IGNORECASE):
        score -= 10
    # 5) Эмоциональная окраска
    if re.search(r"\b(люблю|ненавижу|обожаю|терпеть не могу)\b", text, re.IGNORECASE):
        score -= 10
    # 6) Чрезмерная формальность
    if re.search(r"\b(уважаемый|позвольте сообщить|высокоуважаемый|достопочтенный)\b", text, re.IGNORECASE):
        score -= 5
    # 7) Аббревиатуры и сленг
    if re.search(r"\b(лол|кек|имхо|тпк|вк|ок|зз)\b", text, re.IGNORECASE):
        score -= 20

    return max(score, 0)
class Grade(BaseModel):
    score: int = Field(..., ge=0, le=100)
    notes: str

LLM = ChatOpenAI(temperature=0)

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"Ты — строгий ревьюер соответствия голосу бренда {BRAND}"),
    ("system", f"Тон: {STYLE['tone']['persona']}. Избегай: {', '.join(STYLE['tone']['avoid'])}. "
               f"Обязательно: {', '.join(STYLE['tone']['must_include'])}."),
    ("human", "Ответ ассистента:\n{answer}\n\nДай целочисленный score 0..100 и краткие заметки почему.")
])

def llm_grade(text: str) -> dict:
    parser = LLM.with_structured_output(Grade)
    return (GRADE_PROMPT | parser).invoke({"answer": text})

def eval_batch(prompts: List[str]) -> dict:
    results = []
    for p in prompts:
        reply = ask(p)
        answer = reply.get("answer", "")
        rule = rule_checks(answer)
        g = llm_grade(answer)
        
        final = int(0.4 * rule + 0.6 * g.score)
        if final >= 80:
            tone = "Ответ соответствует голосу бренда."
        else:
            tone = "Ответ не соответствует голосу бренда."

        results.append({
            "prompt": p,
            "answer": answer,
            "actions": reply.get("actions", []),
            "tone": tone,
            "rule_score": rule,
            "llm_score": g.score,
            "final_score": final,
            "notes": g.notes,
        })

    mean_final = round(statistics.mean(r["final_score"] for r in results), 2)
    out = {"mean_final_score": mean_final, "items": results}
    (REPORTS / "style_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding = "utf-8")
    return out

def eval_log_pairs(pairs: list[dict]) -> dict:
    results = []

    for item in pairs:
        answer = item["answer"]

        rule = rule_checks(answer)
        g = llm_grade(answer)
        final = int(0.4 * rule + 0.6 * g.score)

        if final >= 85:
            tone = "да — соответствует голосу бренда."
        elif final >= 70:
            tone = "частично — есть отклонения."
        else:
            tone = "нет — не соответствует."

        results.append({
            "prompt": item["prompt"],
            "answer": answer,
            "tone": tone,
            "rule_score": rule,
            "llm_score": g.score,
            "final_score": final,
            "notes": g.notes,
        })

    mean_final = round(
        statistics.mean(r["final_score"] for r in results),
        2
    )

    out = {"mean_final_score": mean_final, "items": results}
    (REPORTS / "chat_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding = "utf-8")
    return out


if __name__ == "__main__":
    eval_prompts = (BASE / "data" / "eval_prompts.txt").read_text(encoding="utf-8").strip().splitlines()
    report = eval_batch(eval_prompts)
    print(f"Средний финальный балл: {report['mean_final_score']}")
    print(f"Детальный отчёт сохранён в {REPORTS / 'style_eval.json'}")
    
    # Оценка по логам
    # log_path = BASE / "chat_session.log"
    # log_pairs = parse_chat_logs(log_path)
    # log_report = eval_log_pairs(log_pairs)
    # print(f"Средний финальный балл по логам: {log_report['mean_final_score']}")
    # print(f"Детальный отчёт по логам сохранён в {REPORTS / 'chat_eval.json'}")
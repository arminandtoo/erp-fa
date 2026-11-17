"""ارزیابی:  python eval/evaluate.py [--agent] [--note "چه چیزی عوض شد"]
بازیابی RAG: ماده درست در رتبه چندم برمی‌گردد (hit@k و MRR).
انتخاب ابزار (--agent): مدل برای هر پرسش ابزار درست را صدا می‌زند یا نه.
هر اجرا یک خط به eval/history.jsonl اضافه می‌کند."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag

HISTORY_PATH = Path(__file__).parent / "history.jsonl"
TOP_K = 5

# (پرسش، منبع‌های قابل قبول) — پرسش‌ها عمداً محاوره‌ای و با واژه‌های متفاوت از متن نوشته شده‌اند
QUESTIONS = [
    ("اگه ۲۰ تا کیسه کود بگیرم ارزون‌تر حساب می‌کنید؟", ["ماده ۱۷", "پرسش متداول ۷"]),
    ("۶۰ بسته بخرم چند درصد تخفیف می‌خوره؟", ["ماده ۱۷", "پرسش متداول ۷"]),
    ("عضو تعاونی روستایی هستم، تخفیف خاصی دارم؟", ["ماده ۱۷"]),
    ("هزینه ارسال بار ۲۰۰ کیلویی به شیراز چقدره؟", ["ماده ۱۸"]),
    ("از چه مبلغی به بالا ارسال رایگانه؟", ["ماده ۱۸"]),
    ("چند روزه بار به مشهد می‌رسه؟", ["ماده ۱۸", "پرسش متداول ۱۳"]),
    ("کود رو باز نکردم، می‌تونم پسش بدم؟", ["ماده ۲۰"]),
    ("سمی که خریدم رو میشه مرجوع کرد؟", ["ماده ۲۰", "پرسش متداول ۹"]),
    ("بذرهایی که کاشتم سبز نشدن، خسارت میدید؟", ["ماده ۲۱", "پرسش متداول ۱۰", "ماده ۱۵"]),
    ("سمپاش چند ماه گارانتی داره؟", ["ماده ۲۱"]),
    ("قیمتی که الان میگید تا چند روز دیگه همینه؟", ["ماده ۱۶", "پرسش متداول ۱۴"]),
    ("روی سمپاش مالیات ارزش افزوده میاد؟", ["ماده ۱۶"]),
    ("کارشناس‌ها پنجشنبه تا ساعت چند جواب میدن؟", ["ماده ۸"]),
    ("شکایت کنم کی جواب می‌گیرم؟", ["ماده ۷"]),
    ("هر جریب چند متر مربعه؟", ["ماده ۵", "پرسش متداول ۱۱"]),
    ("یک لیتر خاک گلدون چند کیلوئه؟", ["ماده ۵"]),
    ("دوره کارنس چیه؟", ["ماده ۴", "پرسش متداول ۵"]),
    ("روغن ولک رو با کاپتان قاطی کنم اشکال داره؟", ["ماده ۱۴", "پرسش متداول ۴"]),
    ("موقع گل دادن درخت‌ها میشه حشره‌کش زد؟", ["ماده ۱۴"]),
    ("موقع سم‌پاشی چه وسایل ایمنی لازمه؟", ["ماده ۱۴"]),
    ("کود حیوانی تازه دارید؟", ["پرسش متداول ۸", "ماده ۱۳"]),
    ("نیترات کلسیم رو با چه کودهایی نباید مخلوط کرد؟", ["ماده ۱۳", "پرسش متداول ۴"]),
    ("بذر رو کجا نگه دارم خراب نشه؟", ["ماده ۱۵"]),
    ("برگ لیموم زرد شده چیکار کنم؟", ["پرسش متداول ۳"]),
    ("برای گل‌های آپارتمانی چه کودی خوبه؟", ["پرسش متداول ۲"]),
    ("کود مرغی رو دو برابر بریزم بهتر جواب میده؟", ["پرسش متداول ۱۵", "ماده ۱۲"]),
    ("اگه جنس به اندازه کافی تو انبار نباشه چی میشه؟", ["ماده ۱۹"]),
    ("قبل از اینکه بگید چقدر کود بریزم چه اطلاعاتی باید بدم؟", ["ماده ۹"]),
    ("تعداد بسته رو چطوری حساب می‌کنید، رو به بالا گرد میشه؟", ["ماده ۱۰"]),
    ("برای ۱۵ هکتار زمین هم خود ربات برنامه میده؟", ["ماده ۱۲"]),
]


# (پرسش، ابزار مورد انتظار، سرویس مورد انتظار) — None یعنی مدل نباید ابزاری صدا بزند
TOOL_CASES = [
    ("قیمت کود اوره چنده؟", "search_products", None),
    ("برای شته چه سمی دارید؟", "search_products", None),
    ("بذر گوجه‌فرنگی موجود دارید؟", "search_products", None),
    ("مشخصات کامل محصول F-001 رو بگو", "get_product", None),
    ("۱۲ بسته از محصول F-001 چقدر میشه؟", "calculate_usage", None),
    ("محصول F-001 برای ۲ هکتار چند کیسه لازمه؟", "calculate_usage", None),
    ("با محصول P-001 برای سمپاش ۲۰ لیتری چقدر سم بریزم؟", "calculate_usage", None),
    ("هزینه ارسال به شیراز چقدره؟", "search_regulations", None),
    ("کودی که خریدم رو میشه پس داد؟", "search_regulations", None),
    ("شرایط گارانتی کالاها چیه؟", "search_regulations", None),
    ("پشتیبانی تا ساعت چند جواب میده؟", "search_regulations", None),
    ("خرید عمده تخفیف داره؟", "search_regulations", None),
    ("مالیات امسالم چقدر میشه؟", "citizen_service", "tax"),
    ("خلافی ماشینم چقدره؟", "citizen_service", "traffic"),
    ("وضعیت سربازیم چیه؟", "citizen_service", "military"),
    ("چند ماه سابقه بیمه دارم؟", "citizen_service", "insurance"),
    ("یارانه‌ام چقدره؟", "citizen_service", "subsidy"),
    ("سوء پیشینه دارم؟", "citizen_service", "judicial"),
    ("پایتخت فرانسه کجاست؟", None, None),
    ("یه شعر درباره بهار بگو", None, None),
    ("سلام، خسته نباشید", None, None),
]


def matches(source, accepted):
    return any(source.startswith(a + " ") or source == a for a in accepted)


def metrics(ranks):
    """ranks: رتبه اولین نتیجه درست برای هر پرسش (None = پیدا نشد)."""
    n = len(ranks)
    result = {f"hit@{k}": round(sum(r is not None and r <= k for r in ranks) / n, 3) for k in (1, 3, 5)}
    result["mrr"] = round(sum(1 / r for r in ranks if r) / n, 3)
    return result


def eval_retrieval():
    ranks, failures = [], []
    for question, accepted in QUESTIONS:
        sources = [h["source"] for h in rag.search(question, k=TOP_K)]
        rank = next((i + 1 for i, s in enumerate(sources) if matches(s, accepted)), None)
        ranks.append(rank)
        if rank != 1:
            failures.append({"kind": "retrieval", "question": question, "expected": " / ".join(accepted),
                             "got": f"rank={rank} | " + "، ".join(sources[:3])})
    return metrics(ranks), failures


def eval_tool_choice():
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent import SYSTEM_PROMPT
    from config import get_llm
    from tools import TOOLS

    system = SystemMessage(content=f"{SYSTEM_PROMPT}\n\nUser identity: not verified.")
    # فقط اولین تصمیم مدل سنجیده می‌شود؛ ابزارها اجرا نمی‌شوند
    responses = get_llm().bind_tools(TOOLS).batch([[system, HumanMessage(content=q)] for q, _, _ in TOOL_CASES])
    correct, failures = 0, []
    for (question, tool, service), response in zip(TOOL_CASES, responses):
        call = response.tool_calls[0] if response.tool_calls else None
        got = (call["name"], call["args"].get("service")) if call else (None, None)
        if got == (tool, service):
            correct += 1
        else:
            failures.append({"kind": "tool", "question": question, "expected": f"{tool} {service or ''}".strip(),
                             "got": f"{got[0]} {got[1] or ''}".strip()})
    return round(correct / len(TOOL_CASES), 3), failures


def load_history():
    if not HISTORY_PATH.exists():
        return []
    return [json.loads(line) for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(note="", with_agent=False):
    from config import MODEL

    scores, failures = eval_retrieval()
    record = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "note": note, "embedding": rag.EMBEDDING_MODEL,
              "chunks": len(rag.load_index()[0]), "llm": MODEL if with_agent else None, **scores, "tool_accuracy": None}
    if with_agent:
        record["tool_accuracy"], tool_failures = eval_tool_choice()
        failures += tool_failures
    record["failures"] = failures
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true", help="also evaluate tool selection (calls the LLM)")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    record = run(args.note, args.agent)
    for failure in record.pop("failures"):
        print(f"[{failure['kind']}] {failure['question']}\n    expected: {failure['expected']}\n    got: {failure['got']}")
    print("\n" + " | ".join(f"{k} = {v}" for k, v in record.items() if k.startswith("hit@") or k in ("mrr", "tool_accuracy")))

import json
import uuid

import gradio as gr
import pandas as pd

import agent
from eval import evaluate
from verification import VERIFICATION_INSTRUCTION

TOOL_LABELS = {
    "search_products": "🔍 جستجوی محصولات",
    "get_product": "📦 مشخصات محصول",
    "calculate_usage": "🧮 محاسبه مقدار مصرف و هزینه",
    "citizen_service": "👤 استعلام اطلاعات شخصی",
    "search_regulations": "📖 جستجو در آیین‌نامه",
    "verify_identity": "🔐 احراز هویت با تصویر",
}
EMPTY_TRACE = "### 🔎 ردپای پاسخ\n\nهنوز پاسخی داده نشده است."
MAX_RESULT_CHARS = 700

CSS = """
.rtl, .rtl .prose, .rtl .message {direction: rtl; text-align: right;}
.rtl pre, .rtl code {direction: ltr; text-align: left;}
.status-bar {background: linear-gradient(90deg, #2e7d32 0%, #558b2f 100%); color: white;
             padding: 10px 15px; border-radius: 8px; font-weight: bold;}
.status-bar p {color: white;}
"""


def render_trace(trace):
    lines = ["### 🔎 ردپای پاسخ", ""]
    if not trace["steps"]:
        lines.append("این پاسخ بدون استفاده از ابزار داده شد.")
    for number, step in enumerate(trace["steps"], 1):
        result = step["result"] if isinstance(step["result"], str) else json.dumps(step["result"], ensure_ascii=False)
        if len(result) > MAX_RESULT_CHARS:
            result = result[:MAX_RESULT_CHARS] + " …"
        lines.append(f"**{number}. {TOOL_LABELS.get(step['tool'], step['tool'])}** (`{step['tool']}`)")
        if step["args"]:
            lines.append(f"ورودی: `{json.dumps(step['args'], ensure_ascii=False)}`")
        lines += ["نتیجه:", f"```\n{result}\n```", ""]
    lines.append(f"---\n⏱ {trace['seconds']} ثانیه | 🔢 {trace['tokens']:,} توکن")
    return "\n".join(lines)


def status_text(session_id):
    citizen = agent.get_citizen(session_id)
    if citizen:
        return f"✅ احراز هویت شده: {citizen['first_name']} {citizen['last_name']}"
    if agent.is_awaiting_image(session_id):
        return "⏸ منتظر تصویر احراز هویت (گفتگو متوقف شده است)"
    return "⏳ احراز هویت نشده (برای خدمات شخصی لازم است)"


def welcome():
    return [{"role": "assistant", "content": agent.WELCOME_MESSAGE}]


def send_message(message, history, session_id):
    if not message or not message.strip():
        return history, "", gr.skip(), gr.skip()
    history = history + [{"role": "user", "content": message}]
    try:
        reply, trace = agent.chat(session_id, message.strip())
        trace_md = render_trace(trace)
    except Exception as e:
        reply, trace_md = f"❌ خطا در ارتباط با مدل: {e}", EMPTY_TRACE
    history.append({"role": "assistant", "content": reply})
    return history, "", trace_md, status_text(session_id)


def send_image(image, history, session_id):
    if image is None:
        return history, None, gr.skip(), gr.skip()
    history = history + [{"role": "user", "content": "📷 [تصویر برای احراز هویت ارسال شد]"}]
    try:
        reply, trace = agent.verify(session_id, image)
        trace_md = render_trace(trace)
    except Exception as e:
        reply, trace_md = f"❌ خطا در ارتباط با مدل: {e}", EMPTY_TRACE
    history.append({"role": "assistant", "content": reply})
    return history, None, trace_md, status_text(session_id)


def reset_chat():
    session_id = str(uuid.uuid4())
    return session_id, welcome(), EMPTY_TRACE, status_text(session_id)

METRICS = ["hit@1", "hit@3", "hit@5", "mrr", "tool_accuracy"]


def eval_view():
    history = evaluate.load_history()
    table = [[i, r["time"], r["note"], *[r[m] for m in METRICS], r["chunks"], r["embedding"], r["llm"]]
             for i, r in enumerate(history, 1)]
    points = pd.DataFrame(
        [{"run": i, "metric": m, "value": r[m]} for i, r in enumerate(history, 1) for m in METRICS if r[m] is not None],
        columns=["run", "metric", "value"])
    failures = [[f["kind"], f["question"], f["expected"], f["got"]] for f in history[-1]["failures"]] if history else []
    return table[::-1], points, failures


def run_eval(note, with_agent):
    evaluate.run(note.strip(), with_agent)
    return eval_view()


def create_app():
    with gr.Blocks(title="سامانه خدمات هوشمند سبزدشت") as app:
        session_id = gr.State(lambda: str(uuid.uuid4()))

        gr.Markdown("# 🌱 سامانه خدمات هوشمند سبزدشت", elem_classes="rtl")

        with gr.Tab("💬 گفتگو"):
            with gr.Row():
                with gr.Column(scale=2):
                    status_bar = gr.Markdown("⏳ احراز هویت نشده (برای خدمات شخصی لازم است)", elem_classes=["rtl", "status-bar"])
                    chatbot = gr.Chatbot(value=welcome(), label="گفتگو", height=480, elem_classes="rtl")
                    with gr.Row():
                        msg_input = gr.Textbox(
                            show_label=False, scale=4, lines=1, elem_classes="rtl",
                            placeholder="مثال: کود اوره برای ۲ هکتار گندم چقدر می‌شود؟")
                        send_btn = gr.Button("📤 ارسال", scale=1, variant="primary")
                    reset_btn = gr.Button("🔄 گفتگوی جدید", variant="secondary")

                with gr.Column(scale=1):
                    with gr.Accordion("🔐 احراز هویت", open=True):
                        gr.Markdown(f"📸 {VERIFICATION_INSTRUCTION}", elem_classes="rtl")
                        image_input = gr.Image(label="تصویر احراز هویت", type="filepath", height=180)
                        upload_btn = gr.Button("📷 ارسال تصویر", variant="primary")
                    trace_panel = gr.Markdown(EMPTY_TRACE, elem_classes="rtl")

        with gr.Tab("📊 ارزیابی"):
            gr.Markdown(
                "هر اجرا یک ردیف به `eval/history.jsonl` اضافه می‌کند. بعد از تغییر آیین‌نامه، تکه‌بندی، مدل امبدینگ "
                "یا پرامپت دوباره اجرا کنید و عددها را مقایسه کنید.", elem_classes="rtl")
            with gr.Row():
                eval_note = gr.Textbox(label="یادداشت این اجرا (چه چیزی عوض شد؟)", scale=3, elem_classes="rtl")
                eval_agent = gr.Checkbox(label="ارزیابی انتخاب ابزار هم اجرا شود (۲۱ فراخوانی LLM)", scale=1)
                eval_btn = gr.Button("▶ اجرای ارزیابی", variant="primary", scale=1)
            eval_plot = gr.LinePlot(x="run", y="value", color="metric", y_lim=[0, 1.05], height=300, title="روند معیارها")
            eval_table = gr.Dataframe(
                headers=["#", "زمان", "یادداشت", *METRICS, "تکه‌ها", "امبدینگ", "LLM"], label="تاریخچه اجراها", interactive=False)
            eval_failures = gr.Dataframe(
                headers=["نوع", "پرسش", "مورد انتظار", "نتیجه"], label="خطاهای آخرین اجرا (رتبه ۱ نشده‌ها)",
                interactive=False, wrap=True, elem_classes="rtl")

        chat_outputs = [chatbot, msg_input, trace_panel, status_bar]
        send_btn.click(send_message, [msg_input, chatbot, session_id], chat_outputs)
        msg_input.submit(send_message, [msg_input, chatbot, session_id], chat_outputs)
        upload_btn.click(send_image, [image_input, chatbot, session_id], [chatbot, image_input, trace_panel, status_bar])
        reset_btn.click(reset_chat, None, [session_id, chatbot, trace_panel, status_bar])
        eval_outputs = [eval_table, eval_plot, eval_failures]
        eval_btn.click(run_eval, [eval_note, eval_agent], eval_outputs)
        app.load(eval_view, None, eval_outputs)

    return app


if __name__ == "__main__":
    create_app().launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft(), css=CSS)

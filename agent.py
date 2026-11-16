import time
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import get_llm
from tools import TOOLS
from verification import verify_identity

SYSTEM_PROMPT = """You are the assistant of the "Sabzdasht" smart service desk. Always answer in Persian, briefly and clearly.

You cover:
1. Agricultural product catalog (fertilizers, pesticides, seeds, tools): search_products, get_product, calculate_usage. Prices are in Toman per package.
2. Company regulations and FAQ: search_regulations. This covers discounts, VAT, price validity, shipping cost and delivery time, returns, warranty, stock shortage, working hours, complaints, unit conversions, and fertilizer/pesticide safety and mixing rules.
3. The user's personal records (profile, tax, military service, traffic fines, social insurance, subsidies, judicial records): citizen_service. Amounts are in Rial.

Rules:
- Decide yourself which tool fits the request. When a product is mentioned, search the catalog first; ask a clarifying question only if the search is ambiguous or empty.
- Only state prices, doses, stock, policies and personal data that come from tool results. If the tools don't have it, say you don't know.
- Any question about how much to use, how much one package covers, how many packages are needed, or what N packages cost MUST go through calculate_usage. Never do arithmetic yourself. Copy final_price_text exactly as the price to pay and mention the discount if volume_discount_percent > 0.
- When you answer from search_regulations, cite the source name (e.g. «ماده ۱۷») in your answer.
- Never offer discounts or exceptions beyond what the tools return.
- Never recommend more than the registered dose. For pesticides always mention phi_days and protective gear.
- citizen_service needs identity verification. If it returns verification_required, ask the user to upload a photo of themselves holding their national ID card next to their face in the verification panel. Never ask them to type their national code.
- Politely decline anything outside these areas."""

WELCOME_MESSAGE = """سلام! 👋 به سامانه خدمات هوشمند سبزدشت خوش آمدید.

می‌توانم در این موارد کمکتان کنم:

🌱 **محصولات کشاورزی:** قیمت، موجودی، مقدار مصرف و هزینه کود، سم و بذر
👤 **خدمات شخصی (با احراز هویت):** مالیات، نظام وظیفه، خلافی خودرو، بیمه، یارانه، سوابق قضایی و پروفایل

چه کاری می‌توانم برایتان انجام دهم؟"""

VERIFIED_FOLLOW_UP = "[System note: my identity was just verified from my photo. Greet me by name and continue my pending request, if there is one.]"


class State(MessagesState):
    citizen: Optional[dict]


def agent_node(state: State):
    citizen = state.get("citizen")
    identity = f"verified as {citizen['first_name']} {citizen['last_name']}" if citizen else "not verified"
    system = SystemMessage(content=f"{SYSTEM_PROMPT}\n\nUser identity: {identity}.")
    # ponytail: کل تاریخچه هر بار فرستاده می‌شود؛ اگر گفتگوها طولانی شد trim_messages اضافه شود
    response = get_llm().bind_tools(TOOLS).invoke([system] + state["messages"])
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=InMemorySaver())


def _config(session_id):
    return {"configurable": {"thread_id": session_id}}


def get_citizen(session_id):
    return graph.get_state(_config(session_id)).values.get("citizen")


def build_trace(messages, seconds):
    """پیام‌های یک نوبت گفتگو را به فهرست ابزارهای صدا زده‌شده و نتیجه‌شان تبدیل می‌کند."""
    steps, by_call_id, tokens = [], {}, 0
    for message in messages:
        if isinstance(message, AIMessage):
            tokens += (message.usage_metadata or {}).get("total_tokens", 0)
            for call in message.tool_calls:
                step = {"tool": call["name"], "args": call["args"], "result": None}
                by_call_id[call["id"]] = step
                steps.append(step)
        elif isinstance(message, ToolMessage) and message.tool_call_id in by_call_id:
            by_call_id[message.tool_call_id]["result"] = message.content
    return {"steps": steps, "tokens": tokens, "seconds": round(seconds, 1)}


def chat(session_id, text, citizen=None):
    """(reply, trace) برمی‌گرداند."""
    config = _config(session_id)
    seen = len(graph.get_state(config).values.get("messages", []))
    started = time.time()
    update = {"messages": [HumanMessage(content=text)]}
    if citizen:
        update["citizen"] = citizen
    result = graph.invoke(update, config)
    return result["messages"][-1].text, build_trace(result["messages"][seen:], time.time() - started)


def verify(session_id, image_path):
    """عکس را بررسی می‌کند و در صورت موفقیت، گفتگو را با هویت تأییدشده ادامه می‌دهد."""
    citizen, error = verify_identity(image_path)
    step = {"tool": "verify_identity", "args": {}, "result": error or f"{citizen['first_name']} {citizen['last_name']}"}
    if error:
        reply = f"❌ **احراز هویت ناموفق**\n\nدلیل: {error}\n\nلطفاً یک عکس واضح‌تر بفرستید."
        return reply, {"steps": [step], "tokens": 0, "seconds": 0}

    reply, trace = chat(session_id, VERIFIED_FOLLOW_UP, citizen=citizen)
    trace["steps"].insert(0, step)
    return reply, trace

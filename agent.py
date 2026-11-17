import time
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from config import get_llm
from tools import TOOLS
from verification import VERIFICATION_INSTRUCTION, verify_identity

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
- For personal records just call citizen_service, even if the user is not verified: the system pauses and collects the identity photo by itself. Never ask the user for their national code or a photo yourself.
- Politely decline anything outside these areas."""

WELCOME_MESSAGE = """سلام! 👋 به سامانه خدمات هوشمند سبزدشت خوش آمدید.

می‌توانم در این موارد کمکتان کنم:

🌱 **محصولات کشاورزی:** قیمت، موجودی، مقدار مصرف و هزینه کود، سم و بذر
👤 **خدمات شخصی (با احراز هویت):** مالیات، نظام وظیفه، خلافی خودرو، بیمه، یارانه، سوابق قضایی و پروفایل

چه کاری می‌توانم برایتان انجام دهم؟"""

PROTECTED_TOOL = "citizen_service"
NO_IMAGE_NEEDED = "در حال حاضر نیازی به تصویر نیست. لطفاً ابتدا درخواست خود را مطرح کنید."


class State(MessagesState):
    citizen: Optional[dict]
    verification_error: Optional[str]


def agent_node(state: State):
    citizen = state.get("citizen")
    identity = f"verified as {citizen['first_name']} {citizen['last_name']}" if citizen else "not verified"
    system = SystemMessage(content=f"{SYSTEM_PROMPT}\n\nUser identity: {identity}.")
    # TODO: کل تاریخچه هر بار فرستاده می‌شود؛ اگر گفتگوها طولانی شد trim_messages اضافه شود
    response = get_llm().bind_tools(TOOLS).invoke([system] + state["messages"])
    return {"messages": [response]}


def route_after_agent(state: State) -> Literal["verification", "tools", "__end__"]:
    calls = state["messages"][-1].tool_calls
    if not calls:
        return END
    if not state.get("citizen") and any(call["name"] == PROTECTED_TOOL for call in calls):
        return "verification"
    return "tools"


def verification_node(state: State) -> Command[Literal["tools", "verification", "agent"]]:
    """انسان در حلقه: گراف همین‌جا می‌ایستد تا کاربر عکس بفرستد (یا با یک پیام متنی منصرف شود)."""
    answer = interrupt({"instruction": VERIFICATION_INSTRUCTION, "error": state.get("verification_error")})

    if answer.get("image_path"):
        citizen, error = verify_identity(answer["image_path"])
        if citizen:
            return Command(goto="tools", update={"citizen": citizen, "verification_error": None})
        return Command(goto="verification", update={"verification_error": error})

    # هر tool call باید پاسخ داشته باشد، وگرنه مدل پیام بعدی را قبول نمی‌کند
    cancelled = [
        ToolMessage(content="cancelled: user sent a new message instead of the identity photo", tool_call_id=call["id"])
        for call in state["messages"][-1].tool_calls
    ]
    return Command(
        goto="agent",
        update={"messages": cancelled + [HumanMessage(content=answer["text"])], "verification_error": None})


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("verification", verification_node)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent)
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=InMemorySaver())


def _config(session_id):
    return {"configurable": {"thread_id": session_id}}


def get_citizen(session_id):
    return graph.get_state(_config(session_id)).values.get("citizen")


def is_awaiting_image(session_id):
    return bool(graph.get_state(_config(session_id)).interrupts)


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


def _run(session_id, graph_input, seen):
    """گراف را اجرا می‌کند و (reply, trace) برمی‌گرداند؛ اگر گراف متوقف شد، پیام درخواست عکس پاسخ است."""
    started = time.time()
    result = graph.invoke(graph_input, _config(session_id))
    trace = build_trace(result["messages"][seen:], time.time() - started)

    if not result.get("__interrupt__"):
        return result["messages"][-1].text, trace

    request = result["__interrupt__"][0].value
    for step in trace["steps"]:
        if step["result"] is None:
            step["result"] = "⏸ interrupt: waiting for identity photo"
    if request["error"]:
        return f"❌ **احراز هویت ناموفق**\n\nدلیل: {request['error']}\n\n📸 دوباره تلاش کنید: {request['instruction']}", trace
    return f"🔐 برای این خدمت احراز هویت لازم است.\n\n📸 {request['instruction']}", trace


def chat(session_id, text):
    snapshot = graph.get_state(_config(session_id))
    messages = snapshot.values.get("messages", [])
    if snapshot.interrupts:
        return _run(session_id, Command(resume={"text": text}), seen=len(messages))
    return _run(session_id, {"messages": [HumanMessage(content=text)]}, seen=len(messages))


def verify(session_id, image_path):
    snapshot = graph.get_state(_config(session_id))
    if not snapshot.interrupts:
        return NO_IMAGE_NEEDED, {"steps": [], "tokens": 0, "seconds": 0}

    # seen یکی عقب‌تر است تا tool call معلق هم در ردپا بیاید
    reply, trace = _run(session_id, Command(resume={"image_path": image_path}), seen=len(snapshot.values["messages"]) - 1)
    citizen = get_citizen(session_id)
    outcome = f"{citizen['first_name']} {citizen['last_name']}" if citizen else "rejected"
    trace["steps"].insert(0, {"tool": "verify_identity", "args": {}, "result": outcome})
    return reply, trace

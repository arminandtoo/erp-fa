"""تست‌های بدون نیاز به LLM:  python -m pytest"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, ToolMessage

import database as db
import tools
from agent import build_trace
from app import render_trace


def test_search_finds_urea():
    assert db.search_products("اوره")[0]["sku"] == "F-001"
    assert db.search_products("كود اوره")[0]["sku"] == "F-001"  # ك عربی
    assert all(p["category"] == "بذر" for p in db.search_products(category="بذر"))


def test_usage_soil_and_field():
    urea = db.get_product("f-001")
    assert tools.usage(urea)["one_package_covers"]["soil_kg"] == 100_000
    field = tools.usage(urea, area_hectare=2)["field"]
    assert field["packages"] == 10 and field["total_price_toman"] == 14_500_000
    assert field["volume_discount_percent"] == 5 and field["final_price_toman"] == 13_775_000  # مثال ۲ آیین‌نامه


def test_volume_discount_rules():
    urea, sprayer = db.get_product("F-001"), db.get_product("T-001")
    assert tools.usage(urea, packages=9)["order"]["volume_discount_percent"] == 0
    assert tools.usage(urea, packages=12)["order"]["final_price_toman"] == 16_530_000
    assert tools.usage(urea, packages=50)["order"]["volume_discount_percent"] == 10
    assert tools.usage(sprayer, packages=20)["order"]["volume_discount_percent"] == 0  # ابزار تخفیف ندارد
    assert tools.usage(urea, soil_kg=100_001)["soil"]["packages"] == 2  # گرد کردن به بالا


def test_get_product_includes_coverage():
    assert tools.get_product.func("F-004")["one_package_covers"]["spray_water_liters"] == 5000


def test_usage_spray_units():
    imidacloprid = tools.usage(db.get_product("P-001"), tank_liters=20)["spray"]
    assert imidacloprid["amount_needed"] == 10 and imidacloprid["unit"] == "میلی‌لیتر"
    assert imidacloprid["packages"] == 1
    assert "error" in tools.usage(db.get_product("P-001"), soil_kg=5)


def test_tax_matches_reference_formula():
    citizen = db.find_citizen_by_national_code("0023456789")  # درآمد ۱.۵ میلیارد، نرخ شغلی ۲۰٪
    tax = tools.calculate_tax(citizen)
    # مشمول: ۱ میلیارد → پلکانی ۵۰ میلیون + شغلی ۲۰۰ میلیون، میانگین = ۱۲۵ میلیون
    assert tax["base_tax"] == 125_000_000
    assert 1400 < tax["year"] < 1500


def test_citizen_service_requires_verification():
    assert tools.citizen_service.func(service="tax", state={}) == {"error": "verification_required"}
    citizen = db.find_citizen_by_national_code("0012345678")
    traffic = tools.citizen_service.func(service="traffic", state={"citizen": citizen})
    assert len(traffic["vehicles"]) == 2 and traffic["total_unpaid"] == 4_000_000


def test_trace_pairs_calls_with_results():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "get_product", "args": {"sku": "F-001"}, "id": "c1"}]),
        ToolMessage(content='{"sku": "F-001"}', tool_call_id="c1"),
        AIMessage(content="پاسخ"),
    ]
    trace = build_trace(messages, 1.23)
    assert trace["steps"] == [{"tool": "get_product", "args": {"sku": "F-001"}, "result": '{"sku": "F-001"}'}]
    assert "get_product" in render_trace(trace)


class ScriptedLLM:
    """به‌جای مدل واقعی: پاسخ‌های از پیش نوشته‌شده را به ترتیب برمی‌گرداند."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.prompts.append(messages)
        return next(self.responses)


def _tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_graph_runs_tools_and_keeps_verified_citizen(monkeypatch):
    import agent

    llm = ScriptedLLM([
        _tool_call("citizen_service", {"service": "tax"}, "c1"), AIMessage(content="نیاز به احراز هویت"),
        _tool_call("citizen_service", {"service": "tax"}, "c2"), AIMessage(content="مالیات شما"),
        _tool_call("calculate_usage", {"sku": "F-001", "area_hectare": 2}, "c3"), AIMessage(content="۱۰ کیسه"),
    ])
    monkeypatch.setattr(agent, "get_llm", lambda: llm)

    reply, trace = agent.chat("s1", "مالیات من؟")
    assert reply == "نیاز به احراز هویت" and "verification_required" in trace["steps"][0]["result"]
    assert "not verified" in llm.prompts[0][0].content

    citizen = db.find_citizen_by_national_code("0023456789")
    reply, trace = agent.chat("s1", agent.VERIFIED_FOLLOW_UP, citizen=citizen)
    assert "125000000" in trace["steps"][0]["result"] and len(trace["steps"]) == 1

    reply, trace = agent.chat("s1", "اوره برای ۲ هکتار؟")
    assert "14500000" in trace["steps"][0]["result"]
    assert agent.get_citizen("s1")["last_name"] == "احمدی"
    assert "verified as مریم احمدی" in llm.prompts[-1][0].content
    assert agent.get_citizen("other-session") is None

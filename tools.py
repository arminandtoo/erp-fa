import math
from typing import Annotated, Literal, Optional

import jdatetime
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import database as db
import rag

MINIMUM_EXEMPTION = 500_000_000
TAX_BRACKETS = [
    (500_000_000, 0.0),
    (1_000_000_000, 0.10),
    (1_500_000_000, 0.15),
    (2_000_000_000, 0.20),
    (float("inf"), 0.25),
]
RETIREMENT_MONTHS = 300
# ماده ۱۷ آیین‌نامه: (حداقل تعداد بسته، درصد تخفیف)
VOLUME_DISCOUNTS = [(50, 10), (10, 5)]
DISCOUNT_EXEMPT_CATEGORY = "ابزار و تجهیزات"

# ضریب تبدیل هر واحد به واحد پایه (گرم / میلی‌لیتر / عدد)
UNIT_FACTORS = {"کیلوگرم": 1000, "گرم": 1, "لیتر": 1000, "میلی‌لیتر": 1, "عدد": 1, "عدد بذر": 1}


def calculate_tax(citizen):
    today = jdatetime.date.today()
    income = citizen["annual_income"] or 0
    taxable = max(0, income - MINIMUM_EXEMPTION)

    bracket_tax, previous = 0, 0
    for limit, rate in TAX_BRACKETS:
        if taxable <= previous:
            break
        bracket_tax += (min(taxable, limit) - previous) * rate
        previous = limit
    base_tax = (bracket_tax + taxable * (citizen["tax_rate"] or 0)) / 2

    exemptions = db.get_citizen_exemptions(citizen["id"], today.strftime("%Y-%m-%d"))
    exemption_percent = min(sum(e["percentage"] for e in exemptions), 100)
    return {
        "year": today.year,
        "annual_income": income,
        "base_tax": round(base_tax),
        "exemptions": exemptions,
        "exemption_percent": exemption_percent,
        "final_tax": round(base_tax * (1 - exemption_percent / 100)),
    }


def tax_service(citizen):
    return {**calculate_tax(citizen), "job": citizen["job_name"], "history": db.get_tax_history(citizen["id"])}


def military_service(citizen):
    if citizen["gender"] == "زن":
        return {"note": "military service applies to men only"}
    return db.get_military_status(citizen["id"]) or {"note": "no record; user should visit a Police+10 office"}


def traffic_service(citizen):
    violations = db.get_traffic_violations_by_citizen(citizen["id"])
    return {
        "vehicles": db.get_vehicles_by_citizen(citizen["id"]),
        "violations": violations,
        "total_unpaid": sum(v["fine_amount"] for v in violations if not v["is_paid"]),
    }


def insurance_service(citizen):
    record = db.get_social_security_status(citizen["id"])
    if not record:
        return {"note": "no social security record"}
    return {**record, "months_until_retirement": max(0, RETIREMENT_MONTHS - record["total_months"])}


def subsidy_service(citizen):
    subsidies = db.get_subsidies_by_citizen(citizen["id"])
    return {
        "subsidies": subsidies,
        "total_active_monthly": sum(s["monthly_amount"] for s in subsidies if s["is_active"]),
    }


def judicial_service(citizen):
    return {"records": db.get_judicial_records(citizen["id"]), "note": "informational only, not an official certificate"}


def profile_service(citizen):
    return {k: v for k, v in citizen.items() if k != "id"}


CITIZEN_SERVICES = {
    "profile": profile_service,
    "tax": tax_service,
    "military": military_service,
    "traffic": traffic_service,
    "insurance": insurance_service,
    "subsidy": subsidy_service,
    "judicial": judicial_service,
}


@tool
def citizen_service(
    service: Literal["profile", "tax", "military", "traffic", "insurance", "subsidy", "judicial"],
    state: Annotated[dict, InjectedState],
) -> dict:
    """Fetch the verified user's own personal record. Requires identity verification.
    profile: personal info | tax: yearly tax calculation + history | military: conscription status |
    traffic: vehicles and traffic fines | insurance: social security | subsidy: government subsidies |
    judicial: court records. All amounts are in Rial; dates are Jalali."""
    citizen = state.get("citizen")
    if not citizen:
        return {"error": "verification_required"}
    return CITIZEN_SERVICES[service](citizen)


@tool
def search_products(text: str = "", category: str = "") -> list:
    """Search the agricultural product catalog (fertilizers, pesticides, seeds, tools).
    text: Persian keywords (product name, crop, or problem), e.g. "اوره" or "گوجه‌فرنگی".
    category: optional Persian category name filter, e.g. "کود آلی", "بذر", "سم قارچ‌کش".
    Returns up to 10 products with sku, price (Toman per package), package size and stock."""
    return db.search_products(text, category)


@tool
def get_product(sku: str) -> dict:
    """Get full details of one product by sku: price, stock, doses, phi_days, usage notes.
    dose_soil_g_per_kg: grams per kg of soil | dose_hectare: per hectare, in dose_hectare_unit |
    dose_spray_per_1000l: grams (solid) or ml (liquid) per 1000 L of water |
    phi_days: pre-harvest interval for pesticides.
    one_package_covers: precomputed coverage of a single package; quote it as-is."""
    product = db.get_product(sku)
    if not product:
        return {"error": "product_not_found"}
    return {**product, "one_package_covers": usage(product)["one_package_covers"]}


def _price(product, count):
    exempt = product["category"] == DISCOUNT_EXEMPT_CATEGORY
    discount = 0 if exempt else next((percent for minimum, percent in VOLUME_DISCOUNTS if count >= minimum), 0)
    total = count * product["price"]
    final = round(total * (1 - discount / 100))
    return {
        "packages": count,
        "total_price_toman": total,
        "volume_discount_percent": discount,
        "final_price_toman": final,
        "final_price_text": f"{final:,} تومان",
        "in_stock": count <= product["stock"],
    }


def _packages(product, amount, unit):
    needed = amount * UNIT_FACTORS[unit]
    count = math.ceil(needed / (product["package_size"] * UNIT_FACTORS[product["package_unit"]]) - 1e-9)
    return {"amount_needed": round(amount, 3), "unit": unit, **_price(product, count)}


def usage(product, soil_kg=None, area_hectare=None, tank_liters=None, packages=None):
    liquid = product["package_unit"] in ("لیتر", "میلی‌لیتر")
    package_base = product["package_size"] * UNIT_FACTORS[product["package_unit"]]
    result = {"sku": product["sku"], "name": product["name"], "price_per_package_toman": product["price"]}

    if packages is not None:
        result["order"] = _price(product, packages)
    if soil_kg is not None:
        if not product["dose_soil_g_per_kg"]:
            return {"error": "no_soil_dose", "hint": "this product is not dosed per kg of soil"}
        result["soil"] = _packages(product, soil_kg * product["dose_soil_g_per_kg"], "گرم")
    if area_hectare is not None:
        if not product["dose_hectare"]:
            return {"error": "no_hectare_dose"}
        result["field"] = _packages(product, area_hectare * product["dose_hectare"], product["dose_hectare_unit"])
    if tank_liters is not None:
        if not product["dose_spray_per_1000l"]:
            return {"error": "no_spray_dose"}
        result["spray"] = _packages(
            product, tank_liters * product["dose_spray_per_1000l"] / 1000, "میلی‌لیتر" if liquid else "گرم")

    if all(v is None for v in (soil_kg, area_hectare, tank_liters, packages)):
        covers = {}
        if product["dose_soil_g_per_kg"]:
            covers["soil_kg"] = round(package_base / product["dose_soil_g_per_kg"])
        if product["dose_hectare"]:
            dose_base = product["dose_hectare"] * UNIT_FACTORS[product["dose_hectare_unit"]]
            covers["area_hectare"] = round(package_base / dose_base, 3)
        if product["dose_spray_per_1000l"]:
            covers["spray_water_liters"] = round(package_base / product["dose_spray_per_1000l"] * 1000)
        result["one_package_covers"] = covers
    return result


@tool
def calculate_usage(
    sku: str,
    soil_kg: Optional[float] = None,
    area_hectare: Optional[float] = None,
    tank_liters: Optional[float] = None,
    packages: Optional[int] = None,
) -> dict:
    """Exact dose and cost calculation for a product. Always use this instead of doing arithmetic.
    Give soil_kg (pot/greenhouse soil weight), area_hectare (field area; 1 hectare = 10000 m2),
    tank_liters (sprayer water volume), or packages (price of a given number of packages).
    With none of them, returns how much one package covers.
    Returns amount needed, number of packages (rounded up), stock check and prices in Toman.
    The volume discount is already applied: final_price_text is the amount to pay."""
    product = db.get_product(sku)
    if not product:
        return {"error": "product_not_found"}
    if packages is not None and packages < 1:
        return {"error": "packages_must_be_positive"}
    return usage(product, soil_kg, area_hectare, tank_liters, packages)


@tool
def search_regulations(query: str) -> list:
    """Search the company regulations and FAQ (Persian). Use for policies and rules: volume discounts,
    VAT, price validity, shipping cost and delivery time, returns, warranty, stock shortage, working hours,
    response times, unit conversions (jarib, liters of soil), fertilizer/pesticide safety and mixing rules.
    query: a short Persian question. Returns the best matching articles with their source name."""
    return rag.search(query, k=4)


TOOLS = [citizen_service, search_products, get_product, calculate_usage, search_regulations]

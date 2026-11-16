import sqlite3
from contextlib import closing
from pathlib import Path

DATABASE_PATH = Path(__file__).parent / "data" / "erp.db"

ARABIC_TO_PERSIAN = str.maketrans("يكۀة", "یکهه")


def query(sql, params=(), one=False):
    with closing(sqlite3.connect(DATABASE_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(sql, params)]
    return (rows[0] if rows else None) if one else rows


def execute(sql, params=()):
    with closing(sqlite3.connect(DATABASE_PATH)) as conn:
        conn.execute(sql, params)
        conn.commit()


def find_citizen_by_national_code(national_code):
    return query('''
        SELECT c.id, c.national_code, c.first_name, c.last_name,
               c.father_name, c.birth_date, c.gender, c.annual_income,
               c.phone_number, c.address, c.is_active,
               ct.name AS city_name, p.name AS province_name,
               j.name AS job_name, j.tax_rate
        FROM citizens c
        LEFT JOIN cities ct ON c.city_id = ct.id
        LEFT JOIN provinces p ON ct.province_id = p.id
        LEFT JOIN job_categories j ON c.job_category_id = j.id
        WHERE c.national_code = ?
    ''', (national_code,), one=True)


def get_citizen_exemptions(citizen_id, today):
    # تاریخ‌های جدول شمسی‌اند؛ today هم باید شمسی و به شکل YYYY-MM-DD باشد
    return query('''
        SELECT e.name, e.percentage
        FROM citizen_exemptions ce
        JOIN tax_exemptions e ON ce.exemption_id = e.id
        WHERE ce.citizen_id = ? AND ce.valid_from <= ? AND ce.valid_until >= ?
    ''', (citizen_id, today, today))


def get_tax_history(citizen_id):
    return query('''
        SELECT year, income, tax_amount, is_paid, payment_date
        FROM tax_records WHERE citizen_id = ? ORDER BY year DESC
    ''', (citizen_id,))


def get_military_status(citizen_id):
    return query('''
        SELECT status, service_type, start_date, end_date, exemption_type, card_number, notes
        FROM military_service WHERE citizen_id = ?
    ''', (citizen_id,), one=True)


def get_vehicles_by_citizen(citizen_id):
    return query(
        "SELECT plate_number, brand, model, color, year FROM vehicles WHERE citizen_id = ?",
        (citizen_id,))


def get_traffic_violations_by_citizen(citizen_id):
    return query('''
        SELECT v.plate_number, tv.violation_type, tv.violation_date, tv.location,
               tv.fine_amount, tv.is_paid, tv.payment_date
        FROM traffic_violations tv
        JOIN vehicles v ON tv.vehicle_id = v.id
        WHERE v.citizen_id = ?
        ORDER BY tv.violation_date DESC
    ''', (citizen_id,))


def get_social_security_status(citizen_id):
    return query('''
        SELECT insurance_number, status, employer, start_date, total_months,
               last_payment_date, monthly_salary
        FROM social_security WHERE citizen_id = ?
    ''', (citizen_id,), one=True)


def get_subsidies_by_citizen(citizen_id):
    return query('''
        SELECT subsidy_type, monthly_amount, is_active, start_date, bank_name
        FROM subsidies WHERE citizen_id = ?
    ''', (citizen_id,))


def get_judicial_records(citizen_id):
    return query('''
        SELECT case_number, case_type, status, court_name, filing_date, verdict, is_cleared
        FROM judicial_records WHERE citizen_id = ?
    ''', (citizen_id,))


def log_verification(citizen_id, national_code, step, is_successful, failure_reason=None):
    execute('''
        INSERT INTO verification_logs (citizen_id, national_code, scenario_type, is_successful, failure_reason)
        VALUES (?, ?, ?, ?, ?)
    ''', (citizen_id, national_code, step, is_successful, failure_reason))


PRODUCT_LIST_COLUMNS = '''
    p.sku, p.name, c.name AS category, p.brand, p.package_size, p.package_unit,
    p.price, p.stock, p.suitable_crops
'''


def search_products(text="", category=""):
    words = text.translate(ARABIC_TO_PERSIAN).split()
    haystack = "p.name || ' ' || c.name || ' ' || COALESCE(p.suitable_crops, '') || ' ' || COALESCE(p.description, '')"
    score = " + ".join(f"({haystack} LIKE ?)" for _ in words) or "1"
    params = [f"%{w}%" for w in words]
    sql = f'''
        SELECT {PRODUCT_LIST_COLUMNS}, {score} AS score
        FROM products p JOIN product_categories c ON p.category_id = c.id
        WHERE p.is_active = 1 AND score > 0
    '''
    if category:
        sql += " AND c.name LIKE ?"
        params.append(f"%{category.translate(ARABIC_TO_PERSIAN)}%")
    rows = query(sql + " ORDER BY score DESC, p.sku LIMIT 10", params)
    for r in rows:
        del r["score"]
    return rows


def get_product(sku):
    return query('''
        SELECT p.sku, p.name, c.name AS category, p.brand, p.package_size, p.package_unit,
               p.price, p.stock, p.dose_soil_g_per_kg, p.dose_hectare, p.dose_hectare_unit,
               p.dose_spray_per_1000l, p.phi_days, p.suitable_crops, p.description
        FROM products p JOIN product_categories c ON p.category_id = c.id
        WHERE p.sku = ? AND p.is_active = 1
    ''', (sku.strip().upper(),), one=True)


def list_categories():
    return [r["name"] for r in query("SELECT name FROM product_categories ORDER BY id")]

# erp-fa — دستیار هوشمند فارسی

دستیار گفتگومحور فارسی که به پایگاه داده یک سازمان وصل می‌شود: قیمت و مقدار مصرف محصولات کشاورزی را حساب می‌کند و پس از احراز هویت با تصویر، اطلاعات شخصی کاربر (مالیات، خلافی، بیمه و…) را می‌دهد.

## چطور کار می‌کند

```
Gradio (app.py)
   │
LangGraph agent (agent.py)  ── LLM خودش ابزار مناسب را انتخاب می‌کند
   │
tools.py ── search_products / get_product / calculate_usage / citizen_service / search_regulations
   │                                                              │
database.py ── data/erp.db (SQLite)                 rag.py ── آیین‌نامه + امبدینگ محلی

verification.py ── مدل بینایی: شخص + کارت ملی → کد ملی → جستجو در پایگاه داده
```

- **بدون intent classifier:** تشخیص نوع درخواست با tool-calling خود مدل انجام می‌شود.
- **محاسبات در کد:** دوز، تعداد بسته، تخفیف حجمی و هزینه را `calculate_usage` حساب می‌کند، نه مدل.
- **RAG محلی:** آیین‌نامه ماده‌به‌ماده تکه می‌شود و با `intfloat/multilingual-e5-small` (روی CPU، بدون API) جستجو می‌شود. مدل در پاسخ نام ماده را ذکر می‌کند.
- **ردپای پاسخ:** کنار گفتگو نشان داده می‌شود که کدام ابزار با چه ورودی‌ای صدا زده شد و چه برگرداند.
- **خدمات شخصی محافظت‌شده:** `citizen_service` بدون احراز هویت فقط `verification_required` برمی‌گرداند.

## اجرا

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cp .env.example .env        # کلید AvalAI را داخلش بگذارید
.venv/Scripts/python app.py
```

سپس <http://127.0.0.1:7860> را باز کنید.

## تست

```bash
.venv/Scripts/python -m pytest
```

## ارزیابی بازیابی

```bash
.venv/Scripts/python eval/eval_retrieval.py
```

روی ۳۰ پرسش محاوره‌ای: **hit@1 = 80٪، hit@3 = 90٪، MRR = 0.85**

بار اول مدل امبدینگ (حدود ۴۷۰ مگابایت) از HuggingFace دانلود می‌شود.

## داده‌ها

- `data/erp.db` — شهروندان نمونه و ۳۵ محصول کشاورزی (ساختگی و آموزشی)
- `data/seed_products.py` — ساخت دوباره جدول محصولات
- `docs/aeen-nameh-porsesh-pasokh.md` — آیین‌نامه پرسش و پاسخ (منبع RAG)

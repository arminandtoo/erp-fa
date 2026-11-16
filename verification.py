import base64
import mimetypes
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import get_llm
from database import find_citizen_by_national_code, log_verification

VERIFICATION_INSTRUCTION = "یک عکس واضح از خودتان بگیرید که کارت ملی را کنار صورتتان نگه داشته‌اید."

VERIFICATION_PROMPT = """You are an identity verification checker. Inspect the photo.
Pass conditions:
1. A real person's face is clearly visible.
2. The same person is holding an Iranian national ID card (کارت ملی) next to them.
3. The 10-digit national code on the card is readable.
Return the national code as exactly 10 ASCII digits (convert Persian digits), or null if unreadable.
If any condition fails, give a short rejection_reason in Persian."""

PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


class ImageCheck(BaseModel):
    person_detected: bool
    id_card_detected: bool
    card_held_by_person: bool
    national_code: Optional[str] = Field(None, description="10 ASCII digits or null")
    rejection_reason: Optional[str] = Field(None, description="Short reason in Persian, or null")


def extract_national_code(image_path):
    """(national_code, error) برمی‌گرداند؛ یکی از این دو همیشه None است."""
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    check = get_llm().with_structured_output(ImageCheck).invoke([
        SystemMessage(content=VERIFICATION_PROMPT),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]),
    ])

    if not check.person_detected:
        return None, check.rejection_reason or "شخصی در تصویر شناسایی نشد"
    if not check.id_card_detected:
        return None, check.rejection_reason or "کارت ملی در تصویر یافت نشد"
    if not check.card_held_by_person:
        return None, check.rejection_reason or "کارت ملی باید در دست و کنار خود شما باشد"

    code = "".join(ch for ch in (check.national_code or "").translate(PERSIAN_DIGITS) if ch.isdigit())
    if len(code) != 10:
        return None, "کد ملی روی کارت خوانا نبود"
    return code, None


def verify_identity(image_path):
    """(citizen, error) برمی‌گرداند. فقط کد ملی‌ای که در پایگاه داده فعال باشد پذیرفته می‌شود."""
    try:
        national_code, error = extract_national_code(image_path)
    except Exception as e:
        return None, f"خطا در پردازش تصویر: {e}"

    if error:
        log_verification(None, None, "image_verification", False, error)
        return None, error

    citizen = find_citizen_by_national_code(national_code)
    if not citizen:
        log_verification(None, national_code, "database_search", False, "کاربر یافت نشد")
        return None, f"کد ملی {national_code} در سامانه ثبت نشده است"
    if not citizen["is_active"]:
        log_verification(citizen["id"], national_code, "account_status", False, "حساب غیرفعال")
        return None, "حساب کاربری شما غیرفعال است؛ با پشتیبانی تماس بگیرید"

    log_verification(citizen["id"], national_code, "full_verification", True)
    return citizen, None

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DOC_PATH = ROOT / "docs" / "aeen-nameh-porsesh-pasokh.md"
INDEX_PATH = ROOT / "data" / "regulations_index.npz"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

NORMALIZE = str.maketrans("يكۀة۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "یکهه01234567890123456789")

_model = None
_index = None


def normalize(text):
    return re.sub(r"\s+", " ", text.translate(NORMALIZE)).strip()


def split_document(markdown):
    """هر «ماده» و هر پرسش متداول یک تکه می‌شود؛ عنوان فصل برای زمینه نگه داشته می‌شود."""
    chunks, chapter = [], ""
    for section in re.split(r"\n(?=#{2,3} )", markdown):
        title, _, body = section.partition("\n")
        body = body.replace("---", "").strip()
        if title.startswith("## "):
            chapter = title[3:].strip()
            if "پرسش‌های متداول" in chapter:
                for qa in re.split(r"\n(?=\*\*[۰-۹]+\. )", body):
                    number = re.match(r"\*\*([۰-۹]+)\. ", qa)
                    if number:
                        chunks.append({"source": f"پرسش متداول {number.group(1)}", "chapter": chapter, "text": qa.strip()})
        elif title.startswith("### "):
            chunks.append({"source": title[4:].strip(), "chapter": chapter, "text": body})
    return chunks


def embed(texts, prefix):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    # مدل‌های e5 پیشوند "query: " و "passage: " لازم دارند
    return _model.encode([prefix + normalize(t) for t in texts], normalize_embeddings=True)


def load_index():
    """ایندکس را از فایل می‌خواند و اگر سند یا مدل عوض شده باشد دوباره می‌سازد."""
    global _index
    if _index is not None:
        return _index
    markdown = DOC_PATH.read_text(encoding="utf-8")
    fingerprint = hashlib.sha256((EMBEDDING_MODEL + markdown).encode()).hexdigest()

    if INDEX_PATH.exists():
        saved = np.load(INDEX_PATH)
        if str(saved["fingerprint"]) == fingerprint:
            _index = (json.loads(str(saved["chunks"])), saved["vectors"])
            return _index

    chunks = split_document(markdown)
    vectors = embed([f"{c['chapter']} | {c['source']}\n{c['text']}" for c in chunks], "passage: ")
    np.savez(INDEX_PATH, fingerprint=fingerprint, chunks=json.dumps(chunks, ensure_ascii=False), vectors=vectors)
    _index = (chunks, vectors)
    return _index


def search(query, k=3):
    chunks, vectors = load_index()
    # TODO: جستجوی خطی با numpy؛ برای چند هزار تکه به بالا Qdrant یا pgvector لازم می‌شود
    scores = vectors @ embed([query], "query: ")[0]
    best = np.argsort(-scores)[:k]
    return [{"source": chunks[i]["source"], "text": chunks[i]["text"], "score": round(float(scores[i]), 3)} for i in best]


if __name__ == "__main__":
    import sys
    for hit in search(" ".join(sys.argv[1:]) or "اگر ۱۰ کیسه بخرم تخفیف دارد؟"):
        print(hit["score"], hit["source"])

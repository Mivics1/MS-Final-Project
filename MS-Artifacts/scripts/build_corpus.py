"""
Build data/emails.csv from the two source corpora.

Sources (see README 'Getting the data'; both are published research corpora):

  Class 0 (real-world phishing, operationally treated as human-origin)
      Nazario phishing corpus, yearly mbox files 2022-2024
      https://monkey.org/~jose/phishing/   (licence: CC BY 4.0; attribution:
      Jose Nazario). Recent years are used deliberately so this class is
      contemporaneous with the 2024 AI class (avoids the era-of-writing
      confound). NOTE: the corpus is hand-classified as phishing, not
      hand-verified for authorship; see the dissertation's limitations.

  Class 1 (known AI-generated phishing)
      Eze & Shamir (2024), "Analysis and Prevention of AI-Based Phishing Email
      Attacks", Electronics 13(10):1839. 865 AI-generated phishing emails
      produced through the DeepAI text-generation service.
      https://people.cs.ksu.edu/~lshamir/data/ai_phishing/

Processing:
  - mbox parsing with the stdlib `mailbox` + `email` modules; prefer the
    text/plain part, fall back to text/html stripped to text.
  - Both classes are reduced to BODY TEXT ONLY (the AI corpus 'Header:' subject
    line is removed) so the classifier sees the same kind of evidence per class.
  - Cleaning: drop the mbox folder-internal pseudo-message, collapse whitespace,
    drop bodies < MIN_CHARS or > MAX_CHARS, drop non-English-looking bodies
    (stopword heuristic).
  - CREDENTIAL-HARVESTING SCREEN (rule-based, reproducible): a message is
    retained only if it (a) references an account/credential/verification
    subject (ACCOUNT_TERMS) AND (b) solicits an action through a link, button,
    form or reply (ACTION_PATTERNS, CTA_PATTERNS or a URL). Exclusion counts
    are reported per source. The CTA_PATTERNS arm was added after a manual
    audit showed the URL/verb arms alone behaved asymmetrically across the two
    sources; see scripts/audit_screen_sample.py.
  - Exact-duplicate removal per class on a hash of the lower-cased alphanumeric
    characters (catches reformatted copies of the same template; semantic
    near-duplicates are handled separately by the grouped-split evaluation).
  - The larger class is down-sampled (seeded) so the final dataset is balanced.
  - Provenance: SHA-256 checksums of every raw source file are recorded in
    corpus_summary.json together with licence and download date.

Run:  python scripts/build_corpus.py
Outputs: data/emails.csv, data/corpus_summary.json

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import csv
import hashlib
import html as html_mod
import json
import mailbox
import os
import random
import re
import sys
import unicodedata

RAW = os.path.join("data", "raw")
OUT_CSV = os.path.join("data", "emails.csv")
OUT_SUMMARY = os.path.join("data", "corpus_summary.json")

NAZARIO_FILES = ["phishing-2022", "phishing-2023", "phishing-2024"]
AI_DIR = os.path.join(RAW, "AI_phishing_emails")

SEED = 42
MIN_CHARS = 150      # too short to carry stylometric signal
MAX_CHARS = 8000     # cut off mega-dumps / concatenated digests

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")

# crude English check: proportion of very common English function words
_STOPWORDS = {
    "the", "to", "and", "of", "a", "in", "you", "your", "is", "for",
    "we", "this", "that", "on", "with", "be", "have", "are", "if",
    "please", "or", "not", "will", "it", "as", "our", "by", "from",
}

# ---------------------------------------------------------------------------
# Credential-harvesting screen (operational definition, reported in the
# dissertation). A message is credential-harvesting when it BOTH:
#   (a) references an account / credential / verification subject, and
#   (b) solicits an action through a link, attachment, form or reply.
# The two term lists below ARE the definition; changing them changes the
# corpus, so they are fixed and version-controlled.
# ---------------------------------------------------------------------------
# Stems ("verif" covers verify/verified/verification) so that morphological
# variants do not silently fall through the screen.
ACCOUNT_TERMS = [
    "account", "password", "passcode", "credential", "login", "log in",
    "log-in", "sign in", "sign-in", "signed-in", "verif", "validat",
    "confirm your", "confirmation", "authentic",
    "security", "suspend", "deactivat", "expir",
    "locked", "unusual activity", "unauthorized", "unauthorised",
    "update your", "billing", "payment", "invoice", "mailbox", "webmail",
    "identity", "reset", "restricted", "limited access",
]
ACTION_PATTERNS = [
    "click", "tap ", "follow the link", "follow this link", "visit",
    "go to", "open the attachment", "use the link", "link below",
    "button below", "reply with", "reply to this", "respond with",
    "submit", "fill", "provide your", "enter your", "send us your",
    "log on", "download",
]
# Imperative call-to-action wording typical of a link or button. These matter
# because converting an HTML email to text discards the href, leaving only the
# anchor's visible label ("UPDATE MY DETAILS", "START VERIFICATION", "Renew
# Now"). Without them the action test was satisfied by essentially every
# message in the AI corpus (all 865 contain a literal URL) but failed for the
# majority of the real-world HTML messages regardless of their actual intent —
# a source-dependent selection effect found by the manual audit
# (scripts/audit_screen_sample.py) and corrected here.
CTA_PATTERNS = [
    "verif", "update your", "update my", "confirm",
    "activat", "renew", "restore", "release", "retrieve", "recover",
    "resolve", "review", "unlock", "proceed", "log in", "login", "sign in",
    "keep the same", "cancel request",
    "validat", "continue", "view account", "repair", "authenticat",
    "reinput", "re-enter",
]
_URL_HINT_RE = re.compile(r"https?://|www\.|hxxp", re.IGNORECASE)
# invisible characters used to obfuscate keywords (soft hyphen, zero-width)
_INVISIBLE_RE = re.compile(r"[­​‌‍﻿]")
# Cyrillic/Greek homoglyphs substituted for Latin letters to defeat keyword
# filters ("pаssword" with a Cyrillic a). Folded for SCREENING ONLY: the stored
# text keeps the author's original characters, because the choice to obfuscate
# is itself part of the writing style under study.
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ԁ": "d", "ᴏ": "o", "А": "A", "Е": "E", "О": "O",
    "Р": "P", "С": "C", "У": "Y", "Х": "X", "Ι": "I", "Α": "A", "Ε": "E",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Ν": "N", "Μ": "M", "Κ": "K", "Β": "B",
})


def _match_text(text: str) -> str:
    """Normalise away obfuscation so the screen sees the intended words."""
    return _INVISIBLE_RE.sub("", text).translate(_HOMOGLYPHS).lower()


def is_credential_harvesting(text: str) -> bool:
    """Two-condition screen: the message must concern an account/credential
    subject AND solicit an action (link, button, form or reply)."""
    low = _match_text(text)
    has_subject = any(t in low for t in ACCOUNT_TERMS)
    has_action = (
        any(t in low for t in ACTION_PATTERNS)
        or any(t in low for t in CTA_PATTERNS)
        or bool(_URL_HINT_RE.search(low))
    )
    return has_subject and has_action


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def html_to_text(payload: str) -> str:
    payload = _TAG_RE.sub(" ", payload)
    payload = re.sub(r"<br\s*/?>|</p>|</div>|</tr>|</li>", "\n", payload, flags=re.I)
    payload = _HTML_RE.sub(" ", payload)
    return html_mod.unescape(payload)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # strip soft hyphens / zero-width characters. Phishing authors insert these
    # to break up keywords ("Pa­ssw­ord") and defeat filters; left in place they
    # also fragment tokens and corrupt word-length statistics, which are among the
    # most discriminative features.
    text = _INVISIBLE_RE.sub("", text)
    text = text.replace(" ", " ")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def looks_english(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 20:
        return False
    hits = sum(1 for w in words if w in _STOPWORDS)
    return hits / len(words) >= 0.12


def dedupe_key(text: str) -> str:
    return hashlib.sha1(re.sub(r"[^a-z0-9]", "", text.lower()).encode()).hexdigest()


def body_from_message(msg) -> str:
    """Prefer text/plain; fall back to text/html stripped to text."""
    plain, html = [], []
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        charset = part.get_content_charset() or "utf-8"
        try:
            decoded = payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            decoded = payload.decode("utf-8", errors="replace")
        (plain if ctype == "text/plain" else html).append(decoded)
    if plain:
        return "\n".join(plain)
    if html:
        return html_to_text("\n".join(html))
    return ""


def load_nazario() -> tuple[list[dict], dict]:
    rows, stats = [], {}
    for fname in NAZARIO_FILES:
        path = os.path.join(RAW, fname)
        if not os.path.exists(path):
            sys.exit(f"missing {path} - download it first (see README)")
        mbox = mailbox.mbox(path)
        kept = total = excl_cred = excl_internal = excl_len = excl_lang = 0
        for msg in mbox:
            total += 1
            subj = str(msg.get("Subject", ""))
            if "FOLDER INTERNAL DATA" in subj:
                excl_internal += 1
                continue
            text = normalise(body_from_message(msg))
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                excl_len += 1
                continue
            if not looks_english(text):
                excl_lang += 1
                continue
            if not is_credential_harvesting(text):
                excl_cred += 1
                continue
            rows.append({"text": text, "label": 0, "source": fname})
            kept += 1
        stats[fname] = {
            "messages": total,
            "excluded_folder_internal": excl_internal,
            "excluded_length_filter": excl_len,
            "excluded_non_english": excl_lang,
            "excluded_not_credential_harvesting": excl_cred,
            "kept_after_clean": kept,
        }
    return rows, stats


def load_ai() -> tuple[list[dict], dict]:
    if not os.path.isdir(AI_DIR):
        sys.exit(f"missing {AI_DIR} - download & extract it first (see README)")
    rows = []
    excl_cred = excl_len = 0
    names = sorted(os.listdir(AI_DIR))
    for name in names:
        fp = os.path.join(AI_DIR, name)
        if not name.endswith(".txt"):
            continue
        with open(fp, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        # drop the 'Header: <subject>' line so both classes are body-only
        lines = raw.splitlines()
        if lines and lines[0].lower().startswith("header:"):
            lines = lines[1:]
        text = normalise("\n".join(lines))
        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            excl_len += 1
            continue
        if not is_credential_harvesting(text):
            excl_cred += 1
            continue
        rows.append({"text": text, "label": 1, "source": "eze_shamir_2024"})
    return rows, {"files": len([n for n in names if n.endswith('.txt')]),
                  "excluded_length_filter": excl_len,
                  "excluded_not_credential_harvesting": excl_cred,
                  "kept_after_clean": len(rows)}


def dedupe(rows: list[dict]) -> tuple[list[dict], int]:
    seen, out = set(), []
    for r in rows:
        k = dedupe_key(r["text"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out, len(rows) - len(out)


def main() -> None:
    random.seed(SEED)
    human, naz_stats = load_nazario()
    ai, ai_stats = load_ai()

    human, human_dupes = dedupe(human)
    ai, ai_dupes = dedupe(ai)
    human_after_dedupe, ai_after_dedupe = len(human), len(ai)

    # balance: seeded down-sample of the larger class
    n = min(len(human), len(ai))
    random.shuffle(human)
    random.shuffle(ai)
    human, ai = human[:n], ai[:n]
    human_downsampled = human_after_dedupe - n
    ai_downsampled = ai_after_dedupe - n

    rows = human + ai
    random.shuffle(rows)

    os.makedirs("data", exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["text", "label", "source"])
        w.writeheader()
        w.writerows(rows)

    # provenance: checksums of every raw source file, plus licence / download
    checksums = {}
    for fname in NAZARIO_FILES:
        p = os.path.join(RAW, fname)
        if os.path.exists(p):
            checksums[fname] = sha256_file(p)
    ai_tar = os.path.join(RAW, "AI_phishing_emails.tar.gz")
    if os.path.exists(ai_tar):
        checksums["AI_phishing_emails.tar.gz"] = sha256_file(ai_tar)

    provenance = {
        "class0_source": "Nazario phishing corpus, yearly archives 2022-2024",
        "class0_url": "https://monkey.org/~jose/phishing/",
        "class0_licence": "CC BY 4.0 (attribution: Jose Nazario)",
        "class1_source": ("Eze & Shamir (2024) AI-generated phishing corpus "
                          "(865 emails via the DeepAI text-generation service)"),
        "class1_url": "https://people.cs.ksu.edu/~lshamir/data/ai_phishing/",
        "download_date": "2026-07-22",
        "sha256": checksums,
    }

    summary = {
        "seed": SEED,
        "min_chars": MIN_CHARS,
        "max_chars": MAX_CHARS,
        "credential_harvesting_screen": {
            "account_terms": len(ACCOUNT_TERMS),
            "action_patterns": len(ACTION_PATTERNS),
            "rule": "retain iff (account/credential subject) AND "
                    "(action via link/form/reply or a URL is present)",
        },
        "nazario": naz_stats,
        "nazario_exact_duplicates_removed": human_dupes,
        "eze_shamir": ai_stats,
        "ai_exact_duplicates_removed": ai_dupes,
        # full funnel so every count in the report reconciles
        "funnel": {
            "nazario": {
                "parsed": sum(v["messages"] for v in naz_stats.values()),
                "excluded_folder_internal": sum(v["excluded_folder_internal"] for v in naz_stats.values()),
                "excluded_length_filter": sum(v["excluded_length_filter"] for v in naz_stats.values()),
                "excluded_non_english": sum(v["excluded_non_english"] for v in naz_stats.values()),
                "excluded_not_credential_harvesting": sum(v["excluded_not_credential_harvesting"] for v in naz_stats.values()),
                "retained_before_dedupe": sum(v["kept_after_clean"] for v in naz_stats.values()),
                "exact_duplicates_removed": human_dupes,
                "after_dedupe": human_after_dedupe,
                "removed_by_downsampling": human_downsampled,
                "final": n,
            },
            "eze_shamir": {
                "parsed": ai_stats["files"],
                "excluded_folder_internal": 0,
                "excluded_length_filter": ai_stats["excluded_length_filter"],
                "excluded_non_english": 0,
                "excluded_not_credential_harvesting": ai_stats["excluded_not_credential_harvesting"],
                "retained_before_dedupe": ai_stats["kept_after_clean"],
                "exact_duplicates_removed": ai_dupes,
                "after_dedupe": ai_after_dedupe,
                "removed_by_downsampling": ai_downsampled,
                "final": n,
            },
        },
        "per_class_after_balancing": n,
        "total": len(rows),
        "provenance": provenance,
    }
    with open(OUT_SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

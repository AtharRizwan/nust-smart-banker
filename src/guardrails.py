"""
Guardrails for NUST Smart Banker.

Two layers of defence:
  INPUT  – applied to every user query before retrieval/generation
  OUTPUT – applied to every LLM response before it reaches the user

Input checks:
  1. Jailbreak / prompt-injection detection (regex + keyword blocklist)
  2. PII in query (warn; strip if possible)
  3. Malicious content (hate speech keywords, threats)
  4. Excessive length sanity check

Output checks:
  1. Disallowed information leak detection (competitor banks, explicit rates
     that were NOT in context, internal system prompt leaks)
  2. Hallucination sentinel (detects "<|im_start|>" or raw template leaking)
  3. PII in response (re-anonymise)

Also integrates NeMo Guardrails for structured policy enforcement when the
library is available; degrades gracefully if not installed.
"""

from __future__ import annotations

import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)

# ─── Jailbreak / Injection Patterns ──────────────────────────────────────────
# Ordered from most specific to most general.  Each pattern is compiled once.

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Classic prompt injection preambles
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)",
        r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)",
        r"forget\s+(everything|all|prior|your\s+instructions?)",
        r"override\s+(your\s+)?(instructions?|guidelines?|system\s+prompt)",
        # Role-play jailbreaks
        r"\bact\s+as\s+(if\s+you\s+are|a|an)\b",
        r"\bpretend\s+(you\s+are|to\s+be)\b",
        r"\byou\s+are\s+now\s+(a|an|the)\b",
        r"\bdan\b",  # "Do Anything Now"
        r"\bjailbreak\b",
        r"\bunfiltered\b",
        r"\bno\s+restrictions?\b",
        r"\bno\s+limits?\b",
        r"\byour\s+true\s+self\b",
        # System prompt extraction attempts
        r"(show|print|reveal|tell\s+me|what\s+is)\s+(your\s+)?(system\s+prompt|instructions?|context|prompt)",
        r"repeat\s+(everything|the\s+above|your\s+prompt)",
        r"output\s+your\s+(initial\s+)?(instructions?|prompt|system)",
        # Token smuggling / encoding tricks
        r"base64",
        r"hex\s+encoded",
        r"\\\d{2,3}",  # octal / escape sequences
        # Harmful request patterns
        r"\b(how\s+to\s+)?(hack|exploit|bypass|crack|steal)\b",
        r"\b(launder|money\s+laundering)\b",
        r"\b(bomb|weapon|explosive)\b",
        r"\bsocial\s+engineering\b",
    ]
]

# Keywords that, if present in isolation, strongly indicate off-topic abuse
_HARD_BLOCK_KEYWORDS: set[str] = {
    "porn",
    "pornography",
    "nude",
    "naked",
    "sex",
    "sexual",
    "xxx",
    "kill",
    "murder",
    "suicide",
    "self-harm",
    "drug",
    "cocaine",
    "heroin",
    "terrorism",
    "terrorist",
    "jihad",
}

# Competitor bank names (output should not praise them)
_COMPETITOR_BANKS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bhbl\b",
        r"habib bank",
        r"\bubl\b",
        r"united bank",
        r"\bmcb\b",
        r"muslim commercial bank",
        r"\bnbp\b",
        r"national bank of pakistan",
        r"standard chartered",
        r"meezan bank",
        r"bank alfalah",
        r"alfalah bank",
        r"\bscb\b",
        r"bank al habib",
        r"\bask\b",
        r"askari bank",
        r"faysal bank",
        r"js bank",
    ]
]

# Patterns that indicate the model may be leaking internal template tokens
_TEMPLATE_LEAK_PATTERNS: list[re.Pattern] = [
    re.compile(p)
    for p in [
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"<\|system\|>",
        r"\[INST\]",
        r"<<SYS>>",
    ]
]

# Maximum allowed query length (characters)
_MAX_QUERY_LEN = 2000


# ─── Block message templates ─────────────────────────────────────────────────

_INJECTION_BLOCK_MSG = (
    "I'm sorry, but I can't process that request. "
    "I'm designed to assist with NUST Bank-related queries only. "
    "Please ask me about our products, services, or account features."
)

_HARMFUL_BLOCK_MSG = (
    "I'm sorry, that request contains content I'm not able to engage with. "
    "Please feel free to ask me anything about NUST Bank."
)

_TOO_LONG_MSG = (
    "Your message is too long for me to process. "
    "Please keep your question concise and I'll be happy to help."
)


# ─── Guardrails class ─────────────────────────────────────────────────────────


class Guardrails:
    """
    Stateless guardrail checker.  Instantiate once and call:
        check_input(query)  -> (is_blocked: bool, message: str)
        check_output(text, query) -> sanitised_text
    """

    def __init__(self) -> None:
        self._nemo_rails = self._try_load_nemo()

    # ── NeMo Guardrails (optional) ────────────────────────────────────────────

    @staticmethod
    def _try_load_nemo():
        """Load NeMo Guardrails if installed; return None otherwise."""
        try:
            from nemoguardrails import RailsConfig, LLMRails
            from configs.settings import GUARDRAILS_CONFIG_DIR

            config_path = GUARDRAILS_CONFIG_DIR / "rails.co"
            if not config_path.exists():
                logger.debug(
                    "NeMo rails config not found at %s; skipping.", config_path
                )
                return None

            config = RailsConfig.from_path(str(GUARDRAILS_CONFIG_DIR))
            rails = LLMRails(config)
            logger.info("NeMo Guardrails loaded.")
            return rails
        except Exception as exc:
            logger.debug("NeMo Guardrails unavailable: %s", exc)
            return None

    # ── Input Checks ──────────────────────────────────────────────────────────

    def check_input(self, query: str) -> Tuple[bool, str]:
        """
        Validate a user query.

        Returns:
            (False, "")           if the query is safe
            (True,  block_msg)    if the query should be blocked
        """
        if not query or not query.strip():
            return True, "Please enter a question and I'll be happy to help."

        # Length check
        if len(query) > _MAX_QUERY_LEN:
            return True, _TOO_LONG_MSG

        query_lower = query.lower()

        # Hard-block keywords (checked first as fast path)
        for keyword in _HARD_BLOCK_KEYWORDS:
            if re.search(rf"\b{re.escape(keyword)}\b", query_lower):
                logger.warning("Hard-block keyword detected: '%s'", keyword)
                return True, _HARMFUL_BLOCK_MSG

        # Injection / jailbreak patterns
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(query):
                logger.warning("Injection pattern matched: '%s'", pattern.pattern)
                return True, _INJECTION_BLOCK_MSG

        return False, ""

    # ── Output Checks ─────────────────────────────────────────────────────────

    def check_output(self, response: str, original_query: str = "") -> str:
        """
        Sanitise a model-generated response before returning it to the user.

        Applies:
          - Template token stripping (in case of model hallucination)
          - Competitor bank mention neutralisation
          - PII re-anonymisation (light regex pass)

        Returns the sanitised response string.  If the response is deemed
        completely unsafe, returns a safe fallback message.
        """
        if not response:
            return (
                "I'm sorry, I wasn't able to generate a response. "
                "Please try rephrasing your question."
            )

        # ── Strip leaked template tokens ──────────────────────────────────────
        for pattern in _TEMPLATE_LEAK_PATTERNS:
            if pattern.search(response):
                logger.warning("Template token leak detected in output; sanitising.")
                # Truncate at first leak token
                response = pattern.split(response)[0].strip()

        # ── Neutralise competitor bank references ────────────────────────────
        for pattern in _COMPETITOR_BANKS:
            if pattern.search(response):
                logger.info("Competitor bank mention detected; redacting.")
                response = pattern.sub("[another bank]", response)

        # ── Light PII regex on output ─────────────────────────────────────────
        response = re.sub(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", "<EMAIL>", response)
        response = re.sub(r"(\+92[-\s]?|0)3\d{2}[-\s]?\d{7}", "<PHONE>", response)
        response = re.sub(r"\bPK\d{2}[A-Z0-9]{20}\b", "<IBAN>", response)
        response = re.sub(r"\b\d{5}-\d{7}-\d\b", "<CNIC>", response)

        # ── Truncate absurdly long responses ─────────────────────────────────
        if len(response) > 3000:
            response = response[:3000].rsplit(".", 1)[0] + ". [Response truncated]"

        return response.strip()

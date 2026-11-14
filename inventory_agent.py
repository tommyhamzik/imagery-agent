"""Supply Chain Analytics Inventory Decision Agent.

Implements the explicit seven-stage pipeline:
    Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool
        -> Calculate -> Verify -> Explain

Supports:
    1. Economic Order Quantity (EOQ)
    2. Quantity Discount
    3. Safety Stock and Reorder Point
    4. Fixed-Period Review
    5. Newsvendor

All numerical calculations are delegated exclusively to Python functions in
`inventory_tools.py`.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from inventory_tools import (
    TOOL_REGISTRY,
    norm_cdf,
    norm_pdf,
    standard_normal_loss,
)

# ---------------------------------------------------------------------------
# Pipeline Data Structures
# ---------------------------------------------------------------------------

MODEL_LABELS: Dict[str, str] = {
    "eoq": "Economic Order Quantity (EOQ)",
    "quantity_discount": "Quantity Discount (Price-Break EOQ)",
    "safety_stock_rop": "Safety Stock and Reorder Point",
    "fixed_period_review": "Fixed-Period Review (Order-Up-To Policy)",
    "newsvendor": "Newsvendor (Single-Period Model)",
    "unknown": "Unknown / Out of Scope",
}

DEMO_QUESTIONS: List[Tuple[str, str]] = [
    (
        "1. Economic Order Quantity (EOQ)",
        "A distributor sells 12,000 units per year. Each order has a setup/ordering cost of $150, "
        "fixed holding cost is $4 per unit per year, and lead time is 5 days (250 operating days/year). "
        "What is the optimal order quantity and reorder point?",
    ),
    (
        "2. Quantity Discount",
        "Annual demand is 5,000 units and ordering cost is $49 per order. "
        "Percentage holding cost is 20% of unit price. Supplier pricing tiers: "
        "1-999 units at $5.00, 1000-1999 units at $4.80, and 2000+ units at $4.75. "
        "Which order quantity minimizes total annual cost?",
    ),
    (
        "3. Safety Stock & Reorder Point",
        "Average daily demand is 80 units with a demand standard deviation of 15 units/day. "
        "Average lead time is 5 days with a lead-time standard deviation of 1 day. "
        "Target cycle service level is 95%. What are the safety stock and reorder point?",
    ),
    (
        "4. Fixed-Period Review (P-System)",
        "In a periodic review system, review period is 7 days, lead time is 3 days, "
        "daily demand averages 60 units with std dev 12 units/day, and service level is 95%. "
        "On-hand inventory is 180 units, on-order inventory is 100 units, and backorders are 15 units. "
        "Find target inventory level S and order quantity Q.",
    ),
    (
        "5. Newsvendor (Single-Period)",
        "A seasonal jacket has selling price $65, purchase cost $40, and salvage value $12. "
        "Demand is normally distributed with mean demand 240 and demand standard deviation 45. "
        "What is the critical ratio and optimal stocking quantity?",
    ),
]


@dataclass
class ModelIdentification:
    model_type: str
    model_label: str
    confidence: float
    scores: Dict[str, float]
    matched_signals: Dict[str, List[str]]
    reason: str


@dataclass
class InputExtraction:
    inputs: Dict[str, Any]
    missing_required: List[str] = field(default_factory=list)
    conversions: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    extracted_evidence: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return len(self.missing_required) == 0


@dataclass
class ToolSelection:
    model_type: str
    tool_name: str
    display_name: str
    function: Callable[..., Dict[str, Any]]
    bound_kwargs: Dict[str, Any]
    description: str


@dataclass
class VerificationCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class VerificationResult:
    passed: bool
    checks: List[VerificationCheck]
    warnings: List[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    question: str
    status: str  # "success", "needs_clarification", "unsupported", "error"
    identification: ModelIdentification
    extraction: InputExtraction
    tool_selection: Optional[ToolSelection]
    calculation: Optional[Dict[str, Any]]
    verification: Optional[VerificationResult]
    explanation: str
    pipeline_trace: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete response and pipeline trace to JSON-safe dicts."""
        tool_dict: Optional[Dict[str, Any]] = None
        if self.tool_selection is not None:
            tool_dict = {
                "model_type": self.tool_selection.model_type,
                "tool_name": self.tool_selection.tool_name,
                "display_name": self.tool_selection.display_name,
                "bound_kwargs": self.tool_selection.bound_kwargs,
                "description": self.tool_selection.description,
            }
        return {
            "question": self.question,
            "status": self.status,
            "identification": asdict(self.identification),
            "extraction": {
                "inputs": self.extraction.inputs,
                "missing_required": self.extraction.missing_required,
                "conversions": self.extraction.conversions,
                "assumptions": self.extraction.assumptions,
                "extracted_evidence": self.extraction.extracted_evidence,
                "warnings": self.extraction.warnings,
            },
            "tool_selection": tool_dict,
            "calculation": self.calculation,
            "verification": asdict(self.verification) if self.verification else None,
            "explanation": self.explanation,
            "pipeline_trace": self.pipeline_trace,
        }


# ---------------------------------------------------------------------------
# Numeric & Time-Unit Parsing Primitives
# ---------------------------------------------------------------------------

_NUM_RE = r"(?:\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|\$?\s*\d+(?:\.\d+)?)"


def _parse_number(raw: str) -> float:
    cleaned = raw.replace("$", "").replace(",", "").replace("%", "").strip()
    return float(cleaned)


def _normalize_time_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    u = unit.strip().lower().rstrip("s")
    if u in ("day", "daily", "business day", "working day", "operating day"):
        return "days"
    if u in ("week", "weekly", "wk"):
        return "weeks"
    if u in ("month", "monthly", "mo"):
        return "months"
    if u in ("quarter", "quarterly"):
        return "quarters"
    if u in ("year", "yearly", "annual", "annually", "yr"):
        return "years"
    return None


def _detect_operating_days(text: str) -> Tuple[float, Optional[str]]:
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:operating|working|business)?\s*days?\s*(?:per|a|/)\s*year",
        text,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1)), f"Detected {m.group(1)} operating days/year."
    return 365.0, None


def _detect_business_week(text: str) -> float:
    if re.search(r"\b5[- ]day\s*(?:week|workweek|business week)\b", text, re.IGNORECASE):
        return 5.0
    return 7.0


def _convert_duration(
    value: float,
    from_unit: str,
    to_unit: str,
    days_per_year: float = 365.0,
    days_per_week: float = 7.0,
) -> float:
    """Convert a time duration between days, weeks, months, quarters, years."""
    if from_unit == to_unit:
        return value
    to_days_map = {
        "days": 1.0,
        "weeks": days_per_week,
        "months": days_per_year / 12.0,
        "quarters": days_per_year / 4.0,
        "years": days_per_year,
    }
    days = value * to_days_map[from_unit]
    return days / to_days_map[to_unit]


def _annualize_rate(
    value: float,
    unit: str,
    days_per_year: float = 365.0,
    weeks_per_year: float = 52.0,
) -> float:
    factors = {
        "years": 1.0,
        "quarters": 4.0,
        "months": 12.0,
        "weeks": weeks_per_year,
        "days": days_per_year,
    }
    return value * factors[unit]


def _search_first(
    patterns: Sequence[str],
    text: str,
) -> Optional[Tuple[re.Match, str]]:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m, pat
    return None


# ---------------------------------------------------------------------------
# Stage 1: Identify Model
# ---------------------------------------------------------------------------

def _has_discount_tier_structure(text: str) -> bool:
    patterns = [
        r"\d[\d,]*\s*(?:-|to|–)\s*\d[\d,]*\s*(?:units?|items?|pcs|pieces)?\s*(?:at|for|:|@|-|=|\()\s*\$?\s*\d",
        r"\$\s*\d[\d,.]*(?:\s*(?:each|per\s+unit|/unit))?\s*(?:for|on|if|when)\s*(?:orders?\s*(?:of\s*)?)?\d[\d,]*\s*(?:-|to|–|\+)",
        r"(?:\d[\d,]*\s*\+|\d[\d,]*\s*(?:or more|and above|and up)|(?:>=|at least|over|above|more than)\s*\d[\d,]*)\s*(?:units?|items?)?\s*(?:at|for|:|@|-|=|get|\()\s*\$?\s*\d",
        r"(?:under|less than|below|up to|first|<|<=)\s*\d[\d,]*\s*(?:units?|items?)?\s*(?:at|for|:|@|-|=|\()\s*\$?\s*\d",
        r"\d+(?:\.\d+)?%\s*(?:discount|off)\s*(?:for|on|when|at)\s*(?:orders?\s*(?:of\s*)?)?\d",
        r"\bquantity\s+breaks?\b",
    ]
    matches = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
    return matches >= 1


def identify_model(question: str) -> ModelIdentification:
    """Classify a natural-language question into one of the five inventory models."""
    text = question.strip()
    scores: Dict[str, float] = {
        "eoq": 0.0,
        "quantity_discount": 0.0,
        "safety_stock_rop": 0.0,
        "fixed_period_review": 0.0,
        "newsvendor": 0.0,
    }
    signals: Dict[str, List[str]] = {k: [] for k in scores}

    def add_signal(model: str, weight: float, label: str) -> None:
        scores[model] += weight
        signals[model].append(label)

    # --- Quantity Discount signals ---
    qd_rules = [
        (r"\bquantity\s+discounts?\b", 5.0, "keyword: quantity discount"),
        (r"\b(?:price|quantity)\s*[- ]?breaks?\b", 5.0, "keyword: price/quantity break"),
        (r"\b(?:discount|pricing)\s+tiers?\b", 4.5, "keyword: pricing/discount tier"),
        (r"\ball[- ]units?\b", 4.0, "keyword: all-units"),
        (r"\bincremental\s+(?:quantity\s+)?discount\b", 4.5, "keyword: incremental discount"),
        (r"\bbulk\s+discount\b|\bvolume\s+discount\b", 4.0, "keyword: bulk/volume discount"),
        (r"\bdiscount\s+schedule\b", 4.0, "keyword: discount schedule"),
    ]
    for pat, w, lbl in qd_rules:
        if re.search(pat, text, re.IGNORECASE):
            add_signal("quantity_discount", w, lbl)
    if _has_discount_tier_structure(text):
        add_signal("quantity_discount", 4.5, "structure: quantity tier price schedule")

    # --- Fixed-Period Review signals ---
    fp_rules = [
        (r"\bfixed[- ]period\b|\bperiodic\s+review\b|\bp[- ]system\b", 5.5, "keyword: fixed-period/periodic review"),
        (r"\border[- ]up[- ]to\b|\btarget\s+(?:inventory\s+)?level\b", 5.0, "keyword: order-up-to level"),
        (r"\breview\s+(?:period|interval|cycle)\b", 4.5, "keyword: review period/interval"),
        (r"\breview(?:ed|s)?\s+(?:inventory\s+)?every\s+\d+", 4.5, "phrase: reviewed every N periods"),
        (r"\bcheck(?:ed|s)?\s+(?:inventory\s+)?every\s+\d+", 3.5, "phrase: checked every N periods"),
        (r"\bon[- ]hand\b.*\bon[- ]order\b|\binventory\s+position\b|\bbackorders?\b", 3.5, "keyword: inventory position / on-hand / on-order / backorders"),
        (r"\b\(T,\s*S\)\b|\bT\s*\+\s*L\b", 4.0, "notation: (T, S) or T+L"),
    ]
    for pat, w, lbl in fp_rules:
        if re.search(pat, text, re.IGNORECASE):
            add_signal("fixed_period_review", w, lbl)

    # --- Newsvendor signals ---
    nv_rules = [
        (r"\bnewsvendor\b|\bnewsboy\b", 6.0, "keyword: newsvendor"),
        (r"\bsingle[- ]period\b|\bone[- ]time\s+(?:order|buy|purchase)\b", 5.0, "keyword: single-period"),
        (r"\bunderage\b|\boverage\b|\bcritical\s+ratio\b", 5.0, "keyword: underage/overage/critical ratio"),
        (r"\b(?:shortage|stockout)\s+cost\b|\bexcess\s+cost\b", 3.0, "keyword: shortage/excess cost"),
        (r"\bsalvage\b|\bleftover\b|\bunsold\b|\bthrown\s+away\b|\bdonated\b", 4.5, "keyword: salvage/leftover/unsold"),
        (r"\bgoodwill\s+(?:cost|penalty)\b", 3.5, "keyword: goodwill penalty"),
        (r"\bselling\s+price\b|\bretail\s+price\b|\bsells?\s+for\b", 2.5, "keyword: selling price"),
        (r"\buniform(?:ly)?\s+distributed\b|\buniform\s*\(", 4.0, "keyword: uniform demand"),
        (r"\bdemand\s+(?:scenarios?|distribution)\b", 3.0, "keyword: demand distribution/scenarios"),
        (r"\bperishable\b|\bseasonal\b|\bbakery\b|\bnewspaper\b", 2.0, "context: perishable/seasonal item"),
    ]
    for pat, w, lbl in nv_rules:
        if re.search(pat, text, re.IGNORECASE):
            add_signal("newsvendor", w, lbl)

    # --- Safety Stock & Reorder Point signals ---
    ss_rules = [
        (r"\bsafety\s+stock\b", 4.0, "keyword: safety stock"),
        (r"\breorder\s+point\b|\brop\b", 5.0, "keyword: reorder point (ROP)"),
        (r"\bcontinuous\s+review\b|\bq[- ]system\b|\b\(Q,\s*R\)\b", 4.5, "keyword: continuous review / (Q, R)"),
        (r"\bcycle\s+service\s+level\b|\bcsl\b|\bservice\s+level\b|\bfill\s+rate\b", 3.0, "keyword: service level / fill rate"),
        (r"\blead[- ]?time\b", 2.5, "keyword: lead time"),
        (r"\b(?:standard\s+deviation|std\.?\s*dev\.?|sigma|normally\s+distributed)\b", 2.0, "keyword: demand variability"),
    ]
    for pat, w, lbl in ss_rules:
        if re.search(pat, text, re.IGNORECASE):
            add_signal("safety_stock_rop", w, lbl)

    # --- EOQ signals ---
    eoq_rules = [
        (r"\beoq\b|\beconomic\s+order\s+quantity\b", 6.0, "keyword: EOQ"),
        (
            r"\b(?:ordering|order|setup|set-up)\s+(?:cost|expense)s?\b"
            r"|\b(?:ordering|order)\s*(?:/|or)\s*setup\s+(?:cost|expense)s?\b"
            r"|\bsetup\s*(?:/|or)\s*(?:ordering|order)\s+(?:cost|expense)s?\b"
            r"|\bcost\s+(?:per\s+order|of\s+placing\s+(?:an\s+)?order|to\s+place\s+(?:an\s+)?order)\b",
            2.5,
            "keyword: ordering/setup cost",
        ),
        (
            r"\b(?:(?:fixed|percentage)\s+)?(?:holding|carrying|storage)\s+(?:cost|rate|expense)s?\b",
            2.5,
            "keyword: holding/carrying cost",
        ),
        (r"\borders?\s+per\s+year\b|\btime\s+between\s+orders\b|\bcycle\s+time\b|\blot\s+size\b", 2.5, "keyword: order frequency / cycle time"),
        (r"\bannual\s+demand\b|\byearly\s+demand\b|\bunits?\s+(?:per|a)\s+year\b", 1.5, "keyword: annual demand"),
    ]
    for pat, w, lbl in eoq_rules:
        if re.search(pat, text, re.IGNORECASE):
            add_signal("eoq", w, lbl)

    # Contextual disambiguation adjustments
    if scores["fixed_period_review"] >= 4.0 and scores["safety_stock_rop"] > 0.0:
        # Periodic review naturally asks for safety stock, service level, and lead time;
        # avoid letting ROP win unless ROP itself is explicitly requested without review period.
        if not re.search(r"\breorder\s+point\b|\brop\b", text, re.IGNORECASE):
            scores["fixed_period_review"] += 2.0

    if scores["quantity_discount"] >= 4.0:
        # Quantity discount questions naturally share ordering cost, holding rate, and annual demand with EOQ.
        scores["quantity_discount"] += 3.5

    if scores["newsvendor"] >= 4.5 and not re.search(r"\blead[- ]?time\b", text, re.IGNORECASE):
        # Stochastic demand in Newsvendor should not be confused with Safety Stock without lead time.
        scores["newsvendor"] += 2.0

    if (
        scores["eoq"] >= 5.0
        and scores["quantity_discount"] < 4.0
        and scores["fixed_period_review"] < 4.0
        and scores["newsvendor"] < 4.0
        and not re.search(
            r"\bsafety\s+stock\b|\bservice\s+level\b|\bcsl\b|\bfill\s+rate\b|\bstandard\s+deviation\b|\bstd\.?\s*dev\.?\b|\bsigma\b",
            text,
            re.IGNORECASE,
        )
    ):
        # EOQ problems with deterministic lead time often ask for both order quantity and deterministic ROP.
        scores["eoq"] += 4.5

    best_model = max(scores, key=lambda k: scores[k])
    best_score = scores[best_model]
    total_score = sum(scores.values())

    if best_score < 2.0 or total_score == 0.0:
        return ModelIdentification(
            model_type="unknown",
            model_label=MODEL_LABELS["unknown"],
            confidence=0.0,
            scores=scores,
            matched_signals=signals,
            reason=(
                "Could not confidently map the question to EOQ, Quantity Discount, "
                "Safety Stock & Reorder Point, Fixed-Period Review, or Newsvendor."
            ),
        )

    sorted_scores = sorted(scores.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_score
    confidence = min(0.99, max(0.35, 0.55 + 0.08 * margin + 0.03 * best_score))

    matched_str = ", ".join(signals[best_model]) if signals[best_model] else "keyword match"
    reason = f"Matched {MODEL_LABELS[best_model]} based on {matched_str}."

    return ModelIdentification(
        model_type=best_model,
        model_label=MODEL_LABELS[best_model],
        confidence=round(confidence, 3),
        scores={k: round(v, 2) for k, v in scores.items()},
        matched_signals=signals,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Stage 2: Extract Inputs
# ---------------------------------------------------------------------------

def _extract_demand_annual(
    text: str,
    days_per_year: float,
    extraction: InputExtraction,
) -> Optional[float]:
    patterns = [
        # "annual demand is 12,000 units", "yearly usage of 10,000"
        (rf"\b(annual|yearly)\s+(?:demand|usage|sales|requirement)\s*(?:is|of|=|:)?\s*({_NUM_RE})", "years"),
        # "demand is 500 units per month" / "sells 500 widgets per month"
        (rf"(?:demand|usage|sales|sells?|needs?|uses?)\s*(?:is|of|=|:|averages?)?\s*({_NUM_RE})\s*(?:[a-z]+\s+){{0,2}}?(?:per|a|/)\s*(day|week|month|quarter|year)\b", None),
        # "500 units per month" / "40 units/day"
        (rf"({_NUM_RE})\s*(?:units?|widgets?|parts?|items?|pieces?)\s*(?:per|a|/)\s*(day|week|month|quarter|year)\b", None),
        # "daily demand is 40", "monthly demand of 500"
        (rf"\b(daily|weekly|monthly|quarterly|annual|yearly)\s+(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_RE})", "prefix"),
        # "demand of 5,000 units" / "demand = 5000"
        (rf"\bdemand\s*(?:is|of|=|:)?\s*({_NUM_RE})\s*(?:units?|widgets?|parts?|items?)?(?:\s*(?:per|a|/)\s*(year|month|week|day|quarter))?", "tail"),
        # "5,000 parts a year" / "2,400 units a year"
        (rf"({_NUM_RE})\s*(?:units?|widgets?|parts?|items?|pieces?)\s+(?:a|per)\s*(year|month|week|day|quarter)\b", None),
    ]

    for pat, mode in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        if mode == "years":
            val = _parse_number(m.group(2))
            unit = "years"
        elif mode == "prefix":
            unit = _normalize_time_unit(m.group(1)) or "years"
            val = _parse_number(m.group(2))
        elif mode == "tail":
            val = _parse_number(m.group(1))
            unit = _normalize_time_unit(m.group(2)) or "years"
            if m.group(2) is None:
                extraction.assumptions.append("Assumed stated demand is annual (per year).")
        else:
            val = _parse_number(m.group(1))
            unit = _normalize_time_unit(m.group(2)) or "years"

        annual_val = _annualize_rate(val, unit, days_per_year=days_per_year)
        extraction.extracted_evidence["demand"] = m.group(0)
        if unit != "years":
            extraction.conversions.append(
                f"Converted demand from {val:g} units/{unit[:-1]} to {annual_val:g} units/year."
            )
        return annual_val
    return None


def _extract_ordering_cost(text: str, extraction: InputExtraction) -> Optional[float]:
    patterns = [
        rf"(?:fixed\s+)?(?:(?:ordering|order)\s*(?:/|or)\s*setup|ordering|order|setup|set-up)\s+(?:cost|expense)s?\s*(?:S)?\s*(?:is|of|=|:|at)?\s*({_NUM_RE})",
        rf"(?:cost|expense)\s+(?:(?:per|for\s+each|for\s+an)\s+(?:order|setup)|of\s+placing\s+(?:an\s+)?order|to\s+place\s+(?:an\s+)?order)\s*(?:is|of|=|:|at)?\s*({_NUM_RE})",
        rf"(?:placing\s+an\s+order|each\s+order|each\s+setup|every\s+order)\s+costs?\s*({_NUM_RE})",
        rf"({_NUM_RE})\s+(?:cost\s+per\s+order|cost\s+of\s+placing\s+an\s+order|ordering\s*/\s*setup\s+cost|ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense)",
        rf"costs?\s+({_NUM_RE})\s*(?:to\s+place(?:\s+an\s+order)?|per\s+order|per\s+setup|each\s+order|to\s+set\s*up)",
        rf"({_NUM_RE})\s*(?:per\s+order|per\s+setup|/order|/setup)\b",
    ]
    res = _search_first(patterns, text)
    if res:
        m, _ = res
        val = _parse_number(m.group(1))
        extraction.extracted_evidence["ordering_cost"] = m.group(0)
        return val
    return None


def _extract_unit_cost_and_holding(
    text: str,
    days_per_year: float,
    extraction: InputExtraction,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    unit_cost: Optional[float] = None
    holding_cost: Optional[float] = None
    holding_rate: Optional[float] = None
    rate_span: Optional[Tuple[int, int]] = None

    # Unit cost / unit purchase price patterns (avoid matching ordering/holding/underage/overage cost)
    unit_patterns = [
        rf"(?:unit\s+purchase\s+(?:price|cost)|unit\s+(?:cost|price)|purchase\s+(?:cost|price)|item\s+cost|regular\s+price|base\s+(?:unit\s+)?price)\s*(?:C)?\s*(?:is|of|=|:)?\s*({_NUM_RE})",
        rf"\beach\s+(?!order\b|setup\b)\w+\s+costs?\s+({_NUM_RE})",
        rf"(?<!order\s)(?<!setup\s)costs?\s+({_NUM_RE})\s*(?:each|per\s+unit|/unit)",
        rf"({_NUM_RE})\s*(?:per\s+unit|each)\b(?!\s*(?:per\s+year|per\s+month|/\s*year|/\s*month))",
    ]
    res_u = _search_first(unit_patterns, text)
    if res_u:
        m, _ = res_u
        unit_cost = _parse_number(m.group(1))
        extraction.extracted_evidence["unit_cost"] = m.group(0)

    # Percentage holding rate (e.g. "percentage holding cost is 20%", "carrying cost is 25% of the $16 unit cost")
    rate_patterns = [
        r"(?:percentage\s+(?:holding|carrying)\s+(?:cost|rate|expense)|(?:holding|carrying|inventory\s+carrying)\s+(?:cost\s+)?(?:rate|percentage))\s*(?:i)?\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)\s*%",
        r"(?:holding|carrying)\s+(?:cost|expense)?\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)\s*%\s*(?:per\s+year|annually|of\s+(?:the\s+)?(?:unit\s+)?(?:purchase\s+)?(?:price|cost))?",
        r"(\d+(?:\.\d+)?)\s*%\s*(?:annual\s+)?(?:holding|carrying)\s+(?:cost|rate|expense)",
    ]
    res_r = _search_first(rate_patterns, text)
    if res_r:
        m, _ = res_r
        holding_rate = float(m.group(1)) / 100.0
        rate_span = (m.start(), m.end())
        extraction.extracted_evidence["holding_rate"] = m.group(0)
    else:
        dec_rate = re.search(
            r"(?:percentage\s+(?:holding|carrying)\s+(?:cost|rate|expense)|(?:holding|carrying)\s+(?:cost\s+)?(?:rate|percentage))\s*(?:i)?\s*(?:is|of|=|:)?\s*(?!\$)(0\.\d+|\d+(?:\.\d+)?)\b(?!\s*(?:per\s+unit|/unit|\$))",
            text,
            re.IGNORECASE,
        )
        if dec_rate:
            raw_r = float(dec_rate.group(1))
            holding_rate = raw_r / 100.0 if raw_r > 1.0 else raw_r
            rate_span = (dec_rate.start(), dec_rate.end())
            extraction.extracted_evidence["holding_rate"] = dec_rate.group(0)

    # Also catch "$16 unit cost" / "$16 unit purchase price" in phrases like "25% of the $16 unit cost"
    if unit_cost is None:
        m_uc = re.search(rf"({_NUM_RE})\s+unit\s+(?:purchase\s+)?(?:cost|price)", text, re.IGNORECASE)
        if m_uc:
            unit_cost = _parse_number(m_uc.group(1))
            extraction.extracted_evidence["unit_cost"] = m_uc.group(0)

    # Explicit fixed dollar/numeric holding cost per unit per period (exclude percentage holding cost matches)
    hold_patterns = [
        rf"(?:fixed\s+)?(?:annual\s+)?(?:holding|carrying|storage|inventory\s+holding)\s+(?:cost|expense)\s*(?:H)?\s*(?:(?:per|/)?\s*unit(?:\s*(?:per|/)\s*year)?)?\s*(?:is|of|=|:)?\s*\$?\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)(?!\s*%)(?:\s*(?:per\s+unit\s*|/unit\s*)?(?:per|a|/)\s*(day|week|month|quarter|year))?",
        rf"\$?\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)(?!\s*%)\s*(?:per\s+unit\s+per\s+|/unit/|/unit\s+per\s+|/unit\s+a\s+)(day|week|month|quarter|year)\s*(?:to\s+hold|holding|carrying|storage)?",
    ]
    for pat in hold_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            if rate_span and not (m.end() <= rate_span[0] or m.start() >= rate_span[1]):
                continue
            prefix = text[max(0, m.start() - 20):m.start()].lower()
            if "percentage" in prefix:
                continue
            raw_h = _parse_number(m.group(1))
            h_unit = _normalize_time_unit(m.group(2)) or "years"
            holding_cost = _annualize_rate(raw_h, h_unit, days_per_year=days_per_year)
            extraction.extracted_evidence["holding_cost"] = m.group(0)
            if h_unit != "years":
                extraction.conversions.append(
                    f"Converted holding cost from ${raw_h:g}/unit/{h_unit[:-1]} to ${holding_cost:g}/unit/year."
                )
            break
        if holding_cost is not None:
            break

    return unit_cost, holding_cost, holding_rate


def _extract_discount_tiers(
    text: str,
    extraction: InputExtraction,
    base_unit_cost: Optional[float] = None,
) -> List[Dict[str, Any]]:
    tiers: List[Dict[str, Any]] = []
    seen_spans: List[Tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in seen_spans)

    # 1. Explicit range before price: "1-999 at $5.00", "1 to 999 units: $5.00", "0-499 ($10)"
    range_pat = re.compile(
        rf"(\d[\d,]*)\s*(?:-|–|to)\s*(\d[\d,]*)\s*(?:units?|items?|pcs|pieces)?\s*(?:at|for|:|@|-|=|\(|each\s+at)?\s*\$\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)",
        re.IGNORECASE,
    )
    for m in range_pat.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        seen_spans.append((m.start(), m.end()))
        tiers.append(
            {
                "min_qty": _parse_number(m.group(1)),
                "max_qty": _parse_number(m.group(2)),
                "unit_cost": _parse_number(m.group(3)),
            }
        )

    # 1b. Price before explicit range: "$20 per unit for 1-499 units"
    price_range_pat = re.compile(
        rf"\$\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)\s*(?:each|per\s+unit|/unit)?\s*(?:for|on|if|when)\s*(?:orders?\s*(?:of\s*)?)?(\d[\d,]*)\s*(?:-|–|to)\s*(\d[\d,]*)\s*(?:units?|items?|pcs|pieces)?",
        re.IGNORECASE,
    )
    for m in price_range_pat.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        seen_spans.append((m.start(), m.end()))
        tiers.append(
            {
                "min_qty": _parse_number(m.group(2)),
                "max_qty": _parse_number(m.group(3)),
                "unit_cost": _parse_number(m.group(1)),
            }
        )

    # 2. Upper bound before price: "under 500 at $12", "less than 500: $12", "up to 499: $12"
    upper_pat = re.compile(
        rf"\b(under|less\s+than|below|up\s+to|first|<=|<)\s*(\d[\d,]*)\s*(?:units?|items?|pcs|pieces)?\s*(?:at|for|:|@|-|=|\(|cost|are)?\s*\$\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)",
        re.IGNORECASE,
    )
    for m in upper_pat.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        seen_spans.append((m.start(), m.end()))
        kw = m.group(1).lower()
        bound = _parse_number(m.group(2))
        max_qty = bound - 1.0 if kw in ("under", "less than", "below", "<") else bound
        tiers.append(
            {
                "min_qty": 1.0,
                "max_qty": max(1.0, max_qty),
                "unit_cost": _parse_number(m.group(3)),
            }
        )

    # 2b. Price before upper bound: "$12 for orders under 500 units"
    price_upper_pat = re.compile(
        rf"\$\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)\s*(?:each|per\s+unit|/unit)?\s*(?:for|on|if|when)\s*(?:orders?\s*(?:of\s*)?)?(under|less\s+than|below|up\s+to|<=|<)\s*(\d[\d,]*)\s*(?:units?|items?|pcs|pieces)?",
        re.IGNORECASE,
    )
    for m in price_upper_pat.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        seen_spans.append((m.start(), m.end()))
        kw = m.group(2).lower()
        bound = _parse_number(m.group(3))
        max_qty = bound - 1.0 if kw in ("under", "less than", "below", "<") else bound
        tiers.append(
            {
                "min_qty": 1.0,
                "max_qty": max(1.0, max_qty),
                "unit_cost": _parse_number(m.group(1)),
            }
        )

    # 3. Lower bound before price: "1000+ at $4.75", "1,000 or more at $4.75", "above 500 units $9"
    lower_pats = [
        (
            re.compile(
                rf"(\d[\d,]*)\s*(?:\+|or\s+more|and\s+above|and\s+up)\s*(?:units?|items?|pcs|pieces)?\s*(?:at|for|:|@|-|=|\(|each\s+at|get)?\s*\$\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            False,
        ),
        (
            re.compile(
                rf"\b(>=|at\s+least|over|above|more\s+than|>)\s*(\d[\d,]*)\s*(?:units?|items?|pcs|pieces)?\s*(?:at|for|:|@|-|=|\(|cost|get)?\s*\$\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            True,
        ),
    ]
    for pat, has_kw in lower_pats:
        for m in pat.finditer(text):
            if overlaps(m.start(), m.end()):
                continue
            seen_spans.append((m.start(), m.end()))
            if has_kw:
                kw = m.group(1).lower()
                qty = _parse_number(m.group(2))
                min_qty = qty + 1.0 if kw in ("over", "above", "more than", ">") else qty
                price = _parse_number(m.group(3))
            else:
                min_qty = _parse_number(m.group(1))
                price = _parse_number(m.group(2))
            tiers.append({"min_qty": min_qty, "max_qty": None, "unit_cost": price})

    # 3b. Price before lower bound: "$9 per unit for 500+ units", "$8.50 for at least 1,000"
    price_lower_pats = [
        (
            re.compile(
                rf"\$\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)\s*(?:each|per\s+unit|/unit)?\s*(?:for|on|if|when)\s*(?:orders?\s*(?:of\s*)?)?(\d[\d,]*)\s*(?:\+|or\s+more|and\s+above|and\s+up)\s*(?:units?|items?|pcs|pieces)?",
                re.IGNORECASE,
            ),
            False,
        ),
        (
            re.compile(
                rf"\$\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)\s*(?:each|per\s+unit|/unit)?\s*(?:for|on|if|when)\s*(?:orders?\s*(?:of\s*)?)?(>=|at\s+least|over|above|more\s+than|>)\s*(\d[\d,]*)\s*(?:units?|items?|pcs|pieces)?",
                re.IGNORECASE,
            ),
            True,
        ),
    ]
    for pat, has_kw in price_lower_pats:
        for m in pat.finditer(text):
            if overlaps(m.start(), m.end()):
                continue
            seen_spans.append((m.start(), m.end()))
            price = _parse_number(m.group(1))
            if has_kw:
                kw = m.group(2).lower()
                qty = _parse_number(m.group(3))
                min_qty = qty + 1.0 if kw in ("over", "above", "more than", ">") else qty
            else:
                min_qty = _parse_number(m.group(2))
            tiers.append({"min_qty": min_qty, "max_qty": None, "unit_cost": price})

    # 4. Percentage discounts relative to base unit price:
    #    e.g. "regular price $20; 500+ get 10% discount; 1000+ get 15% off"
    if not tiers:
        base_m = re.search(
            rf"(?:base|regular|standard|list|normal|unit\s+purchase)\s+(?:unit\s+)?(?:price|cost)\s*(?:is|of|=|:)?\s*\$\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        base_price = _parse_number(base_m.group(1)) if base_m else base_unit_cost
        if base_price is not None:
            disc_matches = list(
                re.finditer(
                    r"(\d[\d,]*)\s*(?:\+|or\s+more|and\s+above)\s*(?:units?|items?)?\s*(?:get|receive|have|:|-)?\s*(?:a\s+)?(\d+(?:\.\d+)?)\s*%\s*(?:discount|off)",
                    text,
                    re.IGNORECASE,
                )
            )
            if not disc_matches:
                disc_matches = list(
                    re.finditer(
                        r"(\d+(?:\.\d+)?)\s*%\s*(?:discount|off)\s*(?:for|on|at|when)?\s*(?:orders?\s*(?:of\s*)?)?(\d[\d,]*)\s*(?:\+|or\s+more)",
                        text,
                        re.IGNORECASE,
                    )
                )
                parsed_disc = [
                    (_parse_number(dm.group(2)), float(dm.group(1)) / 100.0)
                    for dm in disc_matches
                ]
            else:
                parsed_disc = [
                    (_parse_number(dm.group(1)), float(dm.group(2)) / 100.0)
                    for dm in disc_matches
                ]

            if parsed_disc:
                parsed_disc.sort(key=lambda x: x[0])
                tiers.append({"min_qty": 1.0, "max_qty": parsed_disc[0][0] - 1.0, "unit_cost": base_price})
                for idx, (q_thr, pct) in enumerate(parsed_disc):
                    nxt_max = parsed_disc[idx + 1][0] - 1.0 if idx + 1 < len(parsed_disc) else None
                    tiers.append(
                        {
                            "min_qty": q_thr,
                            "max_qty": nxt_max,
                            "unit_cost": round(base_price * (1.0 - pct), 6),
                        }
                    )

    # 5. Explicit "quantity break" phrasing with threshold -> dollar price
    if len(tiers) < 2:
        qb_pat = re.compile(
            r"(?:quantity|price)\s+breaks?\s*(?:at|of|=|:)?\s*(\d[\d,]*)\s*(?:units?|items?)?[^.\n]{0,50}?\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
            re.IGNORECASE,
        )
        for m in qb_pat.finditer(text):
            if overlaps(m.start(), m.end()):
                continue
            snippet = m.group(0).lower()
            if any(w in snippet for w in ("ordering", "setup", "holding", "carrying")):
                continue
            seen_spans.append((m.start(), m.end()))
            tiers.append(
                {
                    "min_qty": _parse_number(m.group(1)),
                    "max_qty": None,
                    "unit_cost": _parse_number(m.group(2)),
                }
            )

    if tiers:
        tiers.sort(key=lambda t: (t["min_qty"], -t["unit_cost"]))
        if base_unit_cost is not None and tiers[0]["min_qty"] > 1.0:
            if not math.isclose(base_unit_cost, tiers[0]["unit_cost"], rel_tol=1e-9, abs_tol=1e-9):
                tiers.insert(
                    0,
                    {
                        "min_qty": 1.0,
                        "max_qty": tiers[0]["min_qty"] - 1.0,
                        "unit_cost": base_unit_cost,
                    },
                )
        # Fill implicit max_qty between consecutive tiers if missing
        for i in range(len(tiers) - 1):
            if tiers[i]["max_qty"] is None and tiers[i + 1]["min_qty"] > tiers[i]["min_qty"]:
                tiers[i]["max_qty"] = tiers[i + 1]["min_qty"] - 1.0
        extraction.extracted_evidence["tiers"] = f"{len(tiers)} price-break tiers extracted"
    return tiers


def _extract_stochastic_demand(
    text: str,
    extraction: InputExtraction,
    excluded_spans: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[Optional[float], Optional[float], str]:
    """Extract (avg_demand, demand_std, demand_unit) for ROP / Periodic Review."""
    # Pattern 1: N(mean, std) or Normal(mean, std)
    m_norm = re.search(
        r"(?:(daily|weekly|monthly|yearly|annual)\s+)?(?:demand\s*(?:is|=|:)?\s*)?N(?:ormal)?\s*\(\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*,\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*\)",
        text,
        re.IGNORECASE,
    )
    if m_norm:
        unit = _normalize_time_unit(m_norm.group(1))
        if not unit:
            unit_ctx = re.search(
                r"\b(daily|weekly|monthly|yearly|annual)\s+demand\b|\bper\s+(day|week|month|year)\b|/(day|week|month|year)\b",
                text,
                re.IGNORECASE,
            )
            unit = (
                _normalize_time_unit(unit_ctx.group(1) or unit_ctx.group(2) or unit_ctx.group(3))
                if unit_ctx
                else "days"
            )
        mean_val = _parse_number(m_norm.group(2))
        std_val = _parse_number(m_norm.group(3))
        extraction.extracted_evidence["demand_distribution"] = m_norm.group(0)
        return mean_val, std_val, unit

    avg_demand: Optional[float] = None
    demand_std: Optional[float] = None
    demand_unit: Optional[str] = None
    spans = excluded_spans or []

    def _overlaps_excluded(s: int, e: int) -> bool:
        return any(not (e <= es or s >= ee) for es, ee in spans)

    # Avg demand patterns (handle "average demand", "daily demand", "constant demand", etc.)
    avg_pats = [
        (
            rf"\b(daily|weekly|monthly|yearly|annual)\s+(?:average\s+|mean\s+)?demand\s+(?:averages?|is|of|mean\s*(?:is|of|=)?|=|:)?\s*({_NUM_RE})",
            "unit_val",
        ),
        (
            rf"\bdemand\s+(?:averages?|has\s+(?:a\s+)?mean(?:\s+of)?|mean\s*(?:is|of|=)?|is\s+normally\s+distributed\s+with\s+mean(?:\s+of)?)\s*({_NUM_RE})\s*(?:units?|items?)?\s*(?:(?:per|/)\s*(day|week|month|year))?",
            "val_unit",
        ),
        (
            rf"\b(?:average|avg\.?|mean)\s+(?:(daily|weekly|monthly|yearly|annual)\s+)?demand\s*(?:is|of|=|:)?\s*({_NUM_RE})\s*(?:units?|items?)?\s*(?:(?:per|/)\s*(day|week|month|year))?",
            "u_v_u",
        ),
        (
            rf"\b(?:constant|fixed|deterministic)\s+(?:(daily|weekly|monthly|yearly|annual)\s+)?demand\s*(?:is|of|=|:)?\s*({_NUM_RE})\s*(?:units?|items?)?\s*(?:(?:per|/)\s*(day|week|month|year))?",
            "u_v_u",
        ),
        (
            rf"\b(?:(daily|weekly|monthly|yearly|annual)\s+)?demand\s+is\s+(?:constant|fixed|deterministic)(?:\s+at)?\s*({_NUM_RE})\s*(?:units?|items?)?\s*(?:(?:per|/)\s*(day|week|month|year))?",
            "u_v_u",
        ),
        (
            rf"\bdemand\s*(?:is|of|=|:)?\s*({_NUM_RE})\s*(?:units?|items?)?\s*(?:/|\s*per\s*)(day|week|month|year)\b",
            "val_unit",
        ),
    ]
    for pat, mode in avg_pats:
        m = re.search(pat, text, re.IGNORECASE)
        if not m or _overlaps_excluded(m.start(), m.end()):
            continue
        if mode == "unit_val":
            demand_unit = _normalize_time_unit(m.group(1))
            avg_demand = _parse_number(m.group(2))
        elif mode == "val_unit":
            avg_demand = _parse_number(m.group(1))
            demand_unit = _normalize_time_unit(m.group(2))
        else:
            demand_unit = _normalize_time_unit(m.group(1) or m.group(3))
            avg_demand = _parse_number(m.group(2))
        extraction.extracted_evidence["avg_demand"] = m.group(0)
        break

    # Demand standard deviation:
    # 1. Explicit demand standard deviation patterns first
    explicit_std_pats = [
        (rf"\b(?:(daily|weekly|monthly|yearly|annual)\s+)?demand\s+(?:standard\s+deviation|std\.?\s*dev\.?|sd|sigma)\s*(?:is|of|=|:)?\s*({_NUM_RE})", 1, 2),
        (rf"\b(?:standard\s+deviation|std\.?\s*dev\.?|sd|sigma)\s+(?:of|for)\s+(?:the\s+)?(?:(daily|weekly|monthly|yearly|annual)\s+)?demand\s*(?:is|of|=|:)?\s*({_NUM_RE})", 1, 2),
    ]
    for pat, u_idx, v_idx in explicit_std_pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m and not _overlaps_excluded(m.start(), m.end()):
            demand_std = _parse_number(m.group(v_idx))
            if demand_unit is None and m.group(u_idx):
                demand_unit = _normalize_time_unit(m.group(u_idx))
            extraction.extracted_evidence["demand_std"] = m.group(0)
            break

    # 2. Clause-aware fallback for generic "standard deviation" / "sigma" outside lead-time spans
    if demand_std is None:
        generic_std_pats = [
            rf"(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s*(?:is|of|=|:)?\s*({_NUM_RE})",
            rf"\b(?:std\.?|sd)\s*(?:is|of|=|:)?\s*({_NUM_RE})",
        ]
        for pat in generic_std_pats:
            for m in re.finditer(pat, text, re.IGNORECASE):
                if _overlaps_excluded(m.start(), m.end()):
                    continue
                suffix = text[m.end():m.end() + 35].lower()
                if re.match(r"^\s*(?:days?|weeks?|months?|years?)?\s*(?:of|for)\s+(?:the\s+)?lead", suffix):
                    continue
                sent_start = max(
                    text.rfind(".", 0, m.start()),
                    text.rfind(";", 0, m.start()),
                    text.rfind("\n", 0, m.start()),
                ) + 1
                sent_prefix = text[sent_start:m.start()].lower()
                lt_pos = max(sent_prefix.rfind("lead time"), sent_prefix.rfind("lead-time"))
                dem_pos = sent_prefix.rfind("demand")
                if lt_pos > dem_pos:
                    continue
                demand_std = _parse_number(m.group(1))
                extraction.extracted_evidence["demand_std"] = m.group(0)
                break
            if demand_std is not None:
                break

    if demand_std is None and re.search(
        r"\b(?:constant|fixed|deterministic)\s+(?:daily|weekly|monthly|yearly|annual\s+)?demand\b|\bdemand\s+is\s+(?:constant|fixed|deterministic)\b",
        text,
        re.IGNORECASE,
    ):
        demand_std = 0.0
        extraction.extracted_evidence["demand_std"] = "constant demand (0)"

    if demand_unit is None:
        unit_ctx = re.search(
            r"\b(daily|weekly|monthly|yearly|annual)\b|\bper\s+(day|week|month|year)\b|/(day|week|month|year)\b",
            text,
            re.IGNORECASE,
        )
        if unit_ctx:
            demand_unit = _normalize_time_unit(unit_ctx.group(1) or unit_ctx.group(2) or unit_ctx.group(3))
    if demand_unit is None:
        demand_unit = "days"
        extraction.assumptions.append("Assumed demand rate is measured per day.")

    return avg_demand, demand_std, demand_unit


def _extract_lead_time_and_review_with_spans(
    text: str,
    extraction: InputExtraction,
) -> Tuple[Optional[float], str, float, str, Optional[float], str, List[Tuple[int, int]]]:
    lead_time: Optional[float] = None
    lt_unit = "days"
    lead_time_std = 0.0
    lt_std_unit = "days"
    review_period: Optional[float] = None
    rev_unit = "days"
    lt_spans: List[Tuple[int, int]] = []

    lt_pats = [
        rf"(?:average|avg\.?|mean|replenishment|fixed|constant)?\s*lead[- ]?time\s*(?:L)?\s*(?:averages?|has\s+(?:a\s+)?mean(?:\s+of)?|mean\s*(?:is|of|=)?|is|of|=|:)?\s*({_NUM_RE})\s*(days?|weeks?|months?|years?)?",
        rf"\b(?:average|avg\.?|mean)\s+(?:of\s+)?(?:the\s+)?lead[- ]?time\s*(?:is|of|=|:)?\s*({_NUM_RE})\s*(days?|weeks?|months?|years?)?",
        rf"\bL\s*=\s*({_NUM_RE})\s*(days?|weeks?|months?|years?)?",
    ]
    for pat in lt_pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            lead_time = _parse_number(m.group(1))
            lt_unit = _normalize_time_unit(m.group(2)) or "days"
            lt_spans.append((m.start(), m.end()))
            extraction.extracted_evidence["lead_time"] = m.group(0)
            break

    lt_std_pats = [
        rf"lead[- ]?time\s+(?!demand\b)(?:standard\s+deviation|std\.?\s*dev\.?|sd|sigma)\s*(?:is|of|=|:)?\s*({_NUM_RE})\s*(days?|weeks?|months?|years?)?",
        rf"(?:standard\s+deviation|std\.?\s*dev\.?|sd|sigma)\s+(?:of|for)\s+(?:the\s+)?lead[- ]?time\s*(?:is|of|=|:)?\s*({_NUM_RE})\s*(days?|weeks?|months?|years?)?",
        rf"lead[- ]?time[^.;\n]*?\b(?:and|with)\s+(?:a\s+)?(?:standard\s+deviation|std\.?\s*dev\.?|sd|sigma)\s*(?:of|is|=|:)?\s*({_NUM_RE})\s*(days?|weeks?|months?|years?)?",
    ]
    for pat in lt_std_pats:
        m_lts = re.search(pat, text, re.IGNORECASE)
        if m_lts:
            if "demand" in m_lts.group(0).lower():
                continue
            lead_time_std = _parse_number(m_lts.group(1))
            lt_std_unit = _normalize_time_unit(m_lts.group(2)) or lt_unit
            lt_spans.append((m_lts.start(), m_lts.end()))
            extraction.extracted_evidence["lead_time_std"] = m_lts.group(0)
            break

    if "lead_time_std" not in extraction.extracted_evidence and re.search(
        r"\b(?:constant|fixed|deterministic)\s+lead[- ]?time\b|\blead[- ]?time\s+is\s+(?:constant|fixed|deterministic)\b",
        text,
        re.IGNORECASE,
    ):
        extraction.extracted_evidence["lead_time_std"] = "constant lead time (0)"

    rev_pats = [
        rf"review\s*(?:period|interval|cycle)\s*(?:T)?\s*(?:is|of|=|:)?\s*({_NUM_RE})\s*(days?|weeks?|months?|years?)?",
        rf"(?:review(?:ed|s)?|check(?:ed|s)?)\s+(?:inventory\s+)?every\s+({_NUM_RE})\s*(days?|weeks?|months?|years?)?",
        rf"\bT\s*=\s*({_NUM_RE})\s*(days?|weeks?|months?|years?)?",
    ]
    for pat in rev_pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            review_period = _parse_number(m.group(1))
            rev_unit = _normalize_time_unit(m.group(2)) or lt_unit
            extraction.extracted_evidence["review_period"] = m.group(0)
            break

    if review_period is None:
        if re.search(r"\breviewed\s+weekly\b", text, re.IGNORECASE):
            review_period, rev_unit = 1.0, "weeks"
            extraction.extracted_evidence["review_period"] = "reviewed weekly"
        elif re.search(r"\breviewed\s+daily\b", text, re.IGNORECASE):
            review_period, rev_unit = 1.0, "days"
            extraction.extracted_evidence["review_period"] = "reviewed daily"
        elif re.search(r"\breviewed\s+monthly\b", text, re.IGNORECASE):
            review_period, rev_unit = 1.0, "months"
            extraction.extracted_evidence["review_period"] = "reviewed monthly"

    return lead_time, lt_unit, lead_time_std, lt_std_unit, review_period, rev_unit, lt_spans


def _extract_lead_time_and_review(
    text: str,
    extraction: InputExtraction,
) -> Tuple[Optional[float], str, float, str, Optional[float], str]:
    lt, lt_u, lts, lts_u, rev, rev_u, _ = _extract_lead_time_and_review_with_spans(text, extraction)
    return lt, lt_u, lts, lts_u, rev, rev_u


def _extract_service_target(
    text: str,
    extraction: InputExtraction,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    service_level: Optional[float] = None
    z_score: Optional[float] = None
    fill_rate: Optional[float] = None
    order_quantity: Optional[float] = None

    # Fill rate (Type-2 service level)
    fr_m = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:type[- ]?2\s+)?fill\s*rate|fill\s*rate\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(%?)",
        text,
        re.IGNORECASE,
    )
    if fr_m:
        if fr_m.group(1):
            fill_rate = float(fr_m.group(1)) / 100.0
        else:
            raw = float(fr_m.group(2))
            fill_rate = raw / 100.0 if (fr_m.group(3) == "%" or raw > 1.0) else raw
        extraction.extracted_evidence["fill_rate"] = fr_m.group(0)

    # Order quantity Q (used with fill rate)
    q_m = re.search(
        rf"(?:order\s+quantity|lot\s+size|batch\s+size|\bQ\b)\s*(?:is|of|=|:)?\s*({_NUM_RE})",
        text,
        re.IGNORECASE,
    )
    if q_m:
        order_quantity = _parse_number(q_m.group(1))
        extraction.extracted_evidence["order_quantity"] = q_m.group(0)

    # Cycle service level
    csl_pats = [
        r"(\d+(?:\.\d+)?)\s*%\s*(?:cycle\s+service\s+level|service\s+level|\bCSL\b|in[- ]stock\s+probability|service\s+probability)",
        r"(?:cycle\s+service\s+level|\bCSL\b|service\s+level|in[- ]stock\s+probability|service\s+probability)\s*(?:target\s*)?(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)\s*(%?)",
    ]
    for idx, pat in enumerate(csl_pats):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = float(m.group(1))
            if idx == 0 or (len(m.groups()) > 1 and m.group(2) == "%") or raw > 1.0:
                service_level = raw / 100.0
            else:
                service_level = raw
            extraction.extracted_evidence["service_level"] = m.group(0)
            break

    # Direct z-score
    z_m = re.search(r"\bz[- ]?(?:score|value)?\s*(?:is|of|=|:)?\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if z_m and service_level is None and fill_rate is None:
        z_score = float(z_m.group(1))
        extraction.extracted_evidence["z_score"] = z_m.group(0)

    return service_level, z_score, fill_rate, order_quantity


def _extract_newsvendor_inputs(text: str, extraction: InputExtraction) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    special_spans: List[Tuple[int, int]] = []

    def _in_special(s: int, e: int) -> bool:
        return any(not (e <= ss or s >= se) for ss, se in special_spans)

    # Underage / shortage cost
    cu_pats = [
        rf"(?:underage\s*(?:/|or)\s*shortage|shortage\s*(?:/|or)\s*underage|underage|shortage|stockout)\s+cost\s*(?:Cu)?\s*(?:is|are|of|at|=|:)?\s*({_NUM_RE})",
        rf"cost\s+of\s+(?:underage|shortage|stockout)\s*(?:Cu)?\s*(?:is|are|of|at|=|:)?\s*({_NUM_RE})",
        rf"\bCu\s*=\s*({_NUM_RE})",
    ]
    res_cu = _search_first(cu_pats, text)
    if res_cu:
        m_cu, _ = res_cu
        inputs["underage_cost"] = _parse_number(m_cu.group(1))
        special_spans.append((m_cu.start(), m_cu.end()))
        extraction.extracted_evidence["underage_cost"] = m_cu.group(0)

    # Overage / excess cost
    co_pats = [
        rf"(?:overage\s*(?:/|or)\s*excess|excess\s*(?:/|or)\s*overage|overage|excess)\s+cost\s*(?:Co)?\s*(?:is|are|of|at|=|:)?\s*({_NUM_RE})",
        rf"cost\s+of\s+(?:overage|excess)\s*(?:Co)?\s*(?:is|are|of|at|=|:)?\s*({_NUM_RE})",
        rf"\bCo\s*=\s*({_NUM_RE})",
    ]
    res_co = _search_first(co_pats, text)
    if res_co:
        m_co, _ = res_co
        inputs["overage_cost"] = _parse_number(m_co.group(1))
        special_spans.append((m_co.start(), m_co.end()))
        extraction.extracted_evidence["overage_cost"] = m_co.group(0)

    # Goodwill penalty
    goodwill_m = re.search(
        rf"goodwill\s+(?:cost|penalty|loss)\s*(?:is|of|at|=|:)?\s*({_NUM_RE})",
        text,
        re.IGNORECASE,
    )
    if goodwill_m:
        inputs["goodwill_penalty"] = _parse_number(goodwill_m.group(1))
        special_spans.append((goodwill_m.start(), goodwill_m.end()))
        extraction.extracted_evidence["goodwill_penalty"] = goodwill_m.group(0)

    # Salvage value
    salvage_pats = [
        rf"(?:salvage|scrap|liquidation|markdown|discount|donated)\s+(?:value|price)?\s*(?:is|of|for|at|=|:)?\s*({_NUM_RE})",
        rf"({_NUM_RE})\s+salvage\s+(?:value|price)",
        rf"(?:unsold|leftover)\s+[\w\s]{{0,60}}?(?:for|at|worth)\s*({_NUM_RE})",
    ]
    res_s = _search_first(salvage_pats, text)
    if res_s:
        m_s, _ = res_s
        inputs["salvage_value"] = _parse_number(m_s.group(1))
        special_spans.append((m_s.start(), m_s.end()))
        extraction.extracted_evidence["salvage_value"] = m_s.group(0)
    elif re.search(r"\b(?:zero|no)\s+salvage\s+value\b|\bworthless\b|\bthrown\s+away\b", text, re.IGNORECASE):
        inputs["salvage_value"] = 0.0
        extraction.extracted_evidence["salvage_value"] = "zero salvage value"

    # Selling price
    price_pats = [
        rf"(?:unit\s+)?(?:selling|retail|sale)\s+price\s*(?:is|are|of|for|at|=|:)?\s*({_NUM_RE})",
        rf"(?:sells?|sold|retails?)\s+(?:[\w-]+\s+){{0,3}}?(?:for|at)\s*({_NUM_RE})\s*(?:each|per\s+unit)?",
    ]
    res_p = _search_first(price_pats, text)
    if res_p and not _in_special(res_p[0].start(), res_p[0].end()):
        inputs["selling_price"] = _parse_number(res_p[0].group(1))
        special_spans.append((res_p[0].start(), res_p[0].end()))
        extraction.extracted_evidence["selling_price"] = res_p[0].group(0)

    # Purchase / unit cost
    cost_pats = [
        rf"(?:(?:unit\s+)?purchase|(?:unit|manufacturing|production|wholesale))\s+(?:cost|price)\s*(?:is|are|of|for|at|=|:)?\s*({_NUM_RE})",
        rf"(?<!underage\s)(?<!overage\s)(?<!shortage\s)(?<!excess\s)(?<!goodwill\s)(?<!holding\s)(?<!ordering\s)(?:buys?|purchases?|makes?|cost(?:s)?\s+to\s+make)\s+(?:[\w-]+\s+){{0,3}}?(?:at|for|is|of)?\s*\$\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)",
        rf"that\s+cost\s*\$\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)\s*(?:to\s+make|each)?",
    ]
    for pat in cost_pats:
        for m_c in re.finditer(pat, text, re.IGNORECASE):
            if _in_special(m_c.start(), m_c.end()):
                continue
            inputs["unit_cost"] = _parse_number(m_c.group(1))
            extraction.extracted_evidence["unit_cost"] = m_c.group(0)
            break
        if "unit_cost" in inputs:
            break

    if "salvage_value" not in inputs and "selling_price" in inputs and "unit_cost" in inputs:
        inputs["salvage_value"] = 0.0
        extraction.assumptions.append("Assumed salvage value is $0.00 (none specified).")

    # Demand distribution: Discrete pairs e.g. "(100: 0.2, 150: 0.3)", "100 (20%)", or "100 with probability 0.2"
    disc_colon = re.findall(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*:\s*(0?\.\d+|1(?:\.0+)?)\b", text)
    disc_pct = re.findall(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*\(\s*(\d+(?:\.\d+)?)\s*%\s*\)", text)
    disc_prob = re.findall(
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:units?\s*)?(?:with\s+prob(?:ability)?(?:\s+of)?|prob(?:ability)?\s*=)\s*(0?\.\d+|1(?:\.0+)?|\d+(?:\.\d+)?\s*%)",
        text,
        re.IGNORECASE,
    )
    if len(disc_colon) >= 2:
        inputs["demand_distribution"] = [(_parse_number(d), float(p)) for d, p in disc_colon]
        extraction.extracted_evidence["demand_distribution"] = f"{len(disc_colon)} discrete scenarios"
        return inputs
    if len(disc_pct) >= 2:
        inputs["demand_distribution"] = [(_parse_number(d), float(p) / 100.0) for d, p in disc_pct]
        extraction.extracted_evidence["demand_distribution"] = f"{len(disc_pct)} discrete percentage scenarios"
        return inputs
    if len(disc_prob) >= 2:
        pairs = []
        for d, p_raw in disc_prob:
            p_val = float(p_raw.replace("%", "").strip()) / 100.0 if "%" in p_raw else float(p_raw)
            pairs.append((_parse_number(d), p_val))
        inputs["demand_distribution"] = pairs
        extraction.extracted_evidence["demand_distribution"] = f"{len(disc_prob)} discrete probability scenarios"
        return inputs

    # Uniform distribution
    uni_m = re.search(
        rf"(?:uniform(?:ly)?\s+distributed\s+(?:between|from)|uniform\s*\()\s*({_NUM_RE})\s*(?:and|to|,)\s*({_NUM_RE})",
        text,
        re.IGNORECASE,
    )
    if uni_m:
        inputs["demand_min"] = _parse_number(uni_m.group(1))
        inputs["demand_max"] = _parse_number(uni_m.group(2))
        inputs["demand_distribution"] = "uniform"
        extraction.extracted_evidence["demand_distribution"] = uni_m.group(0)
        return inputs

    # Normal distribution
    m_norm = re.search(
        r"N(?:ormal)?\s*\(\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*,\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*\)",
        text,
        re.IGNORECASE,
    )
    if m_norm:
        inputs["demand_mean"] = _parse_number(m_norm.group(1))
        inputs["demand_std"] = _parse_number(m_norm.group(2))
        inputs["demand_distribution"] = "normal"
        extraction.extracted_evidence["demand_distribution"] = m_norm.group(0)
        return inputs

    mean_pats = [
        rf"(?:mean|average|expected)(?:\s+demand)?\s*(?:is|of|=|:)?\s*({_NUM_RE})",
        rf"demand\s+(?:averages?|has\s+(?:a\s+)?mean(?:\s+of)?|mean\s*(?:is|of|=)?)\s*({_NUM_RE})",
    ]
    res_m = _search_first(mean_pats, text)
    if res_m:
        inputs["demand_mean"] = _parse_number(res_m[0].group(1))
        extraction.extracted_evidence["demand_mean"] = res_m[0].group(0)

    std_m = re.search(
        rf"(?:demand\s+)?(?:standard\s+deviation|std\.?\s*dev\.?|sd|sigma)(?:\s+of\s+demand)?\s*(?:is|of|=|:)?\s*({_NUM_RE})",
        text,
        re.IGNORECASE,
    )
    if std_m:
        inputs["demand_std"] = _parse_number(std_m.group(1))
        extraction.extracted_evidence["demand_std"] = std_m.group(0)

    if re.search(r"\bnormal(?:ly)?\b", text, re.IGNORECASE) or (
        "demand_mean" in inputs and "demand_std" in inputs
    ):
        inputs["demand_distribution"] = "normal"

    return inputs


def extract_inputs(question: str, model_type: str) -> InputExtraction:
    """Extract and normalize quantitative parameters for the identified model."""
    text = question.strip()
    extraction = InputExtraction(inputs={})

    if model_type == "unknown":
        extraction.missing_required.append("supported_inventory_model")
        return extraction

    days_per_year, op_note = _detect_operating_days(text)
    if op_note:
        extraction.assumptions.append(op_note)
    days_per_week = _detect_business_week(text)

    if model_type == "eoq":
        demand = _extract_demand_annual(text, days_per_year, extraction)
        ordering_cost = _extract_ordering_cost(text, extraction)
        unit_cost, holding_cost, holding_rate = _extract_unit_cost_and_holding(
            text, days_per_year, extraction
        )

        if demand is not None:
            extraction.inputs["demand"] = demand
        else:
            extraction.missing_required.append("demand (annual demand)")

        if ordering_cost is not None:
            extraction.inputs["ordering_cost"] = ordering_cost
        else:
            extraction.missing_required.append("ordering_cost ($/order)")

        if holding_cost is not None:
            extraction.inputs["holding_cost"] = holding_cost
        if unit_cost is not None:
            extraction.inputs["unit_cost"] = unit_cost
        if holding_rate is not None:
            extraction.inputs["holding_rate"] = holding_rate

        if holding_cost is None and not (unit_cost is not None and holding_rate is not None):
            extraction.missing_required.append(
                "holding_cost ($/unit/year) or (unit_cost + holding_rate %)"
            )

        lt_m = re.search(
            rf"lead\s+time\s*(?:is|of|=|:)?\s*({_NUM_RE})\s*(days?|weeks?)",
            text,
            re.IGNORECASE,
        )
        if lt_m:
            lt_val = _parse_number(lt_m.group(1))
            lt_unit = _normalize_time_unit(lt_m.group(2)) or "days"
            extraction.inputs["lead_time_days"] = _convert_duration(
                lt_val, lt_unit, "days", days_per_year=days_per_year, days_per_week=days_per_week
            )
            extraction.extracted_evidence["lead_time_days"] = lt_m.group(0)

        extraction.inputs["operating_days_per_year"] = days_per_year
        return extraction

    if model_type == "quantity_discount":
        demand = _extract_demand_annual(text, days_per_year, extraction)
        ordering_cost = _extract_ordering_cost(text, extraction)
        unit_cost, holding_cost, holding_rate = _extract_unit_cost_and_holding(
            text, days_per_year, extraction
        )
        tiers = _extract_discount_tiers(text, extraction, base_unit_cost=unit_cost)

        if demand is not None:
            extraction.inputs["demand"] = demand
        else:
            extraction.missing_required.append("demand (annual demand)")

        if ordering_cost is not None:
            extraction.inputs["ordering_cost"] = ordering_cost
        else:
            extraction.missing_required.append("ordering_cost ($/order)")

        if holding_rate is not None:
            extraction.inputs["holding_rate"] = holding_rate
        elif holding_cost is not None:
            extraction.inputs["holding_cost"] = holding_cost
        else:
            extraction.missing_required.append("holding_rate (% of unit price) or fixed holding_cost ($/unit/year)")

        if tiers:
            extraction.inputs["tiers"] = tiers
        else:
            extraction.missing_required.append("tiers (quantity price breaks and unit prices)")

        if re.search(r"\bincremental\b", text, re.IGNORECASE):
            extraction.inputs["discount_type"] = "incremental"
        else:
            extraction.inputs["discount_type"] = "all_units"
            extraction.assumptions.append("Applied all-units quantity discount schedule.")

        return extraction

    if model_type in ("safety_stock_rop", "fixed_period_review"):
        lead_time, lt_unit, lt_std, lt_std_unit, review_period, rev_unit, lt_spans = (
            _extract_lead_time_and_review_with_spans(text, extraction)
        )
        avg_demand, demand_std, demand_unit = _extract_stochastic_demand(
            text, extraction, excluded_spans=lt_spans
        )
        service_level, z_score, fill_rate, order_quantity = _extract_service_target(text, extraction)

        ss_explicit: Optional[float] = None
        if model_type == "fixed_period_review":
            ss_pats = [
                rf"(?:explicit\s+|desired\s+|required\s+|fixed\s+|policy\s+)?(?:safety|buffer)\s+stock\s*(?:SS)?\s*(?:level\s*)?(?:is|of|=|:|kept\s+at|set\s+at)\s*({_NUM_RE})\s*(?:units?|items?)?",
                rf"({_NUM_RE})\s*(?:units?|items?)\s*(?:of\s+)?(?:safety|buffer)\s+stock",
                rf"\bSS\s*=\s*({_NUM_RE})",
            ]
            res_ss = _search_first(ss_pats, text)
            if res_ss:
                ss_explicit = _parse_number(res_ss[0].group(1))
                extraction.inputs["safety_stock"] = ss_explicit
                extraction.extracted_evidence["safety_stock"] = res_ss[0].group(0)

        if avg_demand is not None:
            extraction.inputs["avg_demand"] = avg_demand
        else:
            extraction.missing_required.append("avg_demand (mean demand per period)")

        if demand_std is not None:
            extraction.inputs["demand_std"] = demand_std
        elif ss_explicit is not None:
            pass
        elif lt_std == 0.0:
            extraction.missing_required.append("demand_std (demand standard deviation)")
        else:
            extraction.inputs["demand_std"] = 0.0
            if "demand_std" not in extraction.extracted_evidence:
                extraction.assumptions.append(
                    "Assumed demand is constant (demand_std = 0) because only lead-time variability was specified."
                )

        if lead_time is not None:
            conv_lt = _convert_duration(
                lead_time, lt_unit, demand_unit, days_per_year=days_per_year, days_per_week=days_per_week
            )
            extraction.inputs["lead_time"] = conv_lt
            if lt_unit != demand_unit:
                extraction.conversions.append(
                    f"Converted lead time from {lead_time:g} {lt_unit} to {conv_lt:g} {demand_unit}."
                )
        else:
            extraction.missing_required.append("lead_time")

        if lt_std > 0.0:
            conv_lt_std = _convert_duration(
                lt_std, lt_std_unit, demand_unit, days_per_year=days_per_year, days_per_week=days_per_week
            )
            extraction.inputs["lead_time_std"] = conv_lt_std
            if lt_std_unit != demand_unit:
                extraction.conversions.append(
                    f"Converted lead-time std dev from {lt_std:g} {lt_std_unit} to {conv_lt_std:g} {demand_unit}."
                )
        else:
            extraction.inputs["lead_time_std"] = 0.0

        extraction.inputs["time_unit"] = demand_unit

        is_deterministic = (
            extraction.inputs.get("demand_std") == 0.0
            and extraction.inputs.get("lead_time_std") == 0.0
        )

        if model_type == "fixed_period_review":
            if review_period is not None:
                conv_rev = _convert_duration(
                    review_period, rev_unit, demand_unit, days_per_year=days_per_year, days_per_week=days_per_week
                )
                extraction.inputs["review_period"] = conv_rev
                if rev_unit != demand_unit:
                    extraction.conversions.append(
                        f"Converted review period from {review_period:g} {rev_unit} to {conv_rev:g} {demand_unit}."
                    )
            else:
                extraction.missing_required.append("review_period (T)")

            # Optional current inventory levels for computing order quantity Q = max(0, S - IP)
            ip_m = re.search(
                rf"(?:(?:current\s+)?inventory\s+position\s*(?:\(\s*IP\s*\))?\s*(?:is|are|of|at|=|:)?|\bIP\s*(?:is|of|=|:))\s*({_NUM_RE})",
                text,
                re.IGNORECASE,
            )
            oh_m = re.search(
                rf"(?:({_NUM_RE})\s*(?:units?\s*)?(?:currently\s+)?on[- ]hand|on[- ]hand\s*(?:inventory|stock)?\s*(?:is|are|were|of|=|:)?\s*({_NUM_RE})|(?:zero|no)\s+(?:units?\s*)?(?:of\s+)?on[- ]hand\s*(?:inventory|stock)?)",
                text,
                re.IGNORECASE,
            )
            oo_m = re.search(
                rf"(?:({_NUM_RE})\s*(?:units?\s*)?(?:currently\s+)?on[- ]order|on[- ]order\s*(?:inventory|stock)?\s*(?:is|are|were|of|=|:)?\s*({_NUM_RE})|(?:zero|no)\s+(?:units?\s*)?(?:of\s+)?on[- ]order\s*(?:inventory|stock)?)",
                text,
                re.IGNORECASE,
            )
            bo_m = re.search(
                rf"(?:({_NUM_RE})\s*(?:units?\s*)?backorder(?:ed|s)?|backorders?\s*(?:inventory|level)?\s*(?:is|are|were|of|=|:)?\s*({_NUM_RE})|(?:zero|no)\s+(?:units?\s*)?backorders?)",
                text,
                re.IGNORECASE,
            )
            if ip_m:
                extraction.inputs["inventory_position"] = _parse_number(ip_m.group(1))
                extraction.extracted_evidence["inventory_position"] = ip_m.group(0)
            if oh_m:
                val = oh_m.group(1) or oh_m.group(2)
                extraction.inputs["on_hand"] = _parse_number(val) if val is not None else 0.0
                extraction.extracted_evidence["on_hand"] = oh_m.group(0)
            if oo_m:
                val = oo_m.group(1) or oo_m.group(2)
                extraction.inputs["on_order"] = _parse_number(val) if val is not None else 0.0
                extraction.extracted_evidence["on_order"] = oo_m.group(0)
            if bo_m:
                val = bo_m.group(1) or bo_m.group(2)
                extraction.inputs["backorders"] = _parse_number(val) if val is not None else 0.0
                extraction.extracted_evidence["backorders"] = bo_m.group(0)

            if service_level is not None:
                extraction.inputs["service_level"] = service_level
            elif z_score is not None:
                extraction.inputs["z_score"] = z_score
            elif ss_explicit is None and not is_deterministic:
                extraction.missing_required.append("service_level (e.g. 95%) or z_score")

        else:  # safety_stock_rop
            if fill_rate is not None and service_level is None and z_score is None:
                extraction.inputs["fill_rate"] = fill_rate
                if order_quantity is not None:
                    extraction.inputs["order_quantity"] = order_quantity
                else:
                    extraction.missing_required.append("order_quantity (Q, required for fill rate)")
            elif service_level is not None:
                extraction.inputs["service_level"] = service_level
                if order_quantity is not None:
                    extraction.inputs["order_quantity"] = order_quantity
            elif z_score is not None:
                extraction.inputs["z_score"] = z_score
                if order_quantity is not None:
                    extraction.inputs["order_quantity"] = order_quantity
            elif not is_deterministic:
                extraction.missing_required.append("service_level (e.g. 95%), fill_rate, or z_score")

        return extraction

    if model_type == "newsvendor":
        nv_inputs = _extract_newsvendor_inputs(text, extraction)
        extraction.inputs.update(nv_inputs)

        has_direct_costs = "underage_cost" in nv_inputs and "overage_cost" in nv_inputs
        has_price_cost = "selling_price" in nv_inputs and "unit_cost" in nv_inputs
        if not (has_direct_costs or has_price_cost):
            extraction.missing_required.append(
                "(underage_cost and overage_cost) or (selling_price and unit_cost)"
            )

        has_normal = "demand_mean" in nv_inputs and "demand_std" in nv_inputs
        has_uniform = "demand_min" in nv_inputs and "demand_max" in nv_inputs
        has_discrete = "demand_distribution" in nv_inputs
        if not (has_normal or has_uniform or has_discrete):
            extraction.missing_required.append(
                "demand distribution: Normal (mean & std), Uniform (min & max), or Discrete scenarios"
            )

        return extraction

    return extraction


# ---------------------------------------------------------------------------
# Stage 3: Select Python Tool
# ---------------------------------------------------------------------------

def select_python_tool(model_type: str, inputs: Dict[str, Any]) -> ToolSelection:
    """Bind the identified model and validated inputs to its Python tool function."""
    if model_type not in TOOL_REGISTRY:
        raise ValueError(f"No Python tool registered for model_type '{model_type}'.")

    spec = TOOL_REGISTRY[model_type]
    accepted_keys = set(spec["required_params"]) | set(spec["optional_params"])
    for group in spec["required_any_of"]:
        for item in group:
            if isinstance(item, tuple):
                accepted_keys.update(item)
            else:
                accepted_keys.add(item)

    bound_kwargs = {k: v for k, v in inputs.items() if k in accepted_keys and v is not None}

    return ToolSelection(
        model_type=model_type,
        tool_name=spec["tool_name"],
        display_name=spec["display_name"],
        function=spec["function"],
        bound_kwargs=bound_kwargs,
        description=spec["description"],
    )


# ---------------------------------------------------------------------------
# Stage 4: Calculate
# ---------------------------------------------------------------------------

def calculate(tool_selection: ToolSelection) -> Dict[str, Any]:
    """Execute the selected Python tool function for all numerical calculations."""
    return tool_selection.function(**tool_selection.bound_kwargs)


# ---------------------------------------------------------------------------
# Stage 5: Verify
# ---------------------------------------------------------------------------

def verify(
    model_type: str,
    inputs: Dict[str, Any],
    result: Dict[str, Any],
) -> VerificationResult:
    """Run independent mathematical sanity checks and merge Python tool verification checks."""
    checks: List[VerificationCheck] = []
    seen_names: set = set()
    warnings: List[str] = []

    def add_check(chk: VerificationCheck) -> None:
        if chk.name not in seen_names:
            checks.append(chk)
            seen_names.add(chk.name)

    for raw_chk in result.get("verification_checks", []):
        add_check(
            VerificationCheck(
                name=str(raw_chk.get("name", "tool_check")),
                passed=bool(raw_chk.get("passed", False)),
                detail=str(raw_chk.get("detail", "")),
            )
        )

    if model_type == "eoq":
        q_star = result["eoq"]
        d = result["demand"]
        s = result["ordering_cost"]
        h = result["holding_cost"]

        ord_c = result["annual_ordering_cost"]
        hold_c = result["annual_holding_cost"]
        balanced = math.isclose(ord_c, hold_c, rel_tol=1e-6, abs_tol=1e-6)
        add_check(
            VerificationCheck(
                name="eoq_cost_balance",
                passed=balanced,
                detail=f"Annual ordering cost (${ord_c:,.2f}) == holding cost (${hold_c:,.2f}).",
            )
        )

        # Neighbor optimality check at 0.9*Q* and 1.1*Q*
        tc_star = result["total_relevant_cost"]
        tc_low = (d / (0.9 * q_star)) * s + (0.9 * q_star / 2.0) * h
        tc_high = (d / (1.1 * q_star)) * s + (1.1 * q_star / 2.0) * h
        is_min = tc_star < tc_low and tc_star < tc_high
        add_check(
            VerificationCheck(
                name="eoq_local_minimum",
                passed=is_min,
                detail=f"Cost at Q* (${tc_star:,.2f}) < neighbors 0.9Q* (${tc_low:,.2f}) and 1.1Q* (${tc_high:,.2f}).",
            )
        )

        positive_ok = q_star > 0 and result["orders_per_year"] > 0 and result["cycle_time_days"] > 0
        add_check(
            VerificationCheck(
                name="positive_quantities",
                passed=positive_ok,
                detail=f"EOQ={q_star:.2f}, orders/yr={result['orders_per_year']:.2f}, cycle={result['cycle_time_days']:.2f} days.",
            )
        )

    elif model_type == "quantity_discount":
        evals = result["tier_evaluations"]
        best_cost = result["total_annual_cost"]
        min_cost_among_tiers = min(e["total_annual_cost"] for e in evals)
        global_min_ok = math.isclose(best_cost, min_cost_among_tiers, rel_tol=1e-9, abs_tol=1e-9)
        add_check(
            VerificationCheck(
                name="global_minimum_across_tiers",
                passed=global_min_ok,
                detail=f"Selected tier {result['optimal_tier_index'] + 1} achieves lowest total cost (${best_cost:,.2f}).",
            )
        )

        chosen_tier = evals[result["optimal_tier_index"]]
        q_opt = result["optimal_order_quantity"]
        in_bounds = q_opt + 1e-9 >= chosen_tier["min_qty"] and (
            chosen_tier["max_qty"] is None or q_opt - 1e-9 <= chosen_tier["max_qty"]
        )
        add_check(
            VerificationCheck(
                name="tier_quantity_feasibility",
                passed=in_bounds,
                detail=(
                    f"Order quantity Q*={q_opt:,.2f} lies within tier bounds "
                    f"[{chosen_tier['min_qty']:g}, {chosen_tier['max_qty'] if chosen_tier['max_qty'] is not None else 'inf'}]."
                ),
            )
        )

        cost_sum = (
            result["annual_purchase_cost"]
            + result["annual_ordering_cost"]
            + result["annual_holding_cost"]
        )
        sum_ok = math.isclose(cost_sum, best_cost, rel_tol=1e-6, abs_tol=1e-6)
        add_check(
            VerificationCheck(
                name="cost_component_identity",
                passed=sum_ok,
                detail=f"Purchase + ordering + holding = ${cost_sum:,.2f} == total cost (${best_cost:,.2f}).",
            )
        )

    elif model_type == "safety_stock_rop":
        if result.get("service_criterion") == "deterministic":
            det_ok = math.isclose(result["safety_stock"], 0.0, abs_tol=1e-9)
            add_check(
                VerificationCheck(
                    name="deterministic_safety_stock",
                    passed=det_ok,
                    detail=f"Deterministic regime ({result.get('demand_regime')}): SS = {result['safety_stock']:.2f}.",
                )
            )
        else:
            z = result["z_score"]
            csl = result["cycle_service_level"]
            csl_ok = math.isclose(norm_cdf(z), csl, rel_tol=1e-6, abs_tol=1e-6)
            add_check(
                VerificationCheck(
                    name="normal_cdf_consistency",
                    passed=csl_ok,
                    detail=f"Phi(z={z:.4f}) = {norm_cdf(z):.4f} matches CSL = {csl:.4f}.",
                )
            )

        rop_expected = result["lead_time_demand_mean"] + result["safety_stock"]
        rop_ok = math.isclose(result["reorder_point"], rop_expected, rel_tol=1e-8, abs_tol=1e-8)
        add_check(
            VerificationCheck(
                name="rop_identity",
                passed=rop_ok,
                detail=f"ROP ({result['reorder_point']:.2f}) == mu_LT ({result['lead_time_demand_mean']:.2f}) + SS ({result['safety_stock']:.2f}).",
            )
        )

        if result["target_fill_rate"] is not None and result["implied_fill_rate"] is not None:
            fr_ok = math.isclose(
                result["target_fill_rate"], result["implied_fill_rate"], rel_tol=1e-4, abs_tol=1e-4
            )
            add_check(
                VerificationCheck(
                    name="fill_rate_solver_accuracy",
                    passed=fr_ok,
                    detail=f"Implied fill rate ({result['implied_fill_rate']:.4%}) matches target ({result['target_fill_rate']:.4%}).",
                )
            )

    elif model_type == "fixed_period_review":
        p_int = result["protection_interval"]
        p_ok = math.isclose(p_int, result["review_period"] + result["lead_time"], rel_tol=1e-9)
        add_check(
            VerificationCheck(
                name="protection_period_identity",
                passed=p_ok,
                detail=f"Protection period T + L = {p_int:g} {result['time_unit']}.",
            )
        )

        s_expected = result["protection_demand_mean"] + result["safety_stock"]
        s_ok = math.isclose(result["order_up_to_level"], s_expected, rel_tol=1e-8, abs_tol=1e-8)
        add_check(
            VerificationCheck(
                name="target_inventory_identity",
                passed=s_ok,
                detail=f"Target inventory S ({result['order_up_to_level']:.2f}) == mu_{{T+L}} ({result['protection_demand_mean']:.2f}) + SS ({result['safety_stock']:.2f}).",
            )
        )

        if result.get("service_criterion") == "deterministic":
            det_ok = math.isclose(result["safety_stock"], 0.0, abs_tol=1e-9)
            add_check(
                VerificationCheck(
                    name="deterministic_safety_stock",
                    passed=det_ok,
                    detail=f"Deterministic regime ({result.get('demand_regime')}): SS = {result['safety_stock']:.2f}.",
                )
            )
        elif result.get("z_score") is not None and result.get("cycle_service_level") is not None:
            z = result["z_score"]
            csl_ok = math.isclose(norm_cdf(z), result["cycle_service_level"], rel_tol=1e-6, abs_tol=1e-6)
            add_check(
                VerificationCheck(
                    name="normal_cdf_consistency",
                    passed=csl_ok,
                    detail=f"Phi(z={z:.4f}) = {norm_cdf(z):.4f} matches CSL = {result['cycle_service_level']:.4f}.",
                )
            )
        else:
            add_check(
                VerificationCheck(
                    name="explicit_safety_stock_consistency",
                    passed=result["safety_stock"] >= 0.0,
                    detail=f"Explicit safety stock SS = {result['safety_stock']:.2f} applied directly.",
                )
            )

        if result["order_quantity"] is not None and result["inventory_position"] is not None:
            q_check = max(0.0, result["order_up_to_level"] - result["inventory_position"])
            q_ok = math.isclose(result["order_quantity"], q_check, rel_tol=1e-8)
            add_check(
                VerificationCheck(
                    name="replenishment_order_identity",
                    passed=q_ok,
                    detail=f"Order Q ({result['order_quantity']:.2f}) == max(0, S - IP).",
                )
            )

    elif model_type == "newsvendor":
        cu = result["underage_cost"]
        co = result["overage_cost"]
        cr = result["critical_ratio"]
        cr_ok = math.isclose(cr, cu / (cu + co), rel_tol=1e-9) and (0.0 < cr < 1.0)
        add_check(
            VerificationCheck(
                name="critical_ratio_bounds",
                passed=cr_ok,
                detail=f"Critical ratio CR = Cu/(Cu+Co) = {cu:g}/({cu:g}+{co:g}) = {cr:.4f}.",
            )
        )

        q_opt = result["optimal_order_quantity"]
        flow_ok = math.isclose(
            result["expected_sales"] + result["expected_leftover"],
            q_opt,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        add_check(
            VerificationCheck(
                name="inventory_conservation",
                passed=flow_ok,
                detail=(
                    f"E[sales] ({result['expected_sales']:.2f}) + E[leftover] ({result['expected_leftover']:.2f}) "
                    f"== Q* ({q_opt:.2f})."
                ),
            )
        )

        if result["distribution"] == "normal" and result["z_score"] is not None:
            # Verify analytical normal optimal mismatch cost: (Cu + Co) * sigma * phi(z*)
            sigma = result["demand_std"]
            z = result["z_score"]
            if q_opt > 0.0:
                closed_form_mismatch = (cu + co) * sigma * norm_pdf(z)
                mismatch_ok = math.isclose(
                    result["expected_mismatch_cost"], closed_form_mismatch, rel_tol=1e-5, abs_tol=1e-5
                )
                add_check(
                    VerificationCheck(
                        name="normal_optimality_identity",
                        passed=mismatch_ok,
                        detail=(
                            f"Expected mismatch cost (${result['expected_mismatch_cost']:,.2f}) matches "
                            f"(Cu+Co)*sigma*phi(z*) (${closed_form_mismatch:,.2f})."
                        ),
                    )
                )
            else:
                warnings.append("Optimal Q* was clamped at 0 because mu + z*sigma <= 0.")

    passed_all = bool(checks) and all(c.passed for c in checks)
    return VerificationResult(passed=passed_all, checks=checks, warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 6: Explain
# ---------------------------------------------------------------------------

def _format_trace_value(value: Any) -> str:
    """Format a Python tool output value for human-readable display without altering calculations."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value)):,}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        if not value:
            return "[]"
        # Price-break tiers or tier evaluations
        if all(isinstance(item, dict) for item in value):
            parts: List[str] = []
            for idx, item in enumerate(value):
                if "tier_index" in item:
                    hi = f"{item['max_qty']:,.0f}" if item.get("max_qty") is not None else "inf"
                    parts.append(
                        f"Tier {item['tier_index'] + 1} [{item['min_qty']:,.0f}–{hi}] "
                        f"@ ${item['unit_cost']:,.2f}: Q={_format_trace_value(item.get('candidate_qty'))} "
                        f"({item.get('feasibility')}), total=${_format_trace_value(item.get('total_annual_cost'))}"
                    )
                elif "min_qty" in item and "unit_cost" in item:
                    hi = f"{item['max_qty']:,.0f}" if item.get("max_qty") is not None else "inf"
                    parts.append(f"[{item['min_qty']:,.0f}–{hi}] @ ${item['unit_cost']:,.2f}")
                else:
                    parts.append(", ".join(f"{k}={_format_trace_value(v)}" for k, v in item.items()))
            return " | ".join(parts)
        if all(isinstance(item, (tuple, list)) and len(item) == 2 for item in value):
            return ", ".join(f"{_format_trace_value(d)} (p={_format_trace_value(p)})" for d, p in value)
        return ", ".join(_format_trace_value(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_format_trace_value(v)}" for k, v in value.items() if v is not None)
    return str(value)


def _append_mapping_section(lines: List[str], heading: str, data: Any, *, include_none: bool = False) -> None:
    """Append a structured dictionary or list section verbatim from Python tool output."""
    lines.append(f"\n**{heading}:**")
    if isinstance(data, dict):
        shown = 0
        for k, v in data.items():
            if v is None and not include_none:
                continue
            lines.append(f"- `{k}`: {_format_trace_value(v)}")
            shown += 1
        if shown == 0:
            lines.append("- None")
    elif isinstance(data, list):
        if not data:
            lines.append("- None")
        else:
            for item in data:
                lines.append(f"- {_format_trace_value(item)}")
    elif data is not None:
        lines.append(f"- {_format_trace_value(data)}")
    else:
        lines.append("- None")


def explain(
    question: str,
    identification: ModelIdentification,
    extraction: InputExtraction,
    tool_selection: Optional[ToolSelection],
    calculation: Optional[Dict[str, Any]],
    verification: Optional[VerificationResult],
) -> str:
    """Format a clear, 8-section verifiable explanation grounded strictly in Python results."""
    lines: List[str] = []

    if identification.model_type == "unknown":
        lines.append("**Model Identification:** Unable to classify question into a supported inventory model.")
        lines.append(
            "Supported models: (1) Economic Order Quantity (EOQ), (2) Quantity Discount, "
            "(3) Safety Stock & Reorder Point, (4) Fixed-Period Review, and (5) Newsvendor."
        )
        lines.append("Please rephrase with your demand, cost, lead-time, or service-level parameters.")
        return "\n".join(lines)

    lines.append(
        f"**Identified Model:** {identification.model_label} "
        f"(confidence: {identification.confidence:.0%})"
    )
    lines.append(f"**Why:** {identification.reason}")

    if not extraction.is_complete or calculation is None:
        _append_mapping_section(lines, "1. Inputs Extracted (Partial)", extraction.inputs)
        if extraction.conversions:
            lines.append("\n**Unit Conversions:**")
            for conv in extraction.conversions:
                lines.append(f"- {conv}")
        if extraction.assumptions:
            lines.append("\n**8. Assumptions Applied:**")
            for asm in extraction.assumptions:
                lines.append(f"- {asm}")
        lines.append("\n**Missing Required Inputs:**")
        for item in extraction.missing_required:
            lines.append(f"- {item}")
        lines.append("\nPlease provide the missing parameters so I can complete the Python calculation.")
        return "\n".join(lines)

    if tool_selection is not None:
        lines.append(f"\n**Python Tool Executed:** `{tool_selection.tool_name}()`")

    # 1. Inputs Extracted
    _append_mapping_section(lines, "1. Inputs Extracted", calculation.get("inputs_extracted", extraction.inputs))
    if extraction.conversions:
        lines.append("  *Unit Conversions:*")
        for conv in extraction.conversions:
            lines.append(f"  - {conv}")

    # 2. Variable Definitions
    _append_mapping_section(lines, "2. Variable Definitions", calculation.get("variable_definitions", {}))

    # 3. Formula
    _append_mapping_section(lines, "3. Formula", calculation.get("formula", {}))

    # 4. Formula with Substituted Values
    _append_mapping_section(lines, "4. Formula with Substituted Values", calculation.get("formula_substituted", {}))

    # 5. Intermediate Calculations
    _append_mapping_section(lines, "5. Intermediate Calculations", calculation.get("intermediate_calculations", {}))

    # 6. Final Answer (strictly from Python tool output)
    _append_mapping_section(lines, "6. Final Answer", calculation.get("final_answer", {}))

    model = identification.model_type
    if model == "eoq":
        lines.append(
            f"\n**Recommendation:** Order **{calculation['rounded_eoq']:,} units** "
            f"(exact EOQ: {calculation['eoq']:,.2f} units) per replenishment "
            f"roughly every **{calculation['cycle_time_days']:,.1f} days**."
        )
    elif model == "quantity_discount":
        lines.append("\n**Tier Evaluation Summary:**")
        for t in calculation["tier_evaluations"]:
            hi = f"{t['max_qty']:,.0f}" if t["max_qty"] is not None else "inf"
            lines.append(
                f"- Tier {t['tier_index'] + 1} [{t['min_qty']:,.0f}–{hi}] @ ${t['unit_cost']:,.2f}: "
                f"candidate Q = {t['candidate_qty']:,.2f} ({t['feasibility']}), "
                f"total cost = ${t['total_annual_cost']:,.2f}"
            )
        lines.append(
            f"\n**Recommendation:** Order **{calculation['rounded_order_quantity']:,} units** "
            f"(Tier {calculation['optimal_tier_index'] + 1} at **${calculation['optimal_unit_cost']:,.2f}/unit**), "
            f"achieving minimum total annual cost of **${calculation['total_annual_cost']:,.2f}** and saving "
            f"**${calculation['annual_savings_vs_base']:,.2f}/year** versus the base price tier."
        )
    elif model == "safety_stock_rop":
        unit = calculation["time_unit"]
        lines.append(
            f"\n**Recommendation:** Maintain **{calculation['rounded_safety_stock']:,} units** of safety stock "
            f"and trigger replenishment whenever inventory position drops to **{calculation['rounded_reorder_point']:,} units** "
            f"(lead time: {calculation['lead_time']:g} {unit})."
        )
    elif model == "fixed_period_review":
        unit = calculation["time_unit"]
        rec = (
            f"\n**Recommendation:** Review stock every **{calculation['review_period']:g} {unit}** and order up to "
            f"**{calculation['rounded_order_up_to_level']:,} units** "
            f"(safety stock: {calculation['safety_stock']:,.2f} units)."
        )
        if calculation["order_quantity"] is not None and calculation["rounded_order_quantity"] is not None:
            rec += f" Place an immediate order of **{calculation['rounded_order_quantity']:,} units**."
        lines.append(rec)
    elif model == "newsvendor":
        lines.append(
            f"\n**Recommendation:** Stock **{calculation['rounded_order_quantity']:,} units** "
            f"(exact Q*: {calculation['optimal_order_quantity']:,.2f} units, CR: {calculation['critical_ratio']:.4f}) "
            f"for the period to balance underage (${calculation['underage_cost']:,.2f}) and overage (${calculation['overage_cost']:,.2f}) risk."
        )

    # 7. Verification Checks
    if verification is not None:
        status_label = "PASSED" if verification.passed else "ATTENTION"
        lines.append(f"\n**7. Verification Checks ({status_label}):**")
        for chk in verification.checks:
            icon = "[PASS]" if chk.passed else "[FAIL]"
            lines.append(f"- {icon} `{chk.name}`: {chk.detail}")
        for warn in verification.warnings:
            lines.append(f"- [WARN] {warn}")
    else:
        tool_checks = calculation.get("verification_checks", [])
        lines.append("\n**7. Verification Checks:**")
        for chk in tool_checks:
            icon = "[PASS]" if chk.get("passed") else "[FAIL]"
            lines.append(f"- {icon} `{chk.get('name', 'check')}`: {chk.get('detail', '')}")

    # 8. Assumptions (merged & deduplicated from extraction + tool)
    merged_assumptions: List[str] = []
    for asm in list(extraction.assumptions) + list(calculation.get("assumptions", [])):
        if asm and asm not in merged_assumptions:
            merged_assumptions.append(asm)
    lines.append("\n**8. Assumptions:**")
    if merged_assumptions:
        for asm in merged_assumptions:
            lines.append(f"- {asm}")
    else:
        lines.append("- Standard model assumptions apply.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator Class
# ---------------------------------------------------------------------------

class InventoryDecisionAgent:
    """End-to-end Supply Chain Analytics Inventory Decision Agent."""

    def answer(self, question: str) -> AgentResponse:
        """Process a natural-language inventory question through all 7 stages."""
        trace: List[Dict[str, Any]] = [
            {"step": 1, "stage": "Customer Question", "status": "completed", "summary": question.strip()}
        ]

        # Step 2: Identify Model
        identification = identify_model(question)
        trace.append(
            {
                "step": 2,
                "stage": "Identify Model",
                "status": "completed" if identification.model_type != "unknown" else "stopped",
                "summary": f"{identification.model_label} (confidence {identification.confidence:.0%})",
                "details": {
                    "model_type": identification.model_type,
                    "scores": identification.scores,
                    "reason": identification.reason,
                },
            }
        )

        # Step 3: Extract Inputs
        extraction = extract_inputs(question, identification.model_type)
        trace.append(
            {
                "step": 3,
                "stage": "Extract Inputs",
                "status": "completed" if extraction.is_complete else "incomplete",
                "summary": (
                    f"Extracted {len(extraction.inputs)} parameter(s)"
                    + (f"; missing: {', '.join(extraction.missing_required)}" if extraction.missing_required else "")
                ),
                "details": {
                    "inputs": extraction.inputs,
                    "conversions": extraction.conversions,
                    "assumptions": extraction.assumptions,
                    "missing_required": extraction.missing_required,
                },
            }
        )

        if identification.model_type == "unknown":
            explanation = explain(question, identification, extraction, None, None, None)
            return AgentResponse(
                question=question,
                status="unsupported",
                identification=identification,
                extraction=extraction,
                tool_selection=None,
                calculation=None,
                verification=None,
                explanation=explanation,
                pipeline_trace=trace,
            )

        if not extraction.is_complete:
            explanation = explain(question, identification, extraction, None, None, None)
            return AgentResponse(
                question=question,
                status="needs_clarification",
                identification=identification,
                extraction=extraction,
                tool_selection=None,
                calculation=None,
                verification=None,
                explanation=explanation,
                pipeline_trace=trace,
            )

        # Step 4: Select Python Tool
        tool_selection = select_python_tool(identification.model_type, extraction.inputs)
        trace.append(
            {
                "step": 4,
                "stage": "Select Python Tool",
                "status": "completed",
                "summary": f"{tool_selection.tool_name}()",
                "details": {"bound_kwargs": tool_selection.bound_kwargs},
            }
        )

        # Step 5: Calculate (pure Python)
        try:
            calculation = calculate(tool_selection)
        except ValueError as exc:
            trace.append(
                {
                    "step": 5,
                    "stage": "Calculate",
                    "status": "error",
                    "summary": str(exc),
                }
            )
            return AgentResponse(
                question=question,
                status="error",
                identification=identification,
                extraction=extraction,
                tool_selection=tool_selection,
                calculation=None,
                verification=None,
                explanation=f"**Calculation Error:** {exc}",
                pipeline_trace=trace,
            )

        trace.append(
            {
                "step": 5,
                "stage": "Calculate",
                "status": "completed",
                "summary": f"Computed {len(calculation)} output metric(s) in Python.",
                "details": calculation,
            }
        )

        # Step 6: Verify
        verification = verify(identification.model_type, extraction.inputs, calculation)
        trace.append(
            {
                "step": 6,
                "stage": "Verify",
                "status": "passed" if verification.passed else "warning",
                "summary": (
                    f"{sum(1 for c in verification.checks if c.passed)}/{len(verification.checks)} "
                    "verification checks passed."
                ),
                "details": [asdict(c) for c in verification.checks],
            }
        )

        # Step 7: Explain
        explanation = explain(
            question, identification, extraction, tool_selection, calculation, verification
        )
        trace.append(
            {
                "step": 7,
                "stage": "Explain",
                "status": "completed",
                "summary": "Generated verified explanation and operational recommendation.",
            }
        )

        return AgentResponse(
            question=question,
            status="success",
            identification=identification,
            extraction=extraction,
            tool_selection=tool_selection,
            calculation=calculation,
            verification=verification,
            explanation=explanation,
            pipeline_trace=trace,
        )

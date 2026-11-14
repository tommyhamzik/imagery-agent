"""Supply Chain Analytics Inventory Decision Agent — Streamlit Web App & CLI.

Supports two primary workflows:
1. Natural-Language Inventory Decision Agent (5 models, 7-stage pipeline,
   8-section deterministic verification trace from Python tools).
2. External CSV Data Upload & Analysis (pandas DataFrame ingestion, filtering,
   groupby aggregation, deterministic row-by-row inventory calculations,
   8-section SKU audit view, and CSV export).

All numerical calculations execute deterministically in `inventory_tools.py`.
Optional Gemini API summaries never perform or override numerical math and
load credentials securely from session input, `st.secrets`, or environment
variables (never stored in repository files).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None  # type: ignore

try:
    import streamlit as st  # type: ignore
except ImportError:
    st = None  # type: ignore

try:
    from google import genai  # type: ignore
except ImportError:
    genai = None  # type: ignore

from inventory_agent import InventoryDecisionAgent


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
from inventory_tools import (
    TOOL_REGISTRY,
    calculate_eoq,
    calculate_newsvendor,
    calculate_periodic_review,
    calculate_quantity_discount,
    calculate_rop,
)


# ---------------------------------------------------------------------------
# Sample CSV Dataset (covers all 5 supported inventory models)
# ---------------------------------------------------------------------------

_SAMPLE_INVENTORY_COLUMNS = [
    "sku",
    "item_name",
    "category",
    "warehouse",
    "model_type",
    "demand",
    "ordering_cost",
    "holding_cost",
    "unit_cost",
    "holding_rate",
    "tiers",
    "discount_type",
    "operating_days_per_year",
    "avg_demand",
    "demand_std",
    "lead_time",
    "lead_time_std",
    "review_period",
    "safety_stock",
    "service_level",
    "on_hand",
    "on_order",
    "backorders",
    "time_unit",
    "selling_price",
    "salvage_value",
    "demand_distribution",
    "demand_mean",
]

_SAMPLE_ROWS_SEED: List[Dict[str, Any]] = [
    {
        "sku": "SKU-101",
        "item_name": "Industrial Bearing",
        "category": "Components",
        "warehouse": "East",
        "model_type": "eoq",
        "demand": 12000,
        "ordering_cost": 50,
        "holding_cost": 3,
        "operating_days_per_year": 250,
    },
    {
        "sku": "SKU-102",
        "item_name": "Copper Busbar",
        "category": "Components",
        "warehouse": "West",
        "model_type": "eoq",
        "demand": 8000,
        "ordering_cost": 75,
        "unit_cost": 18,
        "holding_rate": 0.20,
        "operating_days_per_year": 365,
    },
    {
        "sku": "SKU-201",
        "item_name": "Precision Valve",
        "category": "Components",
        "warehouse": "East",
        "model_type": "quantity_discount",
        "demand": 5000,
        "ordering_cost": 45,
        "holding_rate": 0.22,
        "tiers": "1-499:$24; 500-999:$21.50; 1000+:$19.50",
        "discount_type": "all_units",
    },
    {
        "sku": "SKU-202",
        "item_name": "Stainless Fastener",
        "category": "Hardware",
        "warehouse": "West",
        "model_type": "quantity_discount",
        "demand": 15000,
        "ordering_cost": 60,
        "holding_cost": 2.50,
        "tiers": "1-999:$4.80; 1000-2499:$4.40; 2500+:$4.10",
        "discount_type": "all_units",
    },
    {
        "sku": "SKU-301",
        "item_name": "Sensor Module",
        "category": "Electronics",
        "warehouse": "East",
        "model_type": "safety_stock_rop",
        "avg_demand": 80,
        "demand_std": 15,
        "lead_time": 5,
        "lead_time_std": 1,
        "service_level": 0.95,
        "time_unit": "days",
    },
    {
        "sku": "SKU-302",
        "item_name": "Relay Switch",
        "category": "Electronics",
        "warehouse": "West",
        "model_type": "safety_stock_rop",
        "avg_demand": 45,
        "demand_std": 9,
        "lead_time": 4,
        "lead_time_std": 0,
        "service_level": 0.98,
        "time_unit": "days",
    },
    {
        "sku": "SKU-401",
        "item_name": "Circuit Board",
        "category": "Electronics",
        "warehouse": "East",
        "model_type": "fixed_period_review",
        "avg_demand": 60,
        "demand_std": 12,
        "lead_time": 3,
        "lead_time_std": 0,
        "review_period": 7,
        "service_level": 0.95,
        "on_hand": 180,
        "on_order": 100,
        "backorders": 15,
        "time_unit": "days",
    },
    {
        "sku": "SKU-402",
        "item_name": "Power Adapter",
        "category": "Hardware",
        "warehouse": "West",
        "model_type": "fixed_period_review",
        "avg_demand": 50,
        "lead_time": 4,
        "review_period": 10,
        "safety_stock": 75,
        "on_hand": 210,
        "on_order": 80,
        "backorders": 0,
        "time_unit": "days",
    },
    {
        "sku": "SKU-501",
        "item_name": "Seasonal Jacket",
        "category": "Perishables",
        "warehouse": "East",
        "model_type": "newsvendor",
        "unit_cost": 40,
        "selling_price": 65,
        "salvage_value": 12,
        "demand_distribution": "normal",
        "demand_mean": 240,
        "demand_std": 45,
    },
    {
        "sku": "SKU-502",
        "item_name": "Artisan Pastry Box",
        "category": "Perishables",
        "warehouse": "West",
        "model_type": "newsvendor",
        "unit_cost": 14,
        "selling_price": 28,
        "salvage_value": 2,
        "demand_distribution": "normal",
        "demand_mean": 150,
        "demand_std": 30,
    },
]


def _build_sample_inventory_csv() -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_SAMPLE_INVENTORY_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in _SAMPLE_ROWS_SEED:
        writer.writerow(row)
    return buf.getvalue()


SAMPLE_INVENTORY_CSV = _build_sample_inventory_csv()

_COLUMN_ALIASES: Dict[str, str] = {
    "sku_id": "sku",
    "item": "item_name",
    "product": "item_name",
    "product_name": "item_name",
    "model": "model_type",
    "annual_demand": "demand",
    "order_cost": "ordering_cost",
    "setup_cost": "ordering_cost",
    "setup_expense": "ordering_cost",
    "cost_per_order": "ordering_cost",
    "fixed_holding_cost": "holding_cost",
    "holding_cost_per_unit": "holding_cost",
    "carrying_cost": "holding_cost",
    "unit_price": "unit_cost",
    "purchase_cost": "unit_cost",
    "unit_purchase_price": "unit_cost",
    "holding_rate_pct": "holding_rate",
    "percentage_holding_cost": "holding_rate",
    "carrying_rate": "holding_rate",
    "price_breaks": "tiers",
    "quantity_breaks": "tiers",
    "average_demand": "avg_demand",
    "daily_demand": "avg_demand",
    "demand_standard_deviation": "demand_std",
    "average_lead_time": "lead_time",
    "lead_time_days": "lead_time_days",
    "lead_time_standard_deviation": "lead_time_std",
    "review_interval": "review_period",
    "on_hand_inventory": "on_hand",
    "on_order_inventory": "on_order",
    "shortage_cost": "underage_cost",
    "excess_cost": "overage_cost",
    "mean_demand": "demand_mean",
    "operating_days": "operating_days_per_year",
}

_FLOAT_FIELDS = {
    "demand",
    "ordering_cost",
    "holding_cost",
    "unit_cost",
    "holding_rate",
    "lead_time_days",
    "operating_days_per_year",
    "avg_demand",
    "demand_std",
    "lead_time",
    "lead_time_std",
    "review_period",
    "safety_stock",
    "service_level",
    "z_score",
    "fill_rate",
    "order_quantity",
    "on_hand",
    "on_order",
    "backorders",
    "inventory_position",
    "selling_price",
    "salvage_value",
    "goodwill_penalty",
    "underage_cost",
    "overage_cost",
    "demand_mean",
    "demand_min",
    "demand_max",
}


# ---------------------------------------------------------------------------
# Environment / Credential Helpers
# ---------------------------------------------------------------------------

def is_running_in_streamlit() -> bool:
    """Return True only when executing inside an active Streamlit runtime session."""
    if st is None:
        return False
    try:
        if hasattr(st, "runtime") and callable(getattr(st.runtime, "exists", None)):
            return bool(st.runtime.exists())
    except Exception:
        return False
    return False


def resolve_gemini_api_key(session_key: Optional[str] = None) -> Optional[str]:
    """Load Gemini API key securely without hardcoding credentials in files.

    Precedence: explicit session input -> `st.secrets["GEMINI_API_KEY"]` -> env var.
    """
    if session_key and session_key.strip():
        return session_key.strip()

    if st is not None:
        try:
            secret_val = st.secrets.get("GEMINI_API_KEY")
            if isinstance(secret_val, str) and secret_val.strip():
                return secret_val.strip()
        except Exception:
            pass

    env_val = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_val and env_val.strip():
        return env_val.strip()
    return None


def generate_gemini_summary(
    question: str,
    calculation: Dict[str, Any],
    api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> Tuple[Optional[str], Optional[str]]:
    """Generate an optional plain-English managerial brief grounded in Python outputs.

    The model is instructed never to recalculate, alter, or invent numbers.
    Returns `(summary_text, error_message)`.
    """
    if genai is None:
        return None, "The `google-genai` package is not installed in this environment."
    if not api_key or not api_key.strip():
        return None, "No Gemini API key configured."

    payload = {
        "model": calculation.get("model"),
        "inputs_extracted": calculation.get("inputs_extracted"),
        "formula": calculation.get("formula"),
        "final_answer": calculation.get("final_answer"),
        "verification_checks": calculation.get("verification_checks"),
        "assumptions": calculation.get("assumptions"),
    }
    prompt = (
        "You are a Supply Chain Analytics Advisor summarizing verified Python inventory outputs.\n"
        "STRICT RULES:\n"
        "1. Do NOT perform or recompute any arithmetic independently.\n"
        "2. Quote ONLY numbers, formulas, and statuses present in the JSON below.\n"
        "3. Do NOT invent missing information.\n"
        "4. Write 3-4 concise sentences interpreting the operational recommendation and risks.\n\n"
        f"Customer Question: {question}\n\n"
        f"Verified Python Output JSON:\n{json.dumps(payload, indent=2, default=str)}"
    )
    try:
        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(model=model_name, contents=prompt)
        text = getattr(response, "text", None)
        return (text.strip() if text else None), None
    except Exception as exc:
        return None, f"Gemini API request failed: {exc}"


# ---------------------------------------------------------------------------
# Pure-Python CSV Ingestion, Normalization, & Batch Analytical Engine
# ---------------------------------------------------------------------------

def _is_missing_cell(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "nan", "none", "null", "na", "n/a")
    return False


def _clean_float_cell(value: Any, *, allow_percent: bool = False) -> Optional[float]:
    if _is_missing_cell(value):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        if allow_percent and num > 1.0:
            return num / 100.0
        return num
    raw = str(value).strip().replace("$", "").replace(",", "")
    has_pct = raw.endswith("%")
    if has_pct:
        raw = raw[:-1].strip()
    num = float(raw)
    if has_pct or (allow_percent and num > 1.0):
        return num / 100.0
    return num


def parse_tiers_spec(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """Parse quantity-discount tier definitions from JSON or CSV shorthand.

    Supported formats:
      - List of dicts: `[{"min_qty": 1, "max_qty": 499, "unit_cost": 20}, ...]`
      - JSON string of tier dicts
      - Shorthand string: `"1-499:$24; 500-999:$21.50; 1000+:$19.50"`
    """
    if _is_missing_cell(raw):
        return None
    if isinstance(raw, list):
        return raw
    text = str(raw).strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("JSON `tiers` must be a list of tier objects.")
        return parsed

    tiers: List[Dict[str, Any]] = []
    segments = [seg.strip() for seg in re.split(r"[;|]", text) if seg.strip()]
    for seg in segments:
        m_range = re.match(
            r"^(\d[\d,]*)\s*(?:-|to|–)\s*(\d[\d,]*)\s*(?::|@|=)\s*\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)$",
            seg,
            re.IGNORECASE,
        )
        if m_range:
            tiers.append(
                {
                    "min_qty": float(m_range.group(1).replace(",", "")),
                    "max_qty": float(m_range.group(2).replace(",", "")),
                    "unit_cost": float(m_range.group(3).replace(",", "")),
                }
            )
            continue

        m_plus = re.match(
            r"^(?:>=)?\s*(\d[\d,]*)\s*\+?\s*(?::|@|=)\s*\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)$",
            seg,
            re.IGNORECASE,
        )
        if m_plus:
            tiers.append(
                {
                    "min_qty": float(m_plus.group(1).replace(",", "")),
                    "max_qty": None,
                    "unit_cost": float(m_plus.group(2).replace(",", "")),
                }
            )
            continue

        raise ValueError(f"Unable to parse quantity discount tier segment '{seg}'.")

    if not tiers:
        raise ValueError("`tiers` column contained no valid price tiers.")
    return tiers


def _parse_demand_distribution_cell(raw: Any) -> Any:
    if _is_missing_cell(raw):
        return None
    if isinstance(raw, list):
        return raw
    text = str(raw).strip()
    low = text.lower()
    if low in ("normal", "uniform", "discrete"):
        return low
    if text.startswith("["):
        return json.loads(text)
    if ":" in text:
        pairs = []
        for seg in re.split(r"[;|,]", text):
            seg = seg.strip()
            if not seg:
                continue
            parts = seg.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid discrete demand pair '{seg}'.")
            pairs.append((float(parts[0].replace(",", "").strip()), float(parts[1].strip())))
        return pairs
    return text


def normalize_csv_row(raw_row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize arbitrary CSV column aliases and clean missing values without inventing numbers."""
    normalized: Dict[str, Any] = {}
    for k, v in raw_row.items():
        if k is None:
            continue
        clean_k = re.sub(r"[^a-z0-9]+", "_", str(k).strip().lower()).strip("_")
        canonical = _COLUMN_ALIASES.get(clean_k, clean_k)
        if canonical not in normalized or _is_missing_cell(normalized[canonical]):
            normalized[canonical] = None if _is_missing_cell(v) else v

    # Handle holding_rate / service_level percentage conventions safely
    pct_fields = {"holding_rate", "service_level", "fill_rate"}
    for field_name in list(normalized.keys()):
        val = normalized[field_name]
        if _is_missing_cell(val):
            normalized[field_name] = None
            continue
        if field_name in _FLOAT_FIELDS:
            normalized[field_name] = _clean_float_cell(val, allow_percent=(field_name in pct_fields))

    if "tiers" in normalized and normalized["tiers"] is not None:
        normalized["tiers"] = parse_tiers_spec(normalized["tiers"])

    if "demand_distribution" in normalized and normalized["demand_distribution"] is not None:
        normalized["demand_distribution"] = _parse_demand_distribution_cell(normalized["demand_distribution"])

    # Normalize model_type names
    raw_model = str(normalized.get("model_type") or "").strip().lower().replace("-", "_").replace(" ", "_")
    model_aliases = {
        "eoq": "eoq",
        "economic_order_quantity": "eoq",
        "quantity_discount": "quantity_discount",
        "price_break": "quantity_discount",
        "safety_stock_rop": "safety_stock_rop",
        "safety_stock": "safety_stock_rop",
        "rop": "safety_stock_rop",
        "reorder_point": "safety_stock_rop",
        "fixed_period_review": "fixed_period_review",
        "periodic_review": "fixed_period_review",
        "p_system": "fixed_period_review",
        "newsvendor": "newsvendor",
        "newsboy": "newsvendor",
    }
    normalized["model_type"] = model_aliases.get(raw_model, raw_model)
    return normalized


def parse_csv_text(csv_text: str) -> List[Dict[str, Any]]:
    """Parse raw CSV string content into a list of raw row dictionaries."""
    reader = csv.DictReader(io.StringIO(csv_text))
    return [dict(row) for row in reader]


def load_sample_inventory_rows() -> List[Dict[str, Any]]:
    """Return the built-in multi-model sample CSV dataset as row dictionaries."""
    return parse_csv_text(SAMPLE_INVENTORY_CSV)


def _summarize_tool_row(model_type: str, res: Dict[str, Any]) -> Dict[str, Any]:
    """Extract uniform comparison columns and summary text from deterministic tool results."""
    checks = res.get("verification_checks", [])
    ver_passed = bool(checks) and all(bool(c.get("passed")) for c in checks)

    summary: Dict[str, Any] = {
        "optimal_order_qty": None,
        "safety_stock": None,
        "reorder_point": None,
        "target_inventory": None,
        "critical_ratio": None,
        "annual_or_expected_cost": None,
        "expected_profit": None,
        "verification_passed": ver_passed,
        "primary_recommendation": "",
    }

    if model_type == "eoq":
        q = res["eoq"]
        cost = res["total_annual_cost"] if res["total_annual_cost"] is not None else res["total_relevant_cost"]
        summary["optimal_order_qty"] = round(q, 2)
        summary["reorder_point"] = (
            round(res["deterministic_rop"], 2) if res.get("deterministic_rop") is not None else None
        )
        summary["annual_or_expected_cost"] = round(cost, 2)
        summary["primary_recommendation"] = (
            f"Order {res['rounded_eoq']:,} units every {res['cycle_time_days']:.1f} days "
            f"(relevant cost ${res['total_relevant_cost']:,.2f}/yr)."
        )
    elif model_type == "quantity_discount":
        summary["optimal_order_qty"] = round(res["optimal_order_quantity"], 2)
        summary["annual_or_expected_cost"] = round(res["total_annual_cost"], 2)
        summary["primary_recommendation"] = (
            f"Order {res['rounded_order_quantity']:,} units @ ${res['optimal_unit_cost']:.2f}/unit "
            f"(total cost ${res['total_annual_cost']:,.2f}/yr)."
        )
    elif model_type == "safety_stock_rop":
        summary["optimal_order_qty"] = (
            round(res["order_quantity"], 2) if res.get("order_quantity") is not None else None
        )
        summary["safety_stock"] = round(res["safety_stock"], 2)
        summary["reorder_point"] = round(res["reorder_point"], 2)
        summary["primary_recommendation"] = (
            f"SS = {res['rounded_safety_stock']:,} units; ROP = {res['rounded_reorder_point']:,} units."
        )
    elif model_type == "fixed_period_review":
        summary["optimal_order_qty"] = (
            round(res["order_quantity"], 2) if res.get("order_quantity") is not None else None
        )
        summary["safety_stock"] = round(res["safety_stock"], 2)
        summary["target_inventory"] = round(res["target_inventory"], 2)
        rec = f"SS = {res['rounded_safety_stock']:,}; Target S = {res['rounded_order_up_to_level']:,} units"
        if res.get("rounded_order_quantity") is not None:
            rec += f"; Order now = {res['rounded_order_quantity']:,} units."
        summary["primary_recommendation"] = rec
    elif model_type == "newsvendor":
        summary["optimal_order_qty"] = round(res["optimal_order_quantity"], 2)
        summary["critical_ratio"] = round(res["critical_ratio"], 4)
        summary["annual_or_expected_cost"] = round(res["expected_mismatch_cost"], 2)
        summary["expected_profit"] = (
            round(res["expected_profit"], 2) if res.get("expected_profit") is not None else None
        )
        summary["primary_recommendation"] = (
            f"Stock {res['rounded_order_quantity']:,} units (CR = {res['critical_ratio']:.3f})."
        )
    return summary


def evaluate_csv_row(raw_row: Dict[str, Any], row_index: int = 0) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Run deterministic Python tool calculation for a single CSV row without inventing missing inputs.

    Returns `(summary_record, full_tool_result_or_None)`.
    """
    meta = {
        "row_index": row_index,
        "sku": raw_row.get("sku") or f"ROW-{row_index + 1}",
        "item_name": raw_row.get("item_name") or "",
        "category": raw_row.get("category") or "Uncategorized",
        "warehouse": raw_row.get("warehouse") or "Default",
    }

    try:
        norm = normalize_csv_row(raw_row)
    except Exception as exc:
        return (
            {
                **meta,
                "model_type": str(raw_row.get("model_type") or "unknown"),
                "status": "error",
                "error": f"Row parsing error: {exc}",
                "optimal_order_qty": None,
                "safety_stock": None,
                "reorder_point": None,
                "target_inventory": None,
                "critical_ratio": None,
                "annual_or_expected_cost": None,
                "expected_profit": None,
                "verification_passed": False,
                "primary_recommendation": "",
            },
            None,
        )

    model_type = norm.get("model_type") or ""
    if model_type not in TOOL_REGISTRY:
        return (
            {
                **meta,
                "model_type": model_type or "unknown",
                "status": "error",
                "error": (
                    f"Unsupported or missing model_type '{model_type}'. "
                    f"Expected one of: {', '.join(sorted(TOOL_REGISTRY.keys()))}."
                ),
                "optimal_order_qty": None,
                "safety_stock": None,
                "reorder_point": None,
                "target_inventory": None,
                "critical_ratio": None,
                "annual_or_expected_cost": None,
                "expected_profit": None,
                "verification_passed": False,
                "primary_recommendation": "",
            },
            None,
        )

    try:
        if model_type == "eoq":
            if norm.get("demand") is None or norm.get("ordering_cost") is None:
                raise ValueError("EOQ row requires `demand` and `ordering_cost`.")
            kwargs: Dict[str, Any] = {
                "demand": norm["demand"],
                "ordering_cost": norm["ordering_cost"],
            }
            for opt in ("holding_cost", "unit_cost", "holding_rate", "lead_time_days", "operating_days_per_year"):
                if norm.get(opt) is not None:
                    kwargs[opt] = norm[opt]
            res = calculate_eoq(**kwargs)

        elif model_type == "quantity_discount":
            if norm.get("demand") is None or norm.get("ordering_cost") is None or norm.get("tiers") is None:
                raise ValueError("Quantity Discount row requires `demand`, `ordering_cost`, and `tiers`.")
            kwargs = {
                "demand": norm["demand"],
                "ordering_cost": norm["ordering_cost"],
                "tiers": norm["tiers"],
            }
            for opt in ("holding_cost", "holding_rate", "discount_type"):
                if norm.get(opt) is not None:
                    kwargs[opt] = norm[opt]
            res = calculate_quantity_discount(**kwargs)

        elif model_type == "safety_stock_rop":
            if norm.get("avg_demand") is None or norm.get("lead_time") is None:
                raise ValueError("Safety Stock & ROP row requires `avg_demand` and `lead_time`.")
            kwargs = {
                "avg_demand": norm["avg_demand"],
                "lead_time": norm["lead_time"],
            }
            for opt in ("demand_std", "lead_time_std", "service_level", "z_score", "fill_rate", "order_quantity", "time_unit"):
                if norm.get(opt) is not None:
                    kwargs[opt] = norm[opt]
            res = calculate_rop(**kwargs)

        elif model_type == "fixed_period_review":
            if norm.get("avg_demand") is None or norm.get("review_period") is None or norm.get("lead_time") is None:
                raise ValueError("Fixed-Period Review row requires `avg_demand`, `review_period`, and `lead_time`.")
            kwargs = {
                "avg_demand": norm["avg_demand"],
                "review_period": norm["review_period"],
                "lead_time": norm["lead_time"],
            }
            for opt in (
                "demand_std",
                "lead_time_std",
                "safety_stock",
                "service_level",
                "z_score",
                "on_hand",
                "on_order",
                "backorders",
                "inventory_position",
                "time_unit",
            ):
                if norm.get(opt) is not None:
                    kwargs[opt] = norm[opt]
            res = calculate_periodic_review(**kwargs)

        else:  # newsvendor
            kwargs = {}
            for opt in (
                "underage_cost",
                "overage_cost",
                "selling_price",
                "unit_cost",
                "salvage_value",
                "goodwill_penalty",
                "demand_mean",
                "demand_std",
                "demand_min",
                "demand_max",
                "demand_distribution",
            ):
                if norm.get(opt) is not None:
                    kwargs[opt] = norm[opt]
            res = calculate_newsvendor(**kwargs)

        summary = _summarize_tool_row(model_type, res)
        return (
            {
                **meta,
                "sku": norm.get("sku") or meta["sku"],
                "item_name": norm.get("item_name") or meta["item_name"],
                "category": norm.get("category") or meta["category"],
                "warehouse": norm.get("warehouse") or meta["warehouse"],
                "model_type": model_type,
                "status": "success",
                "error": None,
                **summary,
            },
            res,
        )
    except Exception as exc:
        return (
            {
                **meta,
                "sku": norm.get("sku") or meta["sku"],
                "item_name": norm.get("item_name") or meta["item_name"],
                "category": norm.get("category") or meta["category"],
                "warehouse": norm.get("warehouse") or meta["warehouse"],
                "model_type": model_type,
                "status": "error",
                "error": str(exc),
                "optimal_order_qty": None,
                "safety_stock": None,
                "reorder_point": None,
                "target_inventory": None,
                "critical_ratio": None,
                "annual_or_expected_cost": None,
                "expected_profit": None,
                "verification_passed": False,
                "primary_recommendation": "",
            },
            None,
        )


def analyze_inventory_records(
    records: List[Dict[str, Any]],
    *,
    filter_categories: Optional[List[str]] = None,
    filter_models: Optional[List[str]] = None,
    filter_warehouses: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Filter raw CSV row dicts and run deterministic Python inventory tools on each row."""
    summaries: List[Dict[str, Any]] = []
    details_by_row: Dict[int, Dict[str, Any]] = {}

    cat_set = {c.strip() for c in filter_categories} if filter_categories else None
    mod_set = {m.strip() for m in filter_models} if filter_models else None
    wh_set = {w.strip() for w in filter_warehouses} if filter_warehouses else None

    for idx, row in enumerate(records):
        summary, detail = evaluate_csv_row(row, row_index=idx)
        if cat_set is not None and summary["category"] not in cat_set:
            continue
        if mod_set is not None and summary["model_type"] not in mod_set:
            continue
        if wh_set is not None and summary["warehouse"] not in wh_set:
            continue
        summaries.append(summary)
        if detail is not None:
            details_by_row[idx] = detail

    return summaries, details_by_row


def summarize_grouped_records(
    analyzed_records: List[Dict[str, Any]],
    group_by: str = "model_type",
) -> List[Dict[str, Any]]:
    """Aggregate analyzed inventory records in pure Python when pandas is unavailable."""
    if group_by not in ("model_type", "category", "warehouse", "status"):
        group_by = "model_type"

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for rec in analyzed_records:
        key = str(rec.get(group_by) or "Unknown")
        buckets.setdefault(key, []).append(rec)

    grouped: List[Dict[str, Any]] = []
    for key, rows in sorted(buckets.items()):
        qty_vals = [r["optimal_order_qty"] for r in rows if r.get("optimal_order_qty") is not None]
        ss_vals = [r["safety_stock"] for r in rows if r.get("safety_stock") is not None]
        cost_vals = [r["annual_or_expected_cost"] for r in rows if r.get("annual_or_expected_cost") is not None]
        grouped.append(
            {
                group_by: key,
                "sku_count": len(rows),
                "success_count": sum(1 for r in rows if r.get("status") == "success"),
                "verified_count": sum(1 for r in rows if r.get("verification_passed")),
                "total_order_qty": round(sum(qty_vals), 2),
                "total_safety_stock": round(sum(ss_vals), 2),
                "total_cost": round(sum(cost_vals), 2),
            }
        )
    return grouped


def analyze_inventory_dataframe(
    df: Any,
    *,
    filter_categories: Optional[List[str]] = None,
    filter_models: Optional[List[str]] = None,
    filter_warehouses: Optional[List[str]] = None,
) -> Tuple[Any, Dict[int, Dict[str, Any]]]:
    """Filter a pandas DataFrame and run deterministic Python calculations on every row."""
    if pd is None:
        raise RuntimeError("pandas is required for DataFrame analysis.")

    records = df.to_dict(orient="records")
    summaries, details_by_row = analyze_inventory_records(
        records,
        filter_categories=filter_categories,
        filter_models=filter_models,
        filter_warehouses=filter_warehouses,
    )
    if not summaries:
        empty_cols = [
            "row_index",
            "sku",
            "item_name",
            "category",
            "warehouse",
            "model_type",
            "status",
            "error",
            "optimal_order_qty",
            "safety_stock",
            "reorder_point",
            "target_inventory",
            "critical_ratio",
            "annual_or_expected_cost",
            "expected_profit",
            "verification_passed",
            "primary_recommendation",
        ]
        return pd.DataFrame(columns=empty_cols), details_by_row
    return pd.DataFrame(summaries), details_by_row


def groupby_inventory_dataframe(analyzed_df: Any, group_by: str = "model_type") -> Any:
    """Group and aggregate an analyzed pandas DataFrame using `df.groupby()`."""
    if pd is None:
        raise RuntimeError("pandas is required for DataFrame grouping.")
    if analyzed_df.empty:
        return pd.DataFrame()
    valid_group = group_by if group_by in analyzed_df.columns else "model_type"
    work_df = analyzed_df.copy()
    work_df["_is_success"] = (work_df["status"] == "success").astype(int)
    work_df["_is_verified"] = work_df["verification_passed"].astype(bool).astype(int)
    grouped = (
        work_df.groupby(valid_group, dropna=False, as_index=False)
        .agg(
            sku_count=("sku", "count"),
            success_count=("_is_success", "sum"),
            verified_count=("_is_verified", "sum"),
            total_order_qty=("optimal_order_qty", "sum"),
            total_safety_stock=("safety_stock", "sum"),
            total_cost=("annual_or_expected_cost", "sum"),
        )
        .sort_values("sku_count", ascending=False)
    )
    return grouped


# ---------------------------------------------------------------------------
# Streamlit UI Rendering
# ---------------------------------------------------------------------------

def _render_verifiable_sections_ui(calculation: Dict[str, Any], title: str = "Verifiable Python Audit Trace") -> None:
    """Render all 8 deterministic verifiable sections inside Streamlit."""
    if st is None:
        return

    st.subheader(title)
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("1. Inputs Extracted", expanded=True):
            st.json(calculation.get("inputs_extracted", {}))
        with st.expander("2. Variable Definitions", expanded=False):
            st.json(calculation.get("variable_definitions", {}))
        with st.expander("3. Formula", expanded=True):
            for k, v in calculation.get("formula", {}).items():
                st.markdown(f"- **{k}:** `{v}`")
        with st.expander("4. Formula with Substituted Values", expanded=True):
            for k, v in calculation.get("formula_substituted", {}).items():
                st.markdown(f"- **{k}:** `{v}`")

    with col_b:
        with st.expander("5. Intermediate Calculations", expanded=True):
            st.json(calculation.get("intermediate_calculations", {}))
        with st.expander("6. Final Answer", expanded=True):
            st.json(calculation.get("final_answer", {}))
        with st.expander("7. Verification Checks", expanded=True):
            for chk in calculation.get("verification_checks", []):
                badge = "PASS" if chk.get("passed") else "FAIL"
                st.markdown(f"- **[{badge}]** `{chk.get('name')}`: {chk.get('detail')}")
        with st.expander("8. Assumptions", expanded=False):
            for asm in calculation.get("assumptions", []):
                st.markdown(f"- {asm}")


def render_streamlit_app() -> None:
    """Render the interactive Supply Chain Inventory Decision Agent web application."""
    if st is None or pd is None:
        raise RuntimeError("Both `streamlit` and `pandas` must be installed to run the web UI.")

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        layout="wide",
    )

    st.title("Supply Chain Analytics — Inventory Decision Agent")
    st.caption(
        "Architecture: Customer Question -> Identify Model -> Extract Inputs -> "
        "Select Python Tool -> Calculate -> Verify -> Explain (100% deterministic Python math)"
    )

    with st.sidebar:
        st.header("Configuration")
        st.markdown("**Supported Inventory Models:**")
        st.markdown(
            "1. Economic Order Quantity (EOQ)\n"
            "2. Quantity Discount\n"
            "3. Safety Stock & Reorder Point\n"
            "4. Fixed-Period Review (P-System)\n"
            "5. Newsvendor (Single-Period)"
        )
        st.divider()
        st.markdown("**Optional Gemini Narrative Layer**")
        session_api_key = st.text_input(
            "Gemini API Key (session only, never saved)",
            type="password",
            help="Or configure GEMINI_API_KEY in Streamlit Secrets or environment variables.",
        )
        effective_key = resolve_gemini_api_key(session_api_key)
        use_gemini = st.checkbox(
            "Generate optional Gemini executive summary",
            value=False,
            disabled=(effective_key is None or genai is None),
        )
        if effective_key is None:
            st.caption("Running in pure deterministic Python mode (no API key required).")
        elif genai is None:
            st.caption("Install `google-genai` to enable optional Gemini narrative summaries.")
        else:
            st.success("Gemini API key available for session (grounded strictly in Python outputs).")

    tab_agent, tab_csv = st.tabs(["Natural-Language Decision Agent", "CSV Data Upload & Analysis"])

    # --- Tab 1: Natural-Language Decision Agent ---
    with tab_agent:
        preset_options = ["Custom Question"] + [f"{label}: {q[:70]}..." for label, q in DEMO_QUESTIONS]
        chosen_preset = st.selectbox("Load an example question or enter your own:", preset_options)

        default_question = ""
        if chosen_preset != "Custom Question":
            preset_idx = preset_options.index(chosen_preset) - 1
            default_question = DEMO_QUESTIONS[preset_idx][1]

        question = st.text_area(
            "Ask an inventory analytics question:",
            value=default_question,
            height=130,
            placeholder=(
                "Example: Annual demand is 12,000 units, ordering/setup cost is $50, "
                "and fixed holding cost is $3 per unit per year. What is the EOQ?"
            ),
        )

        if st.button("Analyze Inventory Question", type="primary"):
            if not question.strip():
                st.warning("Please enter an inventory question first.")
            else:
                agent = InventoryDecisionAgent()
                response = agent.answer(question)

                st.markdown(f"**Status:** `{response.status}` | **Model:** {response.identification.model_label} "
                            f"({response.identification.confidence:.0%})")

                with st.expander("7-Stage Agent Pipeline Trace", expanded=False):
                    st.json(response.pipeline_trace)

                if response.status == "success" and response.calculation is not None:
                    st.markdown(response.explanation)
                    _render_verifiable_sections_ui(
                        response.calculation,
                        title="8-Section Verifiable Calculation Inspection",
                    )
                    if use_gemini and effective_key:
                        summary, err = generate_gemini_summary(question, response.calculation, effective_key)
                        if summary:
                            st.info(f"**Gemini Executive Summary (grounded in Python outputs):**\n\n{summary}")
                        elif err:
                            st.warning(err)
                else:
                    st.warning(response.explanation)

    # --- Tab 2: CSV Data Upload & Analysis ---
    with tab_csv:
        st.markdown(
            "Upload an inventory CSV dataset into a **pandas DataFrame** to filter, group, "
            "run deterministic Python inventory calculations across SKUs, and audit every result."
        )

        col_up, col_tmpl = st.columns([3, 1])
        with col_up:
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        with col_tmpl:
            st.download_button(
                "Download Sample CSV",
                data=SAMPLE_INVENTORY_CSV,
                file_name="sample_inventory_data.csv",
                mime="text/csv",
            )
            use_sample = st.checkbox("Use built-in multi-model sample CSV", value=(uploaded_file is None))

        raw_df = None
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
        elif use_sample:
            raw_df = pd.read_csv(io.StringIO(SAMPLE_INVENTORY_CSV))

        if raw_df is None:
            st.info("Upload a CSV file or check 'Use built-in multi-model sample CSV' to begin.")
            return

        st.markdown(f"**Loaded Dataset:** `{len(raw_df):,}` rows x `{len(raw_df.columns)}` columns")
        with st.expander("Raw Uploaded CSV Preview", expanded=False):
            st.dataframe(raw_df, use_container_width=True)

        # First compute normalized metadata values for filters
        all_analyzed_df, _ = analyze_inventory_dataframe(raw_df)

        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        avail_cats = sorted(all_analyzed_df["category"].dropna().unique().tolist()) if not all_analyzed_df.empty else []
        avail_mods = sorted(all_analyzed_df["model_type"].dropna().unique().tolist()) if not all_analyzed_df.empty else []
        avail_whs = sorted(all_analyzed_df["warehouse"].dropna().unique().tolist()) if not all_analyzed_df.empty else []

        with fcol1:
            sel_cats = st.multiselect("Filter Category", avail_cats, default=avail_cats)
        with fcol2:
            sel_mods = st.multiselect("Filter Inventory Model", avail_mods, default=avail_mods)
        with fcol3:
            sel_whs = st.multiselect("Filter Warehouse", avail_whs, default=avail_whs)
        with fcol4:
            group_dim = st.selectbox("Group Summary By", ["model_type", "category", "warehouse", "status"])

        filtered_df, details_by_row = analyze_inventory_dataframe(
            raw_df,
            filter_categories=sel_cats,
            filter_models=sel_mods,
            filter_warehouses=sel_whs,
        )

        # Summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        success_cnt = int((filtered_df["status"] == "success").sum()) if not filtered_df.empty else 0
        error_cnt = int((filtered_df["status"] == "error").sum()) if not filtered_df.empty else 0
        verified_cnt = int(filtered_df["verification_passed"].sum()) if not filtered_df.empty else 0
        total_cost_sum = (
            float(filtered_df["annual_or_expected_cost"].fillna(0.0).sum()) if not filtered_df.empty else 0.0
        )

        k1.metric("Filtered SKUs", f"{len(filtered_df):,}")
        k2.metric("Calculated Successfully", f"{success_cnt:,} (errors: {error_cnt})")
        k3.metric("Verification Passed", f"{verified_cnt:,} / {len(filtered_df):,}")
        k4.metric("Total Annual / Mismatch Cost", f"${total_cost_sum:,.2f}")

        st.subheader("SKU-Level Deterministic Analysis Results")
        st.dataframe(filtered_df, use_container_width=True)

        if not filtered_df.empty:
            st.download_button(
                "Download Analyzed Results CSV",
                data=filtered_df.to_csv(index=False),
                file_name="analyzed_inventory_results.csv",
                mime="text/csv",
            )

            st.subheader(f"Grouped Summary (`df.groupby('{group_dim}')`)")
            grouped_df = groupby_inventory_dataframe(filtered_df, group_by=group_dim)
            st.dataframe(grouped_df, use_container_width=True)
            if not grouped_df.empty:
                chart_df = grouped_df.set_index(group_dim)[["total_order_qty", "total_safety_stock"]]
                st.bar_chart(chart_df)

            # Single-SKU 8-section inspection
            successful_rows = filtered_df[filtered_df["status"] == "success"]
            if not successful_rows.empty:
                sku_options = [
                    (int(r["row_index"]), f"{r['sku']} — {r['item_name']} ({r['model_type']})")
                    for _, r in successful_rows.iterrows()
                ]
                selected_idx, _ = st.selectbox(
                    "Inspect Full 8-Section Calculation Trace for SKU:",
                    sku_options,
                    format_func=lambda item: item[1],
                )
                if selected_idx in details_by_row:
                    _render_verifiable_sections_ui(
                        details_by_row[selected_idx],
                        title=f"Verifiable Calculation Audit — Row #{selected_idx + 1}",
                    )


# ---------------------------------------------------------------------------
# Headless CLI Entrypoint (runs cleanly without Streamlit)
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Supply Chain Analytics Inventory Decision Agent (Streamlit + CLI)"
    )
    parser.add_argument("-q", "--question", type=str, help="Solve a single natural-language inventory question")
    parser.add_argument("--demo", action="store_true", help="Run all 5 built-in model demonstrations")
    parser.add_argument(
        "--csv",
        nargs="?",
        const="__SAMPLE__",
        help="Analyze an inventory CSV file (omit path to run the built-in multi-model sample CSV)",
    )
    parser.add_argument("--group-by", type=str, default="model_type", help="Grouping column for --csv summary")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    args = parser.parse_args(argv)

    if args.csv is not None:
        if args.csv == "__SAMPLE__":
            rows = load_sample_inventory_rows()
        else:
            with open(args.csv, "r", encoding="utf-8") as fh:
                rows = parse_csv_text(fh.read())
        summaries, details = analyze_inventory_records(rows)
        grouped = summarize_grouped_records(summaries, group_by=args.group_by)
        if args.json:
            print(json.dumps({"rows": summaries, "grouped": grouped}, indent=2, default=str))
        else:
            print("=" * 72)
            print(f"CSV INVENTORY ANALYSIS ({len(summaries)} SKUs, {len(details)} computed)")
            print("=" * 72)
            for rec in summaries:
                if rec["status"] == "success":
                    print(
                        f"- {rec['sku']} ({rec['model_type']} | {rec['category']}): "
                        f"{rec['primary_recommendation']} [verified={rec['verification_passed']}]"
                    )
                else:
                    print(f"- {rec['sku']} ({rec['model_type']}): ERROR - {rec['error']}")
            print("\nGrouped Summary:")
            for g in grouped:
                print(f"  * {g[args.group_by]}: {g}")
        return 0

    agent = InventoryDecisionAgent()
    if args.question:
        res = agent.answer(args.question)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2, default=str))
        else:
            print(res.explanation)
        return 0

    if args.demo:
        outputs = []
        for idx, (label, q) in enumerate(DEMO_QUESTIONS, start=1):
            res = agent.answer(q)
            if args.json:
                outputs.append({"demo": label, **res.to_dict()})
            else:
                print("=" * 72)
                print(f"DEMO {idx}: {label}")
                print(f"Question: {q}\n")
                print(res.explanation)
                print()
        if args.json:
            print(json.dumps(outputs, indent=2, default=str))
        return 0

    parser.print_help()
    print("\nTo launch the browser UI: streamlit run app.py")
    return 0


if is_running_in_streamlit():
    render_streamlit_app()
elif __name__ == "__main__":
    raise SystemExit(main())

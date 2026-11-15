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

PRODUCT_REQUIRED_COLUMNS = ["Product", "Category", "Units", "Unit Cost"]

SAMPLE_PRODUCT_CSV = """Product,Category,Units,Unit Cost
Industrial Bearing,Components,1200,4.50
Copper Busbar,Components,850,18.00
Precision Valve,Components,420,24.00
Stainless Fastener,Hardware,3500,0.85
Power Adapter,Hardware,1100,12.50
Steel Bracket,Hardware,2400,3.20
Sensor Module,Electronics,950,35.00
Relay Switch,Electronics,1600,8.75
Circuit Board,Electronics,680,52.00
Control Unit,Electronics,310,115.00
"""


def validate_product_dataframe_columns(df: Any) -> Tuple[bool, List[str]]:
    """Validate that Product, Category, Units, and Unit Cost columns are present."""
    if df is None:
        return False, list(PRODUCT_REQUIRED_COLUMNS)
    cols = list(df.columns) if hasattr(df, "columns") else []
    col_map = {str(c).strip().lower(): c for c in cols}
    missing: List[str] = []
    rename_map: Dict[str, str] = {}
    for req in PRODUCT_REQUIRED_COLUMNS:
        req_clean = req.lower()
        if req in cols:
            continue
        elif req_clean in col_map:
            rename_map[col_map[req_clean]] = req
        else:
            missing.append(req)

    if rename_map and not missing and hasattr(df, "rename"):
        df.rename(columns=rename_map, inplace=True)

    return len(missing) == 0, missing


def analyze_product_dataframe(df: Any) -> Dict[str, Any]:
    """Perform pandas calculations on uploaded product DataFrame:
    1. Calculate total Units.
    2. Calculate average Unit Cost.
    3. Identify the Product with the highest Units.
    4. Group the data by Category and calculate total Units for each Category.
    5. Create a new calculated column: Inventory Value = Units × Unit Cost.
    6. Calculate total Inventory Value across all products.
    """
    if pd is None or df is None:
        raise RuntimeError("pandas is required for DataFrame analysis.")

    work_df = df.copy()

    # Clean Units and Unit Cost to numeric types using pandas
    if work_df["Units"].dtype == object:
        work_df["Units"] = pd.to_numeric(
            work_df["Units"].astype(str).str.replace(",", "").str.strip(),
            errors="coerce",
        ).fillna(0.0)
    else:
        work_df["Units"] = pd.to_numeric(work_df["Units"], errors="coerce").fillna(0.0)

    if work_df["Unit Cost"].dtype == object:
        work_df["Unit Cost"] = pd.to_numeric(
            work_df["Unit Cost"].astype(str).str.replace("$", "").str.replace(",", "").str.strip(),
            errors="coerce",
        ).fillna(0.0)
    else:
        work_df["Unit Cost"] = pd.to_numeric(work_df["Unit Cost"], errors="coerce").fillna(0.0)

    # 1. Calculate total Units
    total_units = float(work_df["Units"].sum())

    # 2. Calculate average Unit Cost
    avg_unit_cost = float(work_df["Unit Cost"].mean()) if len(work_df) > 0 else 0.0

    # 3. Identify the Product with the highest Units
    if len(work_df) > 0:
        max_idx = work_df["Units"].idxmax()
        highest_units_product = str(work_df.loc[max_idx, "Product"])
        highest_units_value = float(work_df.loc[max_idx, "Units"])
    else:
        highest_units_product = "N/A"
        highest_units_value = 0.0

    # 4. Group the data by Category and calculate total Units for each Category
    units_by_category = (
        work_df.groupby("Category", as_index=False)["Units"]
        .sum()
        .sort_values(by="Units", ascending=False)
    )

    # 5. Create a new calculated column: Inventory Value = Units × Unit Cost
    work_df["Inventory Value"] = work_df["Units"] * work_df["Unit Cost"]

    # 6. Calculate total Inventory Value across all products
    total_inventory_value = float(work_df["Inventory Value"].sum())

    return {
        "num_rows": len(work_df),
        "num_columns": len(work_df.columns),
        "column_names": list(work_df.columns),
        "total_units": total_units,
        "avg_unit_cost": avg_unit_cost,
        "highest_units_product": highest_units_product,
        "highest_units_value": highest_units_value,
        "category_summary": units_by_category,
        "total_inventory_value": total_inventory_value,
        "analyzed_df": work_df,
    }


def analyze_product_records(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure-Python fallback for analyzing product CSV records when pandas is unavailable."""
    if not rows:
        return {
            "is_valid": False,
            "missing": list(PRODUCT_REQUIRED_COLUMNS),
            "num_rows": 0,
            "num_columns": 0,
            "column_names": [],
            "total_units": 0.0,
            "avg_unit_cost": 0.0,
            "highest_units_product": "N/A",
            "highest_units_value": 0.0,
            "total_inventory_value": 0.0,
            "category_summary": [],
            "rows": [],
        }

    first = rows[0]
    col_map = {str(k).strip().lower(): k for k in first.keys()}
    missing: List[str] = [req for req in PRODUCT_REQUIRED_COLUMNS if req.lower() not in col_map]
    if missing:
        return {
            "is_valid": False,
            "missing": missing,
            "num_rows": len(rows),
            "num_columns": len(first.keys()),
            "column_names": list(first.keys()),
            "total_units": 0.0,
            "avg_unit_cost": 0.0,
            "highest_units_product": "N/A",
            "highest_units_value": 0.0,
            "total_inventory_value": 0.0,
            "category_summary": [],
            "rows": rows,
        }

    norm_rows: List[Dict[str, Any]] = []
    for r in rows:
        nr: Dict[str, Any] = {}
        for k, v in r.items():
            kl = str(k).strip().lower()
            for rq in PRODUCT_REQUIRED_COLUMNS:
                if rq.lower() == kl:
                    nr[rq] = v
        raw_u = str(nr.get("Units", 0)).replace(",", "").strip()
        units_val = float(raw_u) if raw_u and raw_u.lower() != "none" else 0.0
        raw_c = str(nr.get("Unit Cost", 0)).replace("$", "").replace(",", "").strip()
        cost_val = float(raw_c) if raw_c and raw_c.lower() != "none" else 0.0
        nr["Units"] = units_val
        nr["Unit Cost"] = cost_val
        nr["Inventory Value"] = units_val * cost_val
        norm_rows.append(nr)

    total_units = sum(r["Units"] for r in norm_rows)
    avg_cost = sum(r["Unit Cost"] for r in norm_rows) / len(norm_rows) if norm_rows else 0.0
    highest = max(norm_rows, key=lambda r: r["Units"]) if norm_rows else {"Product": "N/A", "Units": 0.0}
    total_val = sum(r["Inventory Value"] for r in norm_rows)

    cats: Dict[str, float] = {}
    for r in norm_rows:
        c = str(r.get("Category", "Unknown"))
        cats[c] = cats.get(c, 0.0) + r["Units"]

    cat_list = sorted(cats.items(), key=lambda x: x[1], reverse=True)

    return {
        "is_valid": True,
        "missing": [],
        "num_rows": len(norm_rows),
        "num_columns": len(norm_rows[0].keys()) if norm_rows else 0,
        "column_names": list(norm_rows[0].keys()) if norm_rows else [],
        "total_units": total_units,
        "avg_unit_cost": avg_cost,
        "highest_units_product": highest["Product"],
        "highest_units_value": highest["Units"],
        "total_inventory_value": total_val,
        "category_summary": cat_list,
        "rows": norm_rows,
    }


# ---------------------------------------------------------------------------
# Multicriteria Inventory Classification (Sales & Hits -> A, B, C, D Classes)
# ---------------------------------------------------------------------------

CLASSIFICATION_REQUIRED_COLUMNS = ["Item Number", "Item Name", "UoM", "Sales ($)", "Hits"]

SAMPLE_CLASSIFICATION_CSV = """Item Number,Item Name,UoM,Sales ($),Hits
SKU-1001,Heavy Industrial Motor,EA,145000,420
SKU-1002,Hydraulic Pump Assembly,EA,98000,310
SKU-1003,Precision Bearings Box,BX,62000,850
SKU-1004,Stainless Fastener Pack,BX,41000,1250
SKU-1005,Control Circuit Board,EA,35000,280
SKU-1006,Standard Copper Wire,RL,28000,640
SKU-1007,Pneumatic Valve 1/2in,EA,19000,195
SKU-1008,Thermal Sensor Module,EA,14000,520
SKU-1009,Steel Mounting Bracket,EA,9500,410
SKU-1010,Rubber Seal Ring Pack,PK,7200,980
SKU-1011,LED Indicator Lamp,BX,4500,340
SKU-1012,Brass Compression Fitting,EA,3100,210
SKU-1013,Dust Filter Cartridge,EA,2200,140
SKU-1014,Terminal Block Strip,EA,1100,95
SKU-1015,Legacy Fuse 20A,BX,450,22
"""

_CLASSIFICATION_ALIAS_MAP: Dict[str, List[str]] = {
    "item number": [
        "item number", "item_number", "item no", "item #", "item", "item_no", "itemid", "item_id",
        "sku", "sku_id", "part number", "part_number", "part no", "part #", "product id", "id", "code", "material"
    ],
    "item name": [
        "item name", "item_name", "item desc", "item_desc", "item description", "description",
        "product", "product name", "product_name", "name", "desc", "title", "label"
    ],
    "uom": [
        "uom", "unit of measure", "unit_of_measure", "unit", "units", "measure", "pkg", "package"
    ],
    "sales ($)": [
        "sales ($)", "sales($)", "sales", "annual sales", "revenue", "annual revenue", "sales_dollars",
        "total sales", "annual_sales", "dollar sales", "sales amount", "total_sales", "sales ($/yr)", "sales ($/year)"
    ],
    "hits": [
        "hits", "hit", "pick count", "picks", "pick_count", "order count", "orders", "order_count",
        "frequency", "transactions", "transaction count", "order_lines", "lines", "hit count", "hit_count"
    ],
}


def validate_classification_dataframe_columns(df: Any) -> Tuple[bool, List[str]]:
    """Validate that Item Number, Item Name, UoM, Sales ($), and Hits columns are present (with alias support).
    
    If Sales and Hits are present, auto-fills missing metadata columns (Item Number, Item Name, UoM)
    so files such as ProjectDataV2_UTF8.csv parse seamlessly.
    """
    if df is None:
        return False, list(CLASSIFICATION_REQUIRED_COLUMNS)
    if hasattr(df, "columns"):
        df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    cols = list(df.columns) if hasattr(df, "columns") else []
    col_map = {str(c).lstrip("\ufeff").strip().lower(): c for c in cols}

    rename_map: Dict[str, str] = {}
    for req in CLASSIFICATION_REQUIRED_COLUMNS:
        req_clean = req.lower()
        if req in cols:
            continue
        matched_col = None
        for alias in _CLASSIFICATION_ALIAS_MAP.get(req_clean, [req_clean]):
            if alias in col_map:
                matched_col = col_map[alias]
                break
        if matched_col:
            rename_map[matched_col] = req

    if rename_map and hasattr(df, "rename"):
        df.rename(columns=rename_map, inplace=True)

    current_cols = list(df.columns) if hasattr(df, "columns") else []
    
    # Check if primary numeric criteria (Sales and Hits) are present
    has_sales = "Sales ($)" in current_cols
    has_hits = "Hits" in current_cols

    if not has_sales or not has_hits:
        missing = []
        if not has_sales:
            missing.append("Sales ($)")
        if not has_hits:
            missing.append("Hits")
        for col_req in ["Item Number", "Item Name", "UoM"]:
            if col_req not in current_cols:
                missing.append(col_req)
        return False, missing

    # Auto-fill metadata columns if missing
    if "Item Number" not in current_cols:
        candidate = [c for c in current_cols if c not in ["Sales ($)", "Hits"]]
        if candidate:
            df["Item Number"] = df[candidate[0]].astype(str)
        else:
            df["Item Number"] = [f"SKU-{i+1:04d}" for i in range(len(df))]

    if "Item Name" not in current_cols:
        df["Item Name"] = df["Item Number"].astype(str)

    if "UoM" not in current_cols:
        df["UoM"] = "EA"

    return True, []


def classify_inventory_dataframe(
    df: Any,
    sales_cutoffs: Tuple[float, float, float] = (0.70, 0.90, 0.98),
    hits_cutoffs: Tuple[float, float, float] = (0.70, 0.90, 0.98),
    method: str = "matrix",
) -> Tuple[Any, Dict[str, Any]]:
    """Perform multicriteria inventory classification into A, B, C, and D classes using pandas.

    Uses Sales ($) and Hits to perform joint criteria classification:
    - Sorts by Sales ($) descending, calculates cumulative % to assign Sales Class (A, B, C, D).
    - Sorts by Hits descending, calculates cumulative % to assign Hits Class (A, B, C, D).
    - Combines Sales Class and Hits Class into a single Multicriteria Classification (A, B, C, D).
    """
    if pd is None or df is None:
        raise RuntimeError("pandas is required for DataFrame classification.")

    work_df = df.copy()

    # Clean numeric columns using pandas
    for col in ["Sales ($)", "Hits"]:
        if col in work_df.columns:
            if work_df[col].dtype == object:
                work_df[col] = pd.to_numeric(
                    work_df[col].astype(str).str.replace("$", "").str.replace(",", "").str.strip(),
                    errors="coerce",
                ).fillna(0.0)
            else:
                work_df[col] = pd.to_numeric(work_df[col], errors="coerce").fillna(0.0)

    # 1. Sales-based classification
    total_sales = float(work_df["Sales ($)"].sum())
    work_df = work_df.sort_values(by="Sales ($)", ascending=False).reset_index(drop=True)
    if total_sales > 0:
        work_df["Sales Cumulative"] = work_df["Sales ($)"].cumsum()
        work_df["Sales Cumulative %"] = work_df["Sales Cumulative"] / total_sales
    else:
        work_df["Sales Cumulative"] = 0.0
        work_df["Sales Cumulative %"] = 0.0

    s_a, s_b, s_c = sales_cutoffs

    def _assign_s_class(row):
        val = row["Sales ($)"]
        if val <= 0:
            return "D"
        cum = row["Sales Cumulative %"]
        if cum <= s_a or (row.name == 0 and cum > s_a):
            return "A"
        elif cum <= s_b:
            return "B"
        elif cum <= s_c:
            return "C"
        else:
            return "D"

    work_df["Sales Class"] = work_df.apply(_assign_s_class, axis=1)

    # 2. Hits-based classification
    total_hits = float(work_df["Hits"].sum())
    work_df = work_df.sort_values(by="Hits", ascending=False).reset_index(drop=True)
    if total_hits > 0:
        work_df["Hits Cumulative"] = work_df["Hits"].cumsum()
        work_df["Hits Cumulative %"] = work_df["Hits Cumulative"] / total_hits
    else:
        work_df["Hits Cumulative"] = 0.0
        work_df["Hits Cumulative %"] = 0.0

    h_a, h_b, h_c = hits_cutoffs

    def _assign_h_class(row):
        val = row["Hits"]
        if val <= 0:
            return "D"
        cum = row["Hits Cumulative %"]
        if cum <= h_a or (row.name == 0 and cum > h_a):
            return "A"
        elif cum <= h_b:
            return "B"
        elif cum <= h_c:
            return "C"
        else:
            return "D"

    work_df["Hits Class"] = work_df.apply(_assign_h_class, axis=1)

    # 3. Multicriteria Classification (A, B, C, D)
    # Flores & Whybark joint criteria matrix mapping
    matrix_map = {
        ("A", "A"): "A",
        ("A", "B"): "A",
        ("B", "A"): "A",
        ("A", "C"): "B",
        ("C", "A"): "B",
        ("B", "B"): "B",
        ("B", "C"): "B",
        ("C", "B"): "C",
        ("A", "D"): "C",
        ("D", "A"): "B",
        ("B", "D"): "C",
        ("D", "B"): "C",
        ("C", "C"): "C",
        ("C", "D"): "D",
        ("D", "C"): "D",
        ("D", "D"): "D",
    }
    priority_order = {"A": 1, "B": 2, "C": 3, "D": 4}
    rev_priority = {1: "A", 2: "B", 3: "C", 4: "D"}

    def _assign_multi_class(row):
        sc = row["Sales Class"]
        hc = row["Hits Class"]
        if method == "priority":
            p_val = min(priority_order.get(sc, 4), priority_order.get(hc, 4))
            return rev_priority[p_val]
        else:
            return matrix_map.get((sc, hc), "C")

    work_df["Classification"] = work_df.apply(_assign_multi_class, axis=1)
    work_df["Multicriteria Matrix"] = work_df["Sales Class"] + "-" + work_df["Hits Class"]

    # Re-sort by Sales ($) descending for clean presentation
    work_df = work_df.sort_values(by="Sales ($)", ascending=False).reset_index(drop=True)

    # Calculate summary aggregations by Classification class
    class_order = ["A", "B", "C", "D"]
    summary_rows = []
    for cls in class_order:
        sub = work_df[work_df["Classification"] == cls]
        c_items = len(sub)
        c_sales = float(sub["Sales ($)"].sum())
        c_hits = float(sub["Hits"].sum())
        pct_items = (c_items / len(work_df) * 100.0) if len(work_df) > 0 else 0.0
        pct_sales = (c_sales / total_sales * 100.0) if total_sales > 0 else 0.0
        pct_hits = (c_hits / total_hits * 100.0) if total_hits > 0 else 0.0
        summary_rows.append({
            "Class": cls,
            "Item Count": c_items,
            "% of Items": round(pct_items, 1),
            "Total Sales ($)": c_sales,
            "% of Sales": round(pct_sales, 1),
            "Total Hits": c_hits,
            "% of Hits": round(pct_hits, 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    crosstab_df = pd.crosstab(
        work_df["Sales Class"],
        work_df["Hits Class"],
        margins=True,
        margins_name="Total",
    ).reindex(index=["A", "B", "C", "D", "Total"], columns=["A", "B", "C", "D", "Total"], fill_value=0)

    stats = {
        "num_rows": len(work_df),
        "total_sales": total_sales,
        "total_hits": total_hits,
        "class_counts": {cls: len(work_df[work_df["Classification"] == cls]) for cls in class_order},
        "summary_df": summary_df,
        "crosstab_df": crosstab_df,
    }

    return work_df, stats


def classify_inventory_records(
    rows: List[Dict[str, Any]],
    sales_cutoffs: Tuple[float, float, float] = (0.70, 0.90, 0.98),
    hits_cutoffs: Tuple[float, float, float] = (0.70, 0.90, 0.98),
    method: str = "matrix",
) -> Dict[str, Any]:
    """Pure-Python fallback for multicriteria inventory classification when pandas is unavailable."""
    if not rows:
        return {
            "is_valid": False,
            "missing": list(CLASSIFICATION_REQUIRED_COLUMNS),
            "num_rows": 0,
            "rows": [],
            "summary_rows": [],
            "class_counts": {"A": 0, "B": 0, "C": 0, "D": 0},
            "total_sales": 0.0,
            "total_hits": 0.0,
        }

    first = rows[0]
    col_map = {str(k).lstrip("\ufeff").strip().lower(): k for k in first.keys()}

    has_sales = any(alias in col_map for alias in _CLASSIFICATION_ALIAS_MAP["sales ($)"])
    has_hits = any(alias in col_map for alias in _CLASSIFICATION_ALIAS_MAP["hits"])

    if not has_sales or not has_hits:
        missing: List[str] = []
        if not has_sales:
            missing.append("Sales ($)")
        if not has_hits:
            missing.append("Hits")
        for col_req in ["Item Number", "Item Name", "UoM"]:
            if not any(alias in col_map for alias in _CLASSIFICATION_ALIAS_MAP[col_req.lower()]):
                missing.append(col_req)
        return {
            "is_valid": False,
            "missing": missing,
            "num_rows": len(rows),
            "rows": rows,
            "summary_rows": [],
            "class_counts": {"A": 0, "B": 0, "C": 0, "D": 0},
            "total_sales": 0.0,
            "total_hits": 0.0,
        }

    norm_rows: List[Dict[str, Any]] = []
    for idx, r in enumerate(rows):
        nr: Dict[str, Any] = {}
        cleaned_r = {str(k).lstrip("\ufeff").strip(): v for k, v in r.items()}
        for k, v in cleaned_r.items():
            kl = str(k).strip().lower()
            mapped_key = None
            for req, aliases in _CLASSIFICATION_ALIAS_MAP.items():
                if kl == req or kl in aliases:
                    for req_cap in CLASSIFICATION_REQUIRED_COLUMNS:
                        if req_cap.lower() == req:
                            mapped_key = req_cap
                            break
            if mapped_key:
                nr[mapped_key] = v
            else:
                nr[k] = v

        if "Item Number" not in nr:
            non_sales = [k for k in cleaned_r.keys() if "sale" not in k.lower() and "hit" not in k.lower()]
            nr["Item Number"] = str(cleaned_r[non_sales[0]]) if non_sales else f"SKU-{idx+1:04d}"
        if "Item Name" not in nr:
            nr["Item Name"] = str(nr["Item Number"])
        if "UoM" not in nr:
            nr["UoM"] = "EA"

        raw_s = str(nr.get("Sales ($)", 0)).replace("$", "").replace(",", "").strip()
        sales_val = float(raw_s) if raw_s and raw_s.lower() != "none" else 0.0
        raw_h = str(nr.get("Hits", 0)).replace(",", "").strip()
        hits_val = float(raw_h) if raw_h and raw_h.lower() != "none" else 0.0

        nr["Sales ($)"] = sales_val
        nr["Hits"] = hits_val
        norm_rows.append(nr)

    # 1. Sales sorting & cumulative %
    total_sales = sum(r["Sales ($)"] for r in norm_rows)
    norm_rows.sort(key=lambda x: x["Sales ($)"], reverse=True)
    cum_s = 0.0
    s_a, s_b, s_c = sales_cutoffs
    for idx, r in enumerate(norm_rows):
        cum_s += r["Sales ($)"]
        cum_pct = (cum_s / total_sales) if total_sales > 0 else 0.0
        r["Sales Cumulative %"] = cum_pct
        if r["Sales ($)"] <= 0:
            r["Sales Class"] = "D"
        elif cum_pct <= s_a or (idx == 0 and cum_pct > s_a):
            r["Sales Class"] = "A"
        elif cum_pct <= s_b:
            r["Sales Class"] = "B"
        elif cum_pct <= s_c:
            r["Sales Class"] = "C"
        else:
            r["Sales Class"] = "D"

    # 2. Hits sorting & cumulative %
    total_hits = sum(r["Hits"] for r in norm_rows)
    norm_rows.sort(key=lambda x: x["Hits"], reverse=True)
    cum_h = 0.0
    h_a, h_b, h_c = hits_cutoffs
    for idx, r in enumerate(norm_rows):
        cum_h += r["Hits"]
        cum_pct = (cum_h / total_hits) if total_hits > 0 else 0.0
        r["Hits Cumulative %"] = cum_pct
        if r["Hits"] <= 0:
            r["Hits Class"] = "D"
        elif cum_pct <= h_a or (idx == 0 and cum_pct > h_a):
            r["Hits Class"] = "A"
        elif cum_pct <= h_b:
            r["Hits Class"] = "B"
        elif cum_pct <= h_c:
            r["Hits Class"] = "C"
        else:
            r["Hits Class"] = "D"

    # 3. Multicriteria combination
    matrix_map = {
        ("A", "A"): "A", ("A", "B"): "A", ("B", "A"): "A",
        ("A", "C"): "B", ("C", "A"): "B", ("B", "B"): "B", ("B", "C"): "B",
        ("C", "B"): "C", ("A", "D"): "C", ("D", "A"): "B", ("B", "D"): "C",
        ("D", "B"): "C", ("C", "C"): "C", ("C", "D"): "D", ("D", "C"): "D",
        ("D", "D"): "D",
    }
    priority_order = {"A": 1, "B": 2, "C": 3, "D": 4}
    rev_priority = {1: "A", 2: "B", 3: "C", 4: "D"}

    for r in norm_rows:
        sc = r["Sales Class"]
        hc = r["Hits Class"]
        if method == "priority":
            p_val = min(priority_order.get(sc, 4), priority_order.get(hc, 4))
            r["Classification"] = rev_priority[p_val]
        else:
            r["Classification"] = matrix_map.get((sc, hc), "C")
        r["Multicriteria Matrix"] = f"{sc}-{hc}"

    # Re-sort by Sales ($) descending
    norm_rows.sort(key=lambda x: x["Sales ($)"], reverse=True)

    class_order = ["A", "B", "C", "D"]
    summary_rows = []
    class_counts = {}
    for cls in class_order:
        sub = [r for r in norm_rows if r["Classification"] == cls]
        c_items = len(sub)
        class_counts[cls] = c_items
        c_sales = sum(r["Sales ($)"] for r in sub)
        c_hits = sum(r["Hits"] for r in sub)
        pct_items = (c_items / len(norm_rows) * 100.0) if norm_rows else 0.0
        pct_sales = (c_sales / total_sales * 100.0) if total_sales > 0 else 0.0
        pct_hits = (c_hits / total_hits * 100.0) if total_hits > 0 else 0.0
        summary_rows.append({
            "Class": cls,
            "Item Count": c_items,
            "% of Items": round(pct_items, 1),
            "Total Sales ($)": c_sales,
            "% of Sales": round(pct_sales, 1),
            "Total Hits": c_hits,
            "% of Hits": round(pct_hits, 1),
        })

    return {
        "is_valid": True,
        "missing": [],
        "num_rows": len(norm_rows),
        "total_sales": total_sales,
        "total_hits": total_hits,
        "class_counts": class_counts,
        "summary_rows": summary_rows,
        "rows": norm_rows,
    }

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
    clean_text = csv_text.lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(clean_text))
    rows = []
    for row in reader:
        rows.append({str(k).lstrip("\ufeff").strip(): v for k, v in row.items()})
    return rows


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


def _render_sku_decision_analysis_view(raw_df: Any) -> None:
    """Helper to render the multi-model SKU decision agent analysis on raw_df."""
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

        # --- DATA UPLOAD (Last section in sidebar) ---
        st.divider()
        st.header("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload an inventory CSV file with Product, Category, Units, and Unit Cost columns.",
        )

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
        st.header("Inventory Data Upload & Multicriteria ABC Classification")
        st.markdown(
            "Upload an external CSV file (such as **`ProjectDataV2_UTF8.csv`**) to inspect raw data, "
            "view the portfolio summary, and run **Multicriteria ABC Inventory Classification** (Sales & Hits) "
            "— powered 100% by deterministic Python and pandas with no API key required."
        )

        col_uploader, col_sample = st.columns([3, 1])
        with col_uploader:
            uploaded_file = st.file_uploader(
                "Upload CSV File (e.g., ProjectDataV2_UTF8.csv)",
                type=["csv"],
                key="tab_csv_uploader",
                help="Select or drag-and-drop a CSV file containing inventory data.",
            )
        with col_sample:
            st.download_button(
                "Download Sample CSV",
                data=SAMPLE_CLASSIFICATION_CSV,
                file_name="sample_inventory_classification.csv",
                mime="text/csv",
                help="Download a pre-formatted sample dataset to explore the classification features.",
            )
            use_sample = st.checkbox(
                "Use Built-in Sample Dataset",
                value=False,
                key="tab_use_sample_checkbox",
                help="Load sample inventory records to preview the summary and classification.",
            )

        raw_df = None
        file_display_name = "Uploaded Dataset"

        if uploaded_file is not None:
            file_display_name = uploaded_file.name
            try:
                raw_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            except Exception:
                try:
                    uploaded_file.seek(0)
                    raw_df = pd.read_csv(uploaded_file, encoding="utf-8")
                except Exception:
                    try:
                        uploaded_file.seek(0)
                        raw_df = pd.read_csv(uploaded_file, encoding="latin1")
                    except Exception as err:
                        st.error(f"Error reading CSV file '{uploaded_file.name}': {err}")
                        return
        elif use_sample:
            file_display_name = "sample_inventory_classification.csv (Demo)"
            raw_df = pd.read_csv(io.StringIO(SAMPLE_CLASSIFICATION_CSV))

        if raw_df is None:
            st.info("👆 **Please upload a CSV file above** (such as `ProjectDataV2_UTF8.csv`) or check 'Use Built-in Sample Dataset' to begin.")
            return

        # Clean column names (strip leading/trailing whitespace)
        raw_df.columns = [str(c).strip() for c in raw_df.columns]

        # -------------------------------------------------------------
        # 1. DATA PREVIEW
        # -------------------------------------------------------------
        st.divider()
        st.subheader(f"Data Preview: `{file_display_name}`")
        col_m1, col_m2, col_m3 = st.columns([1, 1, 3])
        col_m1.metric("Total Rows", f"{len(raw_df):,}")
        col_m2.metric("Total Columns", f"{len(raw_df.columns):,}")
        col_m3.markdown(f"**Detected Columns:** `{', '.join(list(raw_df.columns))}`")

        st.dataframe(raw_df.head(25), use_container_width=True)
        if len(raw_df) > 25:
            st.caption(f"Showing first 25 of {len(raw_df):,} total records.")

        # Check dataset schema
        is_class_df, missing_class_cols = validate_classification_dataframe_columns(raw_df)
        is_prod_df, missing_prod_cols = validate_product_dataframe_columns(raw_df)

        cols_lower = [str(c).strip().lower() for c in raw_df.columns]
        is_sku_df = "sku" in cols_lower or "model_type" in cols_lower or "annual_demand" in cols_lower

        if is_class_df:
            # -------------------------------------------------------------
            # 2. PORTFOLIO SUMMARY & MULTICRITERIA ABC CLASSIFICATION
            # -------------------------------------------------------------
            st.divider()
            st.subheader("Portfolio Summary & Multicriteria ABC Classification")

            # Classification parameters
            with st.expander("Classification Method & Threshold Parameters", expanded=False):
                col_meth, col_info = st.columns(2)
                with col_meth:
                    class_method = st.selectbox(
                        "Synthesis Rule",
                        ["matrix", "priority"],
                        format_func=lambda x: (
                            "Flores-Whybark Joint Matrix Mapping"
                            if x == "matrix"
                            else "Joint Priority (min class: highest criticality)"
                        ),
                        help="Flores-Whybark maps joint cells (AA, AB, etc.). Priority rule assigns min(Sales Class, Hits Class).",
                    )
                with col_info:
                    st.caption(
                        "• **Class A**: High Sales & High Hits (tight control, continuous review, prime slotting)\n"
                        "• **Class B**: Moderate/High Sales or Hits (standard replenishment, solves high-hits/low-sales paradox)\n"
                        "• **Class C**: Lower velocity or low value items (streamlined review)\n"
                        "• **Class D**: Tail/slow-moving/obsolete inventory (minimum stocking/phase-out)"
                    )

            # Execute classification calculations using pandas
            classified_df, stats = classify_inventory_dataframe(raw_df, method=class_method)

            # Portfolio Summary Metrics
            p1, p2, p3, p4, p5 = st.columns(5)
            avg_sales_val = stats["total_sales"] / stats["num_rows"] if stats["num_rows"] > 0 else 0.0
            avg_hits_val = stats["total_hits"] / stats["num_rows"] if stats["num_rows"] > 0 else 0.0

            p1.metric("Total SKUs", f"{stats['num_rows']:,}")
            p2.metric("Total Annual Sales", f"${stats['total_sales']:,.2f}")
            p3.metric("Avg Sales / SKU", f"${avg_sales_val:,.2f}")
            p4.metric("Total Warehouse Hits", f"{stats['total_hits']:,.0f}")
            p5.metric("Avg Hits / SKU", f"{avg_hits_val:,.1f}")

            # Top items highlight
            max_sales_row = classified_df.loc[classified_df["Sales ($)"].idxmax()] if len(classified_df) > 0 else None
            max_hits_row = classified_df.loc[classified_df["Hits"].idxmax()] if len(classified_df) > 0 else None
            if max_sales_row is not None and max_hits_row is not None:
                st.caption(
                    f"🏆 **Top Revenue SKU:** `{max_sales_row.get('Item Number', 'N/A')}` ({max_sales_row.get('Item Name', '')}) — "
                    f"**${max_sales_row['Sales ($)']:,.2f}** ({max_sales_row['Sales ($)']/stats['total_sales']*100:.1f}% of total) | "
                    f"⚡ **Top Activity SKU:** `{max_hits_row.get('Item Number', 'N/A')}` ({max_hits_row.get('Item Name', '')}) — "
                    f"**{max_hits_row['Hits']:,.0f} hits** ({max_hits_row['Hits']/stats['total_hits']*100:.1f}% of total)"
                )

            # Stratification Summary Cards
            st.markdown("### Multicriteria ABC Classification Breakdown")
            k1, k2, k3, k4 = st.columns(4)
            c_cnts = stats["class_counts"]
            k1.metric(
                "Class A Items",
                f"{c_cnts.get('A', 0):,} items",
                delta=f"{c_cnts.get('A', 0)/stats['num_rows']*100:.1f}% of items",
            )
            k2.metric(
                "Class B Items",
                f"{c_cnts.get('B', 0):,} items",
                delta=f"{c_cnts.get('B', 0)/stats['num_rows']*100:.1f}% of items",
            )
            k3.metric(
                "Class C Items",
                f"{c_cnts.get('C', 0):,} items",
                delta=f"{c_cnts.get('C', 0)/stats['num_rows']*100:.1f}% of items",
            )
            k4.metric(
                "Class D Items",
                f"{c_cnts.get('D', 0):,} items",
                delta=f"{c_cnts.get('D', 0)/stats['num_rows']*100:.1f}% of items",
            )

            # Stratification Breakdown Table & Crosstab Matrix
            col_t1, col_t2 = st.columns([3, 2])
            with col_t1:
                st.markdown("**Portfolio Stratification Breakdown:**")
                styled_sum = stats["summary_df"].copy()
                st.dataframe(
                    styled_sum.style.format({
                        "% of Items": "{:.1f}%",
                        "Total Sales ($)": "${:,.2f}",
                        "% of Sales": "{:.1f}%",
                        "Total Hits": "{:,.0f}",
                        "% of Hits": "{:.1f}%",
                    }),
                    use_container_width=True,
                )
            with col_t2:
                st.markdown("**Sales vs. Hits Cross-Tabulation Matrix (4×4):**")
                st.dataframe(stats["crosstab_df"], use_container_width=True)

            # Visual distribution charts
            ch1, ch2, ch3 = st.columns(3)
            with ch1:
                st.caption("Annual Sales ($) by Class")
                st.bar_chart(stats["summary_df"].set_index("Class")["Total Sales ($)"])
            with ch2:
                st.caption("Warehouse Hits by Class")
                st.bar_chart(stats["summary_df"].set_index("Class")["Total Hits"])
            with ch3:
                st.caption("Number of SKUs by Class")
                st.bar_chart(stats["summary_df"].set_index("Class")["Item Count"])

            # Classification Table & Download
            st.divider()
            st.subheader("Classified Inventory Records")
            st.markdown(
                "Added Columns: `Sales Cumulative %`, `Sales Class`, `Hits Cumulative %`, "
                "`Hits Class`, `Multicriteria Matrix`, and final `Classification` (A, B, C, D)."
            )

            filter_classes = st.multiselect(
                "Filter Table by Classification:",
                ["A", "B", "C", "D"],
                default=["A", "B", "C", "D"],
            )
            filtered_view = classified_df[classified_df["Classification"].isin(filter_classes)]

            try:
                st.dataframe(
                    filtered_view.style.format({
                        "Sales ($)": "${:,.2f}",
                        "Sales Cumulative": "${:,.2f}",
                        "Sales Cumulative %": "{:.1%}",
                        "Hits": "{:,.0f}",
                        "Hits Cumulative": "{:,.0f}",
                        "Hits Cumulative %": "{:.1%}",
                    }),
                    use_container_width=True,
                )
            except Exception:
                st.dataframe(filtered_view, use_container_width=True)

            st.download_button(
                "Download Classified Inventory CSV",
                data=classified_df.to_csv(index=False),
                file_name="classified_inventory_results.csv",
                mime="text/csv",
                key="btn_download_classified",
            )

        elif is_prod_df:
            # -------------------------------------------------------------
            # PRODUCT INVENTORY ANALYSIS (Units & Unit Cost from Prompt 5)
            # -------------------------------------------------------------
            st.divider()
            st.subheader("Product Inventory Analysis (Prompt 5)")

            analysis = analyze_product_dataframe(raw_df)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Units", f"{analysis['total_units']:,.0f}")
            k2.metric("Average Unit Cost", f"${analysis['avg_unit_cost']:,.2f}")
            k3.metric(
                "Product with Highest Units",
                f"{analysis['highest_units_product']}",
                delta=f"{analysis['highest_units_value']:,.0f} units",
            )
            k4.metric("Total Inventory Value", f"${analysis['total_inventory_value']:,.2f}")

            st.subheader("Total Units by Category")
            cat_c1, cat_c2 = st.columns([1, 1])
            with cat_c1:
                st.dataframe(analysis["category_summary"], use_container_width=True)
            with cat_c2:
                try:
                    chart_df = analysis["category_summary"].set_index("Category")
                    st.bar_chart(chart_df["Units"])
                except Exception:
                    pass

            st.subheader("Dataframe with Calculated Inventory Value (`Units × Unit Cost`)")
            analyzed_df = analysis["analyzed_df"]
            try:
                st.dataframe(
                    analyzed_df.style.format({
                        "Units": "{:,.0f}",
                        "Unit Cost": "${:,.2f}",
                        "Inventory Value": "${:,.2f}",
                    }),
                    use_container_width=True,
                )
            except Exception:
                st.dataframe(analyzed_df, use_container_width=True)

            st.download_button(
                "Download Analyzed Products CSV",
                data=analyzed_df.to_csv(index=False),
                file_name="analyzed_inventory_products.csv",
                mime="text/csv",
                key="btn_download_prod",
            )

        elif is_sku_df:
            _render_sku_decision_analysis_view(raw_df)

        else:
            st.error("❌ Uploaded CSV schema does not match any recognized inventory formats.")
            st.markdown(
                "**Expected Schemas:**\n"
                f"1. **Inventory Classification:** `{', '.join(CLASSIFICATION_REQUIRED_COLUMNS)}` "
                f"(Missing: `{', '.join(missing_class_cols)}`)\n"
                f"2. **Product Analysis:** `{', '.join(PRODUCT_REQUIRED_COLUMNS)}` "
                f"(Missing: `{', '.join(missing_prod_cols)}`)\n"
                f"3. **Multi-Model Decision Agent:** `sku, model_type, annual_demand, ...`"
            )
            st.subheader("Uploaded CSV Preview")
            st.dataframe(raw_df, use_container_width=True)


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
        csv_source = args.csv
        if csv_source == "__SAMPLE__":
            raw_text = SAMPLE_CLASSIFICATION_CSV
            rows = parse_csv_text(raw_text)
        else:
            with open(csv_source, "r", encoding="utf-8") as fh:
                raw_text = fh.read()
            rows = parse_csv_text(raw_text)

        first_row_keys = {str(k).strip().lower() for k in rows[0].keys()} if rows else set()

        # Check for Multicriteria Inventory Classification format (Sales & Hits)
        is_class_format = (
            any("sales" in k for k in first_row_keys)
            and any("hit" in k for k in first_row_keys)
        )
        is_product_format = all(col.lower() in first_row_keys for col in PRODUCT_REQUIRED_COLUMNS)

        if is_class_format:
            if pd is not None:
                df = pd.read_csv(io.StringIO(raw_text))
                valid_class, _ = validate_classification_dataframe_columns(df)
                classified_df, stats = classify_inventory_dataframe(df)
                if args.json:
                    output_data = {
                        "num_rows": stats["num_rows"],
                        "total_sales": stats["total_sales"],
                        "total_hits": stats["total_hits"],
                        "class_counts": stats["class_counts"],
                        "summary": stats["summary_df"].to_dict(orient="records"),
                        "records": classified_df.to_dict(orient="records"),
                    }
                    print(json.dumps(output_data, indent=2, default=str))
                else:
                    print("=" * 72)
                    print("MULTICRITERIA INVENTORY CLASSIFICATION (Sales & Hits -> A, B, C, D)")
                    print("=" * 72)
                    print(f"Total Inventory Items: {stats['num_rows']:,}")
                    print(f"Total Annual Sales:   ${stats['total_sales']:,.2f}")
                    print(f"Total Warehouse Hits:   {stats['total_hits']:,.0f}")
                    print("-" * 72)
                    print("CLASSIFICATION STRATIFICATION BREAKDOWN:")
                    for _, row in stats["summary_df"].iterrows():
                        print(
                            f"  * Class {row['Class']}: {row['Item Count']} items ({row['% of Items']}%) | "
                            f"Sales: ${row['Total Sales ($)']:,.2f} ({row['% of Sales']}%) | "
                            f"Hits: {row['Total Hits']:,.0f} ({row['% of Hits']}%)"
                        )
                    print("\nITEM-LEVEL MULTICRITERIA CLASSIFICATION:")
                    for _, row in classified_df.iterrows():
                        item_num = row.get("Item Number", row.get("item_number", "SKU"))
                        item_name = row.get("Item Name", row.get("item_name", "Item"))
                        print(
                            f"  * {item_num} - {item_name}: Sales=${row['Sales ($)']:,.2f} [Class {row['Sales Class']}] | "
                            f"Hits={row['Hits']:,.0f} [Class {row['Hits Class']}] -> FINAL CLASS: {row['Classification']}"
                        )
            else:
                analysis = classify_inventory_records(rows)
                if args.json:
                    print(json.dumps(analysis, indent=2, default=str))
                else:
                    print("=" * 72)
                    print("MULTICRITERIA INVENTORY CLASSIFICATION (Pure-Python Fallback)")
                    print("=" * 72)
                    print(f"Total Inventory Items: {analysis['num_rows']:,}")
                    print(f"Total Annual Sales:   ${analysis['total_sales']:,.2f}")
                    print(f"Total Warehouse Hits:   {analysis['total_hits']:,.0f}")
                    print("-" * 72)
                    print("CLASSIFICATION STRATIFICATION BREAKDOWN:")
                    for row in analysis["summary_rows"]:
                        print(
                            f"  * Class {row['Class']}: {row['Item Count']} items ({row['% of Items']}%) | "
                            f"Sales: ${row['Total Sales ($)']:,.2f} ({row['% of Sales']}%) | "
                            f"Hits: {row['Total Hits']:,.0f} ({row['% of Hits']}%)"
                        )
                    print("\nITEM-LEVEL MULTICRITERIA CLASSIFICATION:")
                    for row in analysis["rows"]:
                        item_num = row.get("Item Number", "SKU")
                        item_name = row.get("Item Name", "Item")
                        print(
                            f"  * {item_num} - {item_name}: Sales=${row['Sales ($)']:,.2f} [Class {row['Sales Class']}] | "
                            f"Hits={row['Hits']:,.0f} [Class {row['Hits Class']}] -> FINAL CLASS: {row['Classification']}"
                        )
            return 0

        if is_product_format:
            if pd is not None:
                df = pd.read_csv(io.StringIO(raw_text))
                analysis = analyze_product_dataframe(df)
                if args.json:
                    output_data = {
                        "num_rows": analysis["num_rows"],
                        "num_columns": analysis["num_columns"],
                        "column_names": analysis["column_names"],
                        "total_units": analysis["total_units"],
                        "avg_unit_cost": analysis["avg_unit_cost"],
                        "highest_units_product": analysis["highest_units_product"],
                        "highest_units_value": analysis["highest_units_value"],
                        "total_inventory_value": analysis["total_inventory_value"],
                        "category_summary": analysis["category_summary"].to_dict(orient="records"),
                        "records": analysis["analyzed_df"].to_dict(orient="records"),
                    }
                    print(json.dumps(output_data, indent=2, default=str))
                else:
                    print("=" * 72)
                    print("PRODUCT CSV DATA PROCESSING & ANALYSIS (pandas)")
                    print("=" * 72)
                    print(f"Dataset Size: {analysis['num_rows']} rows x {analysis['num_columns']} columns")
                    print(f"Column Names: {', '.join(analysis['column_names'])}")
                    print("-" * 72)
                    print("SUMMARY METRICS:")
                    print(f"  * Total Units: {analysis['total_units']:,.0f}")
                    print(f"  * Average Unit Cost: ${analysis['avg_unit_cost']:,.2f}")
                    print(
                        f"  * Product with Highest Units: {analysis['highest_units_product']} "
                        f"({analysis['highest_units_value']:,.0f} units)"
                    )
                    print(f"  * Total Inventory Value: ${analysis['total_inventory_value']:,.2f}")
                    print("\nTOTAL UNITS BY CATEGORY:")
                    for _, row in analysis["category_summary"].iterrows():
                        print(f"  * {row['Category']}: {row['Units']:,.0f} units")
                    print("\nDATAFRAME WITH INVENTORY VALUE (Units x Unit Cost):")
                    for _, row in analysis["analyzed_df"].iterrows():
                        print(
                            f"  * {row['Product']} ({row['Category']}): "
                            f"{row['Units']:,.0f} units @ ${row['Unit Cost']:,.2f} = "
                            f"${row['Inventory Value']:,.2f}"
                        )
            else:
                analysis = analyze_product_records(rows)
                if args.json:
                    print(json.dumps(analysis, indent=2, default=str))
                else:
                    print("=" * 72)
                    print("PRODUCT CSV DATA PROCESSING & ANALYSIS (Pure-Python Fallback)")
                    print("=" * 72)
                    print(f"Dataset Size: {analysis['num_rows']} rows x {analysis['num_columns']} columns")
                    print(f"Column Names: {', '.join(analysis['column_names'])}")
                    print("-" * 72)
                    print("SUMMARY METRICS:")
                    print(f"  * Total Units: {analysis['total_units']:,.0f}")
                    print(f"  * Average Unit Cost: ${analysis['avg_unit_cost']:,.2f}")
                    print(
                        f"  * Product with Highest Units: {analysis['highest_units_product']} "
                        f"({analysis['highest_units_value']:,.0f} units)"
                    )
                    print(f"  * Total Inventory Value: ${analysis['total_inventory_value']:,.2f}")
                    print("\nTOTAL UNITS BY CATEGORY:")
                    for cat, u in analysis["category_summary"]:
                        print(f"  * {cat}: {u:,.0f} units")
                    print("\nDATAFRAME WITH INVENTORY VALUE (Units x Unit Cost):")
                    for row in analysis["rows"]:
                        print(
                            f"  * {row.get('Product', 'Unknown')} ({row.get('Category', '')}): "
                            f"{row.get('Units', 0):,.0f} units @ ${row.get('Unit Cost', 0):,.2f} = "
                            f"${row.get('Inventory Value', 0):,.2f}"
                        )
            return 0

        # SKU-level multi-model CSV analysis fallback
        if csv_source == "__SAMPLE__":
            rows = load_sample_inventory_rows()
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

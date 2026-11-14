"""Deterministic quantitative inventory analytics tools for supply chain decisions.

Implements five validated analytical tools:
1. calculate_eoq()               - Economic Order Quantity (EOQ)
2. calculate_quantity_discount() - Quantity Discount (Fixed or Percentage Holding Cost)
3. calculate_rop()               - Safety Stock & Reorder Point (all 4 variability regimes)
4. calculate_periodic_review()   - Fixed-Period Review (Order-Up-To Policy)
5. calculate_newsvendor()        - Newsvendor (Single-Period Stochastic Model)

Every tool validates inputs strictly (never inventing missing numerical inputs)
and returns all eight verifiable components directly from Python:
    1. inputs_extracted
    2. variable_definitions
    3. formula
    4. formula_substituted
    5. intermediate_calculations
    6. final_answer
    7. verification_checks
    8. assumptions

Uses only the Python 3.9+ standard library.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple

_STD_NORM = NormalDist(mu=0.0, sigma=1.0)


# ---------------------------------------------------------------------------
# Statistical, Formatting & Validation Helpers
# ---------------------------------------------------------------------------

def norm_cdf(z: float) -> float:
    """Standard normal cumulative distribution function Phi(z)."""
    return _STD_NORM.cdf(z)


def norm_pdf(z: float) -> float:
    """Standard normal probability density function phi(z)."""
    return _STD_NORM.pdf(z)


def norm_inv_cdf(p: float) -> float:
    """Inverse standard normal CDF Phi^{-1}(p) for 0 < p < 1."""
    val = _to_finite_float("probability", p)
    if not (0.0 < val < 1.0):
        raise ValueError(f"Probability must be strictly between 0 and 1, got {p}.")
    return _STD_NORM.inv_cdf(val)


def standard_normal_loss(z: float) -> float:
    """Standard normal loss function L(z) = phi(z) - z * (1 - Phi(z))."""
    return norm_pdf(z) - z * (1.0 - norm_cdf(z))


def solve_z_for_loss(target_loss: float, tol: float = 1e-10) -> float:
    """Solve L(z) = target_loss for z via bisection."""
    if target_loss <= 0.0:
        raise ValueError(f"target_loss must be positive, got {target_loss}.")

    low, high = -10.0, 10.0
    while standard_normal_loss(low) < target_loss:
        low -= 5.0
    while standard_normal_loss(high) > target_loss:
        high += 2.0
        if high > 35.0:
            raise ValueError("Target fill rate is too close to 1.0 for numerical precision.")

    for _ in range(120):
        mid = 0.5 * (low + high)
        val = standard_normal_loss(mid)
        if abs(val - target_loss) <= tol:
            return mid
        if val > target_loss:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _fmt(val: float, decimals: int = 4) -> str:
    """Format a number concisely for formula substitution strings."""
    if math.isclose(val, round(val), abs_tol=1e-9):
        return f"{int(round(val)):,}"
    text = f"{val:,.{decimals}f}".rstrip("0").rstrip(".")
    return text


def _to_finite_float(name: str, value: Any) -> float:
    if value is None:
        raise ValueError(f"Missing required parameter '{name}'.")
    if isinstance(value, bool):
        raise ValueError(f"Parameter '{name}' must be numeric, not boolean ({value}).")
    try:
        val = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Parameter '{name}' must be a finite number, got {value!r}.") from exc
    if not math.isfinite(val):
        raise ValueError(f"Parameter '{name}' must be finite, got {value!r}.")
    return val


def _require_positive(name: str, value: Any) -> float:
    val = _to_finite_float(name, value)
    if val <= 0.0:
        raise ValueError(f"Parameter '{name}' must be positive (> 0), got {value}.")
    return val


def _require_non_negative(name: str, value: Any) -> float:
    val = _to_finite_float(name, value)
    if val < 0.0:
        raise ValueError(f"Parameter '{name}' must be non-negative (>= 0), got {value}.")
    return val


def _require_probability(name: str, value: Any) -> float:
    val = _to_finite_float(name, value)
    if not (0.0 < val < 1.0):
        raise ValueError(f"Parameter '{name}' must be strictly between 0 and 1, got {value}.")
    return val


def _resolve_z_and_csl(
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
) -> Tuple[float, float, str]:
    """Resolve z-score and cycle service level (CSL) from provided inputs."""
    if service_level is not None and z_score is not None:
        csl = _require_probability("service_level", service_level)
        z = _to_finite_float("z_score", z_score)
        if not math.isclose(norm_cdf(z), csl, rel_tol=1e-2, abs_tol=1e-2):
            raise ValueError(
                f"Conflicting inputs: service_level={csl} implies z={norm_inv_cdf(csl):.4f}, "
                f"which disagrees with z_score={z}."
            )
        return z, norm_cdf(z), "service_level_and_z_score"
    if service_level is not None:
        csl = _require_probability("service_level", service_level)
        return norm_inv_cdf(csl), csl, "service_level"
    if z_score is not None:
        z = _to_finite_float("z_score", z_score)
        return z, norm_cdf(z), "z_score"
    raise ValueError("Provide either 'service_level' (strictly in (0, 1)) or 'z_score'.")


# ---------------------------------------------------------------------------
# 1. Economic Order Quantity (EOQ): Q = sqrt(2DS / H)
# ---------------------------------------------------------------------------

def calculate_eoq(
    demand: float,
    ordering_cost: float,
    holding_cost: Optional[float] = None,
    unit_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    operating_days_per_year: float = 365.0,
) -> Dict[str, Any]:
    """Calculate the classic Economic Order Quantity: Q = sqrt(2 * D * S / H)."""
    d = _require_positive("demand", demand)
    s = _require_positive("ordering_cost", ordering_cost)
    op_days = _require_positive("operating_days_per_year", operating_days_per_year)

    c: Optional[float] = None
    if unit_cost is not None:
        c = _require_positive("unit_cost", unit_cost)

    i_rate: Optional[float] = None
    if holding_rate is not None:
        i_rate = _require_positive("holding_rate", holding_rate)

    assumptions: List[str] = [
        "Demand rate D is constant and deterministic across the year.",
        "Ordering cost S and annual holding cost H are constant; replenishment is instantaneous with no shortages.",
    ]

    if holding_cost is not None:
        h = _require_positive("holding_cost", holding_cost)
        holding_cost_source = "explicit"
        if c is not None and i_rate is not None and not math.isclose(h, c * i_rate, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(
                f"Conflicting holding inputs: holding_cost={h} != unit_cost*holding_rate={c * i_rate}."
            )
    elif c is not None and i_rate is not None:
        h = c * i_rate
        holding_cost_source = "unit_cost * holding_rate"
        assumptions.append(f"Annual holding cost H computed as C * i = {_fmt(c)} * {_fmt(i_rate)} = ${_fmt(h)}/unit/year.")
    else:
        raise ValueError(
            "Provide either 'holding_cost' ($/unit/year) or both 'unit_cost' and 'holding_rate'."
        )

    two_d_s = 2.0 * d * s
    two_d_s_over_h = two_d_s / h
    eoq = math.sqrt(two_d_s_over_h)
    rounded_eoq = max(1, int(round(eoq)))

    orders_per_year = d / eoq
    cycle_time_years = eoq / d
    cycle_time_days = cycle_time_years * op_days
    average_inventory = eoq / 2.0

    annual_ordering_cost = (d / eoq) * s
    annual_holding_cost = (eoq / 2.0) * h
    total_relevant_cost = annual_ordering_cost + annual_holding_cost

    rounded_ordering_cost = (d / rounded_eoq) * s
    rounded_holding_cost = (rounded_eoq / 2.0) * h
    rounded_relevant_cost = rounded_ordering_cost + rounded_holding_cost

    annual_purchase_cost: Optional[float] = None
    total_annual_cost: Optional[float] = None
    if c is not None:
        annual_purchase_cost = d * c
        total_annual_cost = annual_purchase_cost + total_relevant_cost

    deterministic_rop: Optional[float] = None
    lt_clean: Optional[float] = None
    if lead_time_days is not None:
        lt_clean = _require_non_negative("lead_time_days", lead_time_days)
        daily_demand = d / op_days
        deterministic_rop = daily_demand * lt_clean
        assumptions.append(f"Cycle time and deterministic ROP use {op_days:g} operating days/year.")
    else:
        assumptions.append(f"Cycle time in days uses {op_days:g} days/year.")

    inputs_extracted: Dict[str, Any] = {
        "demand": d,
        "ordering_cost": s,
        "holding_cost": h,
        "unit_cost": c,
        "holding_rate": i_rate,
        "lead_time_days": lt_clean,
        "operating_days_per_year": op_days,
    }

    variable_definitions: Dict[str, str] = {
        "D": "Annual demand (units/year)",
        "S": "Fixed ordering / setup cost per order ($/order)",
        "H": "Annual holding cost per unit per year ($/unit/year)",
        "Q*": "Economic Order Quantity (units/order)",
        "N": "Expected number of orders per year (D / Q*)",
        "T": "Time between orders / cycle length",
        "TC_rel": "Total relevant annual cost (annual ordering cost + annual holding cost)",
    }
    if c is not None:
        variable_definitions["C"] = "Unit purchase cost ($/unit)"
        variable_definitions["TC_total"] = "Total annual cost including purchase cost (D * C + TC_rel)"
    if i_rate is not None:
        variable_definitions["i"] = "Annual holding cost rate (fraction of unit purchase cost)"
    if lt_clean is not None:
        variable_definitions["L"] = "Deterministic replenishment lead time (days)"
        variable_definitions["ROP"] = "Deterministic reorder point ((D / operating_days) * L)"

    formula: Dict[str, str] = {
        "eoq": "Q* = sqrt(2 * D * S / H)",
        "orders_per_year": "N = D / Q*",
        "cycle_time_days": "T_days = operating_days * (Q* / D)",
        "annual_ordering_cost": "C_order = (D / Q*) * S",
        "annual_holding_cost": "C_hold = (Q* / 2) * H",
        "total_relevant_cost": "TC_rel = C_order + C_hold",
    }
    if holding_cost_source == "unit_cost * holding_rate":
        formula["holding_cost"] = "H = C * i"
    if total_annual_cost is not None:
        formula["total_annual_cost"] = "TC_total = (D * C) + TC_rel"
    if deterministic_rop is not None:
        formula["deterministic_rop"] = "ROP = (D / operating_days) * L"

    formula_substituted: Dict[str, str] = {}
    if holding_cost_source == "unit_cost * holding_rate" and c is not None and i_rate is not None:
        formula_substituted["holding_cost"] = f"H = {_fmt(c)} * {_fmt(i_rate)} = ${_fmt(h)}/unit/year"
    formula_substituted["eoq"] = (
        f"Q* = sqrt(2 * {_fmt(d)} * {_fmt(s)} / {_fmt(h)}) = "
        f"sqrt({_fmt(two_d_s)} / {_fmt(h)}) = sqrt({_fmt(two_d_s_over_h)}) = {_fmt(eoq)} units"
    )
    formula_substituted["orders_per_year"] = f"N = {_fmt(d)} / {_fmt(eoq)} = {_fmt(orders_per_year)} orders/year"
    formula_substituted["cycle_time_days"] = (
        f"T_days = {_fmt(op_days)} * ({_fmt(eoq)} / {_fmt(d)}) = {_fmt(cycle_time_days)} days"
    )
    formula_substituted["annual_ordering_cost"] = (
        f"C_order = ({_fmt(d)} / {_fmt(eoq)}) * {_fmt(s)} = ${_fmt(annual_ordering_cost, 2)}"
    )
    formula_substituted["annual_holding_cost"] = (
        f"C_hold = ({_fmt(eoq)} / 2) * {_fmt(h)} = ${_fmt(annual_holding_cost, 2)}"
    )
    formula_substituted["total_relevant_cost"] = (
        f"TC_rel = ${_fmt(annual_ordering_cost, 2)} + ${_fmt(annual_holding_cost, 2)} = ${_fmt(total_relevant_cost, 2)}"
    )
    if total_annual_cost is not None and annual_purchase_cost is not None and c is not None:
        formula_substituted["total_annual_cost"] = (
            f"TC_total = ({_fmt(d)} * ${_fmt(c, 2)}) + ${_fmt(total_relevant_cost, 2)} = ${_fmt(total_annual_cost, 2)}"
        )
    if deterministic_rop is not None and lt_clean is not None:
        formula_substituted["deterministic_rop"] = (
            f"ROP = ({_fmt(d)} / {_fmt(op_days)}) * {_fmt(lt_clean)} = {_fmt(deterministic_rop)} units"
        )

    intermediate_calculations = {
        "holding_cost_source": holding_cost_source,
        "holding_cost_H": h,
        "two_D_S": two_d_s,
        "two_D_S_over_H": two_d_s_over_h,
        "orders_per_year": orders_per_year,
        "cycle_time_years": cycle_time_years,
        "cycle_time_days": cycle_time_days,
        "average_inventory": average_inventory,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "rounded_relevant_cost": rounded_relevant_cost,
    }
    final_answer = {
        "eoq": eoq,
        "rounded_eoq": rounded_eoq,
        "total_relevant_cost": total_relevant_cost,
        "total_annual_cost": total_annual_cost,
        "deterministic_rop": deterministic_rop,
    }

    tc_low = (d / (0.9 * eoq)) * s + (0.9 * eoq / 2.0) * h
    tc_high = (d / (1.1 * eoq)) * s + (1.1 * eoq / 2.0) * h
    verification_checks = [
        {
            "name": "ordering_holding_cost_balance",
            "passed": math.isclose(annual_ordering_cost, annual_holding_cost, rel_tol=1e-6, abs_tol=1e-6),
            "detail": f"Annual ordering cost (${annual_ordering_cost:,.2f}) == holding cost (${annual_holding_cost:,.2f}).",
        },
        {
            "name": "eoq_local_minimum",
            "passed": total_relevant_cost < tc_low and total_relevant_cost < tc_high,
            "detail": f"TC_rel(Q*) = ${total_relevant_cost:,.2f} < neighbors 0.9Q* (${tc_low:,.2f}) and 1.1Q* (${tc_high:,.2f}).",
        },
        {
            "name": "positive_finite_outputs",
            "passed": eoq > 0.0 and orders_per_year > 0.0 and cycle_time_days > 0.0,
            "detail": f"Q*={eoq:.2f} > 0, orders/yr={orders_per_year:.2f} > 0, cycle={cycle_time_days:.2f} days > 0.",
        },
    ]

    return {
        "model": "eoq",
        "inputs_extracted": inputs_extracted,
        "inputs_validated": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_substituted": formula_substituted,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Flat convenience fields
        "eoq": eoq,
        "rounded_eoq": rounded_eoq,
        "demand": d,
        "ordering_cost": s,
        "holding_cost": h,
        "holding_cost_source": holding_cost_source,
        "unit_cost": c,
        "holding_rate": i_rate,
        "orders_per_year": orders_per_year,
        "cycle_time_years": cycle_time_years,
        "cycle_time_days": cycle_time_days,
        "operating_days_per_year": op_days,
        "average_inventory": average_inventory,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "total_relevant_cost": total_relevant_cost,
        "rounded_relevant_cost": rounded_relevant_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "total_annual_cost": total_annual_cost,
        "deterministic_rop": deterministic_rop,
    }


# ---------------------------------------------------------------------------
# 2. Quantity Discount (Fixed Holding Cost & Percentage Holding Cost)
# ---------------------------------------------------------------------------

def _normalize_tiers(
    tiers: Sequence[Dict[str, Any]],
    holding_cost: Optional[float],
    holding_rate: Optional[float],
) -> Tuple[List[Dict[str, Any]], str]:
    if not isinstance(tiers, (list, tuple)) or not tiers:
        raise ValueError("Provide a non-empty list of price-break 'tiers'.")

    if holding_cost is not None and holding_rate is not None:
        raise ValueError("Provide either fixed 'holding_cost' or percentage 'holding_rate', not both.")

    fixed_h: Optional[float] = None
    rate_i: Optional[float] = None
    if holding_cost is not None:
        fixed_h = _require_positive("holding_cost", holding_cost)
    if holding_rate is not None:
        rate_i = _require_positive("holding_rate", holding_rate)

    normalized: List[Dict[str, Any]] = []
    holding_modes_seen = set()

    for idx, tier in enumerate(tiers):
        if not isinstance(tier, dict):
            raise ValueError(f"Tier {idx} must be a dictionary, got {type(tier).__name__}.")
        if "unit_cost" not in tier or tier["unit_cost"] is None:
            raise ValueError(f"Tier {idx} is missing required 'unit_cost'.")
        if "min_qty" not in tier or tier["min_qty"] is None:
            raise ValueError(f"Tier {idx} is missing required 'min_qty'.")

        unit_cost = _require_positive(f"tiers[{idx}].unit_cost", tier["unit_cost"])
        min_qty = _require_non_negative(f"tiers[{idx}].min_qty", tier["min_qty"])
        effective_min = 1.0 if min_qty == 0.0 else float(min_qty)

        raw_max = tier.get("max_qty")
        max_qty: Optional[float] = None
        if raw_max is not None and not (isinstance(raw_max, float) and math.isinf(raw_max)):
            max_qty = _require_positive(f"tiers[{idx}].max_qty", raw_max)
            if max_qty < min_qty:
                raise ValueError(f"Tier {idx} has max_qty ({max_qty}) < min_qty ({min_qty}).")

        if tier.get("holding_cost") is not None:
            h = _require_positive(f"tiers[{idx}].holding_cost", tier["holding_cost"])
            holding_modes_seen.add("per_tier_fixed")
        elif fixed_h is not None:
            h = fixed_h
            holding_modes_seen.add("fixed_holding_cost")
        elif rate_i is not None:
            h = rate_i * unit_cost
            holding_modes_seen.add("percentage_holding_cost")
        else:
            raise ValueError(
                "Provide fixed 'holding_cost' ($/unit/year), percentage 'holding_rate' (fraction of price), "
                "or per-tier 'holding_cost'."
            )

        normalized.append(
            {
                "tier_index": idx,
                "min_qty": float(min_qty),
                "effective_min_qty": effective_min,
                "max_qty": max_qty,
                "unit_cost": unit_cost,
                "holding_cost": h,
            }
        )

    normalized.sort(key=lambda item: (item["min_qty"], -item["unit_cost"]))
    for pos, item in enumerate(normalized):
        item["tier_index"] = pos
        if pos > 0:
            prev = normalized[pos - 1]
            if prev["max_qty"] is None:
                raise ValueError(
                    f"Tier {pos - 1} has unbounded max_qty=None but is followed by tier {pos} (min_qty={item['min_qty']})."
                )
            if item["min_qty"] <= prev["max_qty"]:
                raise ValueError(
                    f"Overlapping quantity tiers: tier {pos - 1} max_qty={prev['max_qty']} "
                    f"overlaps tier {pos} min_qty={item['min_qty']}."
                )

    mode_label = ", ".join(sorted(holding_modes_seen))
    return normalized, mode_label


def calculate_quantity_discount(
    demand: float,
    ordering_cost: float,
    tiers: Sequence[Dict[str, Any]],
    holding_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
    discount_type: str = "all_units",
) -> Dict[str, Any]:
    """Evaluate feasible quantities and compare total annual cost across price tiers.

    Supports both fixed holding cost (`holding_cost`) and percentage holding cost
    (`holding_rate`, where H_j = holding_rate * unit_cost_j).
    """
    d = _require_positive("demand", demand)
    s = _require_positive("ordering_cost", ordering_cost)
    if not isinstance(discount_type, str):
        raise ValueError("discount_type must be a string ('all_units' or 'incremental').")
    mode = discount_type.strip().lower().replace("-", "_")
    if mode not in ("all_units", "incremental"):
        raise ValueError(f"Unsupported discount_type '{discount_type}'. Use 'all_units' or 'incremental'.")

    norm_tiers, holding_mode = _normalize_tiers(tiers, holding_cost=holding_cost, holding_rate=holding_rate)
    tier_evaluations: List[Dict[str, Any]] = []
    assumptions: List[str] = [
        "Demand rate D and ordering cost S are deterministic and constant; no shortages allowed.",
    ]

    if mode == "all_units":
        assumptions.append("All-units discount: the tier unit price applies to every unit in the order.")
    else:
        assumptions.append("Incremental discount: each unit price applies only to units within that quantity bracket.")

    if holding_mode == "percentage_holding_cost":
        assumptions.append(f"Percentage holding cost applied: H_j = {holding_rate:g} * c_j per unit per year.")
    elif holding_mode == "fixed_holding_cost":
        assumptions.append(f"Fixed annual holding cost applied: H = ${_fmt(float(holding_cost), 2)}/unit/year across all tiers.")
    else:
        assumptions.append("Explicit per-tier annual holding cost applied.")

    if mode == "all_units":
        for tier in norm_tiers:
            c = tier["unit_cost"]
            h = tier["holding_cost"]
            q_min = tier["effective_min_qty"]
            q_max = tier["max_qty"]

            raw_eoq = math.sqrt((2.0 * d * s) / h)
            is_raw_feasible = (raw_eoq >= q_min) and (q_max is None or raw_eoq <= q_max)

            if raw_eoq < q_min:
                candidate_q = q_min
                feasibility = "adjusted_to_min_qty"
            elif q_max is not None and raw_eoq > q_max:
                candidate_q = q_max
                feasibility = "adjusted_to_max_qty"
            else:
                candidate_q = raw_eoq
                feasibility = "feasible_eoq"

            purchase_cost = d * c
            order_cost = (d / candidate_q) * s
            hold_cost = (candidate_q / 2.0) * h
            total_cost = purchase_cost + order_cost + hold_cost

            tier_evaluations.append(
                {
                    "tier_index": tier["tier_index"],
                    "min_qty": tier["min_qty"],
                    "max_qty": q_max,
                    "unit_cost": c,
                    "holding_cost_per_unit": h,
                    "raw_eoq": raw_eoq,
                    "is_raw_eoq_feasible": is_raw_feasible,
                    "candidate_qty": candidate_q,
                    "feasibility": feasibility,
                    "annual_purchase_cost": purchase_cost,
                    "annual_ordering_cost": order_cost,
                    "annual_holding_cost": hold_cost,
                    "total_annual_cost": total_cost,
                }
            )
    else:
        if holding_rate is None and holding_cost is None:
            raise ValueError("Incremental quantity discount requires 'holding_rate' or 'holding_cost'.")
        i_rate = _require_positive("holding_rate", holding_rate) if holding_rate is not None else None
        fixed_h = _require_positive("holding_cost", holding_cost) if holding_cost is not None else None

        accumulated_fixed = 0.0
        prev_breakpoint = 0.0
        for idx, tier in enumerate(norm_tiers):
            c = tier["unit_cost"]
            raw_min = tier["min_qty"]
            b_j = 0.0 if idx == 0 else (raw_min - 1.0 if raw_min >= 1.0 and norm_tiers[0]["min_qty"] == 1.0 else raw_min)
            if idx > 0:
                prev_c = norm_tiers[idx - 1]["unit_cost"]
                accumulated_fixed += prev_c * (b_j - prev_breakpoint)
                prev_breakpoint = b_j

            fixed_offset = accumulated_fixed - c * b_j
            h = (i_rate * c) if i_rate is not None else fixed_h
            effective_s = s + fixed_offset
            raw_eoq: Optional[float] = None
            candidate_q: Optional[float] = None
            q_max = tier["max_qty"]
            is_raw_feasible = False

            if effective_s > 0.0:
                raw_eoq = math.sqrt((2.0 * d * effective_s) / h)
                in_lower = raw_eoq >= tier["effective_min_qty"]
                in_upper = q_max is None or raw_eoq <= q_max
                is_raw_feasible = in_lower and in_upper
                if is_raw_feasible:
                    candidate_q = raw_eoq
                    feasibility = "feasible_eoq"
                elif not in_lower:
                    candidate_q = tier["effective_min_qty"]
                    feasibility = "infeasible_below_min_checked"
                else:
                    candidate_q = q_max
                    feasibility = "infeasible_above_max_checked"
            else:
                feasibility = "no_interior_minimum"

            if candidate_q is not None:
                v_q = accumulated_fixed + c * (candidate_q - b_j)
                avg_unit_cost = v_q / candidate_q
                purchase_cost = d * avg_unit_cost
                order_cost = (d / candidate_q) * s
                eff_h = (i_rate * avg_unit_cost) if i_rate is not None else fixed_h
                hold_cost = (candidate_q / 2.0) * eff_h
                total_cost = purchase_cost + order_cost + hold_cost
            else:
                avg_unit_cost = c
                eff_h = (i_rate * c) if i_rate is not None else fixed_h
                purchase_cost = float("inf")
                order_cost = float("inf")
                hold_cost = float("inf")
                total_cost = float("inf")

            tier_evaluations.append(
                {
                    "tier_index": tier["tier_index"],
                    "min_qty": tier["min_qty"],
                    "max_qty": q_max,
                    "unit_cost": c,
                    "average_unit_cost": avg_unit_cost,
                    "holding_cost_per_unit": eff_h,
                    "raw_eoq": raw_eoq,
                    "is_raw_eoq_feasible": is_raw_feasible,
                    "candidate_qty": candidate_q,
                    "feasibility": feasibility,
                    "annual_purchase_cost": purchase_cost,
                    "annual_ordering_cost": order_cost,
                    "annual_holding_cost": hold_cost,
                    "total_annual_cost": total_cost,
                }
            )

    valid_evals = [e for e in tier_evaluations if math.isfinite(e["total_annual_cost"])]
    if not valid_evals:
        raise ValueError("No valid candidate order quantities found across discount tiers.")

    best = min(valid_evals, key=lambda e: (e["total_annual_cost"], -e["candidate_qty"]))
    baseline = tier_evaluations[0]
    savings_vs_tier0 = baseline["total_annual_cost"] - best["total_annual_cost"]
    optimal_qty = best["candidate_qty"]
    orders_per_year = d / optimal_qty

    inputs_extracted = {
        "demand": d,
        "ordering_cost": s,
        "holding_cost": holding_cost,
        "holding_rate": holding_rate,
        "discount_type": mode,
        "tiers": [
            {
                "min_qty": t["min_qty"],
                "max_qty": t["max_qty"],
                "unit_cost": t["unit_cost"],
                "holding_cost": t["holding_cost"],
            }
            for t in norm_tiers
        ],
    }

    variable_definitions = {
        "D": "Annual demand (units/year)",
        "S": "Fixed ordering cost per order ($/order)",
        "c_j": "Unit purchase price in tier j ($/unit)",
        "H_j": "Annual holding cost per unit in tier j ($/unit/year)",
        "Q_j": "Unconstrained EOQ for tier j",
        "Q_j*": "Feasible / boundary candidate order quantity for tier j",
        "TC_j": "Total annual cost for tier j (purchase + ordering + holding)",
        "Q*": "Cost-minimizing optimal order quantity across all tiers",
    }
    if holding_rate is not None:
        variable_definitions["i"] = "Annual percentage holding cost rate (fraction of unit purchase price)"

    formula: Dict[str, str] = {
        "holding_cost_j": "H_j = i * c_j" if holding_mode == "percentage_holding_cost" else "H_j = H (fixed $/unit/year)",
        "tier_eoq": "Q_j = sqrt(2 * D * S / H_j)" if mode == "all_units" else "Q_j = sqrt(2 * D * (S + F_j - c_j * b_j) / H_j)",
        "feasibility_adjustment": "Q_j* = clamp(Q_j, min_qty_j, max_qty_j)",
        "total_annual_cost": "TC_j = (D * c_j) + (D / Q_j*) * S + (Q_j* / 2) * H_j",
        "optimal_selection": "Q* = argmin_{j} TC_j(Q_j*)",
    }

    formula_substituted: Dict[str, str] = {}
    for t in tier_evaluations:
        j = t["tier_index"] + 1
        hi = _fmt(t["max_qty"], 0) if t["max_qty"] is not None else "inf"
        raw_str = _fmt(t["raw_eoq"]) if t["raw_eoq"] is not None else "N/A"
        formula_substituted[f"tier_{j}"] = (
            f"Tier {j} [{_fmt(t['min_qty'], 0)}..{hi}] @ ${_fmt(t['unit_cost'], 2)}: "
            f"H_{j}=${_fmt(t['holding_cost_per_unit'], 4)}, Q_{j}={raw_str} -> Q_{j}*={_fmt(t['candidate_qty'])}; "
            f"TC_{j} = ${_fmt(t['annual_purchase_cost'], 2)} + ${_fmt(t['annual_ordering_cost'], 2)} + "
            f"${_fmt(t['annual_holding_cost'], 2)} = ${_fmt(t['total_annual_cost'], 2)}"
        )
    formula_substituted["optimal_selection"] = (
        f"Minimum TC = ${_fmt(best['total_annual_cost'], 2)} at Tier {best['tier_index'] + 1} "
        f"(Q* = {_fmt(optimal_qty)} units @ ${_fmt(best['unit_cost'], 2)}/unit)"
    )

    intermediate_calculations = {
        "holding_cost_mode": holding_mode,
        "baseline_total_cost": baseline["total_annual_cost"],
        "tier_evaluations": tier_evaluations,
    }
    final_answer = {
        "optimal_order_quantity": optimal_qty,
        "rounded_order_quantity": max(1, int(round(optimal_qty))),
        "optimal_tier_index": best["tier_index"],
        "optimal_unit_cost": best["unit_cost"],
        "orders_per_year": orders_per_year,
        "annual_purchase_cost": best["annual_purchase_cost"],
        "annual_ordering_cost": best["annual_ordering_cost"],
        "annual_holding_cost": best["annual_holding_cost"],
        "total_annual_cost": best["total_annual_cost"],
        "annual_savings_vs_base": savings_vs_tier0,
    }

    min_tc = min(e["total_annual_cost"] for e in valid_evals)
    chosen_eval = tier_evaluations[best["tier_index"]]
    in_bounds = optimal_qty + 1e-9 >= chosen_eval["min_qty"] and (
        chosen_eval["max_qty"] is None or optimal_qty - 1e-9 <= chosen_eval["max_qty"]
    )
    comp_sum = best["annual_purchase_cost"] + best["annual_ordering_cost"] + best["annual_holding_cost"]

    verification_checks = [
        {
            "name": "global_minimum_across_tiers",
            "passed": math.isclose(best["total_annual_cost"], min_tc, rel_tol=1e-9, abs_tol=1e-9),
            "detail": f"Selected Tier {best['tier_index'] + 1} achieves lowest total cost (${best['total_annual_cost']:,.2f}).",
        },
        {
            "name": "tier_quantity_feasibility",
            "passed": in_bounds,
            "detail": (
                f"Order quantity Q*={optimal_qty:,.2f} lies within tier bounds "
                f"[{chosen_eval['min_qty']:g}, {chosen_eval['max_qty'] if chosen_eval['max_qty'] is not None else 'inf'}]."
            ),
        },
        {
            "name": "cost_component_identity",
            "passed": math.isclose(comp_sum, best["total_annual_cost"], rel_tol=1e-6, abs_tol=1e-6),
            "detail": f"Purchase + ordering + holding = ${comp_sum:,.2f} == total annual cost (${best['total_annual_cost']:,.2f}).",
        },
    ]

    return {
        "model": "quantity_discount",
        "inputs_extracted": inputs_extracted,
        "inputs_validated": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_substituted": formula_substituted,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Flat convenience fields
        "discount_type": mode,
        "holding_cost_mode": holding_mode,
        "demand": d,
        "ordering_cost": s,
        "holding_cost": holding_cost,
        "holding_rate": holding_rate,
        "optimal_order_quantity": optimal_qty,
        "rounded_order_quantity": max(1, int(round(optimal_qty))),
        "optimal_tier_index": best["tier_index"],
        "optimal_unit_cost": best["unit_cost"],
        "orders_per_year": orders_per_year,
        "annual_purchase_cost": best["annual_purchase_cost"],
        "annual_ordering_cost": best["annual_ordering_cost"],
        "annual_holding_cost": best["annual_holding_cost"],
        "total_annual_cost": best["total_annual_cost"],
        "baseline_total_cost": baseline["total_annual_cost"],
        "annual_savings_vs_base": savings_vs_tier0,
        "tier_evaluations": tier_evaluations,
    }


# ---------------------------------------------------------------------------
# 3. Safety Stock and Reorder Point: calculate_rop()
# ---------------------------------------------------------------------------

def calculate_rop(
    avg_demand: float,
    demand_std: Optional[float] = None,
    lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    fill_rate: Optional[float] = None,
    order_quantity: Optional[float] = None,
    lead_time_std: Optional[float] = None,
    time_unit: str = "days",
) -> Dict[str, Any]:
    """Calculate Safety Stock (SS) and Reorder Point (ROP) across all 4 regimes.

    Supported regimes:
        1. Constant demand and constant lead time   (demand_std = 0, lead_time_std = 0)
        2. Variable demand and constant lead time   (demand_std > 0, lead_time_std = 0)
        3. Constant demand and variable lead time   (demand_std = 0, lead_time_std > 0)
        4. Variable demand and variable lead time   (demand_std > 0, lead_time_std > 0)

    Formulas:
        sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        SS = z * sigma_LT
        ROP = d * L + SS
    """
    d_bar = _require_positive("avg_demand", avg_demand)
    lt = _require_positive("lead_time", lead_time)

    if demand_std is None and lead_time_std is None:
        raise ValueError(
            "Provide 'demand_std' and/or 'lead_time_std' (pass 0.0 explicitly for constant demand and constant lead time)."
        )

    assumptions: List[str] = [
        f"Average demand rate and lead time share consistent base unit '{time_unit}'.",
    ]

    if demand_std is None:
        sigma_d = 0.0
        assumptions.append("Demand assumed constant (sigma_d = 0.0) as only lead_time_std was provided.")
    else:
        sigma_d = _require_non_negative("demand_std", demand_std)

    if lead_time_std is None:
        sigma_l = 0.0
        assumptions.append("Lead time assumed constant (sigma_L = 0.0) as only demand_std was provided.")
    else:
        sigma_l = _require_non_negative("lead_time_std", lead_time_std)

    if sigma_d == 0.0 and sigma_l == 0.0:
        demand_regime = "constant_demand_constant_lead_time"
        assumptions.append("Both demand and lead time are deterministic; safety stock SS = 0 and ROP = d * L.")
    elif sigma_d > 0.0 and sigma_l == 0.0:
        demand_regime = "variable_demand_constant_lead_time"
        assumptions.append("Period demands are i.i.d. normal; lead time is constant.")
    elif sigma_d == 0.0 and sigma_l > 0.0:
        demand_regime = "constant_demand_variable_lead_time"
        assumptions.append("Demand per period is constant; lead time is normally distributed.")
    else:
        demand_regime = "variable_demand_variable_lead_time"
        assumptions.append("Demand and lead time are independent; lead-time demand is approximated as normal.")

    lead_time_demand_mean = d_bar * lt
    lead_time_demand_variance = lt * (sigma_d ** 2) + (d_bar ** 2) * (sigma_l ** 2)
    lead_time_demand_std = math.sqrt(lead_time_demand_variance)

    q_val: Optional[float] = None
    if order_quantity is not None:
        q_val = _require_positive("order_quantity", order_quantity)

    target_fill_rate: Optional[float] = None
    z: float
    csl: float
    service_criterion: str

    if lead_time_demand_std == 0.0:
        z = _to_finite_float("z_score", z_score) if z_score is not None else 0.0
        csl = 1.0
        service_criterion = "deterministic"
        safety_stock = 0.0
        expected_shortage_per_cycle = 0.0
        implied_fill_rate = 1.0 if q_val is not None else None
    else:
        if fill_rate is not None and service_level is None and z_score is None:
            beta = _require_probability("fill_rate", fill_rate)
            if q_val is None:
                raise ValueError("order_quantity (Q) is required when targeting a fill rate.")
            target_loss = ((1.0 - beta) * q_val) / lead_time_demand_std
            z = solve_z_for_loss(target_loss)
            csl = norm_cdf(z)
            service_criterion = "fill_rate"
            target_fill_rate = beta
        else:
            z, csl, service_criterion = _resolve_z_and_csl(
                service_level=service_level,
                z_score=z_score,
            )

        safety_stock = z * lead_time_demand_std
        expected_shortage_per_cycle = lead_time_demand_std * standard_normal_loss(z)
        implied_fill_rate = (
            max(0.0, 1.0 - (expected_shortage_per_cycle / q_val)) if q_val is not None else None
        )

    reorder_point = lead_time_demand_mean + safety_stock
    rounded_ss = int(math.ceil(safety_stock)) if safety_stock >= 0.0 else int(round(safety_stock))
    rounded_rop = int(math.ceil(reorder_point)) if reorder_point >= 0.0 else int(round(reorder_point))

    inputs_extracted = {
        "avg_demand": d_bar,
        "demand_std": sigma_d,
        "lead_time": lt,
        "lead_time_std": sigma_l,
        "service_level": service_level,
        "z_score": z_score,
        "fill_rate": fill_rate,
        "order_quantity": q_val,
        "time_unit": time_unit,
    }

    variable_definitions = {
        "d": f"Average demand per {time_unit[:-1]} (units/{time_unit[:-1]})",
        "sigma_d": f"Standard deviation of demand per {time_unit[:-1]}",
        "L": f"Average replenishment lead time ({time_unit})",
        "sigma_L": f"Standard deviation of lead time ({time_unit})",
        "mu_LT": "Mean demand during lead time (d * L)",
        "sigma_LT": "Standard deviation of demand during lead time",
        "z": "Safety factor corresponding to target service level",
        "SS": "Safety stock (units)",
        "ROP": "Reorder point (units)",
    }

    formula = {
        "lead_time_demand_mean": "mu_LT = d * L",
        "lead_time_demand_std": "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)",
        "safety_stock": "SS = z * sigma_LT",
        "reorder_point": "ROP = d * L + SS",
    }

    formula_substituted: Dict[str, str] = {
        "lead_time_demand_mean": f"mu_LT = {_fmt(d_bar)} * {_fmt(lt)} = {_fmt(lead_time_demand_mean)} units",
        "lead_time_demand_std": (
            f"sigma_LT = sqrt({_fmt(lt)} * {_fmt(sigma_d)}^2 + {_fmt(d_bar)}^2 * {_fmt(sigma_l)}^2) = "
            f"sqrt({_fmt(lead_time_demand_variance)}) = {_fmt(lead_time_demand_std)} units"
        ),
        "safety_stock": (
            "SS = 0 (deterministic regime)"
            if service_criterion == "deterministic"
            else f"SS = {_fmt(z)} * {_fmt(lead_time_demand_std)} = {_fmt(safety_stock)} units"
        ),
        "reorder_point": f"ROP = {_fmt(lead_time_demand_mean)} + {_fmt(safety_stock)} = {_fmt(reorder_point)} units",
    }

    intermediate_calculations = {
        "demand_regime": demand_regime,
        "lead_time_demand_mean": lead_time_demand_mean,
        "lead_time_demand_variance": lead_time_demand_variance,
        "lead_time_demand_std": lead_time_demand_std,
        "service_criterion": service_criterion,
        "z_score": z,
        "cycle_service_level": csl,
        "target_fill_rate": target_fill_rate,
        "expected_shortage_per_cycle": expected_shortage_per_cycle,
        "implied_fill_rate": implied_fill_rate,
    }
    final_answer = {
        "safety_stock": safety_stock,
        "rounded_safety_stock": rounded_ss,
        "reorder_point": reorder_point,
        "rounded_reorder_point": rounded_rop,
    }

    verification_checks = [
        {
            "name": "sigma_lt_identity",
            "passed": math.isclose(
                lead_time_demand_std,
                math.sqrt(lt * (sigma_d ** 2) + (d_bar ** 2) * (sigma_l ** 2)),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ),
            "detail": f"sigma_LT = {lead_time_demand_std:.4f} matches sqrt(L*sigma_d^2 + d^2*sigma_L^2).",
        },
        {
            "name": "rop_identity",
            "passed": math.isclose(reorder_point, lead_time_demand_mean + safety_stock, rel_tol=1e-9, abs_tol=1e-9),
            "detail": f"ROP ({reorder_point:.2f}) == mu_LT ({lead_time_demand_mean:.2f}) + SS ({safety_stock:.2f}).",
        },
    ]
    if service_criterion == "deterministic":
        verification_checks.append(
            {
                "name": "deterministic_safety_stock",
                "passed": math.isclose(safety_stock, 0.0, abs_tol=1e-9),
                "detail": f"Deterministic regime ({demand_regime}): SS = {safety_stock:.2f}.",
            }
        )
    else:
        verification_checks.append(
            {
                "name": "normal_cdf_consistency",
                "passed": math.isclose(norm_cdf(z), csl, rel_tol=1e-6, abs_tol=1e-6),
                "detail": f"Phi(z={z:.4f}) = {norm_cdf(z):.4f} matches CSL = {csl:.4f}.",
            }
        )
        if target_fill_rate is not None and implied_fill_rate is not None:
            verification_checks.append(
                {
                    "name": "fill_rate_solver_accuracy",
                    "passed": math.isclose(target_fill_rate, implied_fill_rate, rel_tol=1e-4, abs_tol=1e-4),
                    "detail": f"Implied fill rate ({implied_fill_rate:.4%}) matches target ({target_fill_rate:.4%}).",
                }
            )

    return {
        "model": "safety_stock_rop",
        "inputs_extracted": inputs_extracted,
        "inputs_validated": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_substituted": formula_substituted,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Flat convenience fields
        "demand_regime": demand_regime,
        "time_unit": time_unit,
        "avg_demand": d_bar,
        "demand_std": sigma_d,
        "lead_time": lt,
        "lead_time_std": sigma_l,
        "lead_time_variability_included": sigma_l > 0.0,
        "service_criterion": service_criterion,
        "z_score": z,
        "cycle_service_level": csl,
        "target_fill_rate": target_fill_rate,
        "order_quantity": q_val,
        "lead_time_demand_mean": lead_time_demand_mean,
        "lead_time_demand_variance": lead_time_demand_variance,
        "lead_time_demand_std": lead_time_demand_std,
        "safety_stock": safety_stock,
        "rounded_safety_stock": rounded_ss,
        "reorder_point": reorder_point,
        "rounded_reorder_point": rounded_rop,
        "expected_shortage_per_cycle": expected_shortage_per_cycle,
        "implied_fill_rate": implied_fill_rate,
    }


def calculate_safety_stock_rop(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Backwards-compatible alias for calculate_rop()."""
    return calculate_rop(*args, **kwargs)


# ---------------------------------------------------------------------------
# 4. Fixed-Period Review: calculate_periodic_review()
# ---------------------------------------------------------------------------

def calculate_periodic_review(
    avg_demand: float,
    demand_std: Optional[float] = None,
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    lead_time_std: Optional[float] = None,
    safety_stock: Optional[float] = None,
    on_hand: Optional[float] = None,
    on_order: Optional[float] = None,
    backorders: Optional[float] = None,
    inventory_position: Optional[float] = None,
    time_unit: str = "days",
) -> Dict[str, Any]:
    """Calculate Fixed-Period (P-system) protection period, target inventory, and order qty.

    Formulas:
        Protection period = review_period + lead_time
        Demand during protection period = avg_demand * protection_period
        sigma_{T+L} = sqrt(protection_period * demand_std^2 + avg_demand^2 * lead_time_std^2)
        Safety stock = z * sigma_{T+L} (or directly provided safety_stock)
        Target inventory (S) = demand_during_protection_period + safety_stock
        Order quantity = max(0, target_inventory - inventory_position)
    """
    d_bar = _require_positive("avg_demand", avg_demand)
    t_rev = _require_positive("review_period", review_period)
    lt = _require_non_negative("lead_time", lead_time)

    assumptions: List[str] = [
        f"Demand rate, review period T, and lead time L share consistent base unit '{time_unit}'.",
        "Inventory is reviewed strictly at fixed intervals T; replenishment protects across T + L.",
    ]

    protection_period = t_rev + lt
    demand_during_protection = d_bar * protection_period

    explicit_ss: Optional[float] = None
    if safety_stock is not None:
        explicit_ss = _require_non_negative("safety_stock", safety_stock)

    if explicit_ss is not None and (demand_std is None or demand_std == 0.0) and (
        lead_time_std is None or lead_time_std == 0.0
    ):
        sigma_d = 0.0 if demand_std is not None else 0.0
        sigma_l = 0.0 if lead_time_std is not None else 0.0
        protection_variance = 0.0
        protection_std = 0.0
        demand_regime = "explicit_safety_stock"
        z = _to_finite_float("z_score", z_score) if z_score is not None else None
        csl = _require_probability("service_level", service_level) if service_level is not None else None
        service_criterion = "explicit_safety_stock"
        ss_val = explicit_ss
        expected_shortage_per_cycle = None
        assumptions.append("Using explicitly supplied safety stock SS directly.")
    else:
        if demand_std is None and lead_time_std is None:
            raise ValueError(
                "Provide 'demand_std' and/or 'lead_time_std' (or explicit 'safety_stock', or 0.0 for constant demand/lead time)."
            )

        if demand_std is None:
            sigma_d = 0.0
            assumptions.append("Demand assumed constant (sigma_d = 0.0) as only lead_time_std was provided.")
        else:
            sigma_d = _require_non_negative("demand_std", demand_std)

        if lead_time_std is None:
            sigma_l = 0.0
            assumptions.append("Lead time assumed constant (sigma_L = 0.0).")
        else:
            sigma_l = _require_non_negative("lead_time_std", lead_time_std)

        protection_variance = protection_period * (sigma_d ** 2) + (d_bar ** 2) * (sigma_l ** 2)
        protection_std = math.sqrt(protection_variance)

        if protection_std == 0.0:
            demand_regime = "constant_demand_constant_lead_time"
            z = _to_finite_float("z_score", z_score) if z_score is not None else 0.0
            csl = 1.0
            service_criterion = "deterministic"
            ss_val = explicit_ss if explicit_ss is not None else 0.0
            expected_shortage_per_cycle = 0.0
            assumptions.append("Both demand and lead time are deterministic; safety stock SS = 0.")
        else:
            demand_regime = (
                "variable_demand_variable_lead_time" if (sigma_d > 0.0 and sigma_l > 0.0)
                else ("variable_demand_constant_lead_time" if sigma_d > 0.0 else "constant_demand_variable_lead_time")
            )
            if explicit_ss is not None and service_level is None and z_score is None:
                ss_val = explicit_ss
                z = ss_val / protection_std
                csl = norm_cdf(z)
                service_criterion = "explicit_safety_stock"
                expected_shortage_per_cycle = protection_std * standard_normal_loss(z)
            else:
                z, csl, service_criterion = _resolve_z_and_csl(
                    service_level=service_level,
                    z_score=z_score,
                )
                ss_val = z * protection_std
                if explicit_ss is not None and not math.isclose(ss_val, explicit_ss, rel_tol=1e-2, abs_tol=1e-2):
                    raise ValueError(
                        f"Conflicting inputs: computed safety stock ({ss_val:.2f}) disagrees with explicit safety_stock ({explicit_ss:.2f})."
                    )
                expected_shortage_per_cycle = protection_std * standard_normal_loss(z)

    target_inventory = demand_during_protection + ss_val
    expected_cycle_stock = 0.5 * d_bar * t_rev
    expected_average_inventory = expected_cycle_stock + ss_val
    avg_order_size = d_bar * t_rev
    implied_fill_rate = (
        max(0.0, 1.0 - (expected_shortage_per_cycle / avg_order_size))
        if (expected_shortage_per_cycle is not None and avg_order_size > 0.0)
        else None
    )

    resolved_ip: Optional[float] = None
    order_qty_now: Optional[float] = None
    ip_formula_note = ""
    oh_val: Optional[float] = None
    oo_val: Optional[float] = None
    bo_val: Optional[float] = None

    if inventory_position is not None:
        resolved_ip = _to_finite_float("inventory_position", inventory_position)
        ip_formula_note = f"IP = {_fmt(resolved_ip)} (directly provided)"
    elif on_hand is not None or on_order is not None or backorders is not None:
        if on_hand is None:
            raise ValueError("Provide 'on_hand' (or 'inventory_position') to compute inventory position.")
        oh_val = _require_non_negative("on_hand", on_hand)
        oo_val = _require_non_negative("on_order", on_order) if on_order is not None else 0.0
        bo_val = _require_non_negative("backorders", backorders) if backorders is not None else 0.0
        if on_order is None or backorders is None:
            assumptions.append(
                "Inventory position IP = on_hand + on_order - backorders with unspecified on_order/backorders treated as 0."
            )
        resolved_ip = oh_val + oo_val - bo_val
        ip_formula_note = f"IP = {_fmt(oh_val)} + {_fmt(oo_val)} - {_fmt(bo_val)} = {_fmt(resolved_ip)}"

    if resolved_ip is not None:
        order_qty_now = max(0.0, target_inventory - resolved_ip)
    else:
        assumptions.append(
            "Inventory position not provided; order_quantity is omitted (order_quantity = max(0, S - IP))."
        )

    rounded_ss = int(math.ceil(ss_val)) if ss_val >= 0.0 else int(round(ss_val))
    rounded_s = int(math.ceil(target_inventory))
    rounded_q = int(math.ceil(order_qty_now)) if order_qty_now is not None else None

    inputs_extracted = {
        "avg_demand": d_bar,
        "demand_std": sigma_d if service_criterion != "explicit_safety_stock" or demand_std is not None else None,
        "review_period": t_rev,
        "lead_time": lt,
        "lead_time_std": sigma_l if service_criterion != "explicit_safety_stock" or lead_time_std is not None else None,
        "safety_stock_input": explicit_ss,
        "service_level": service_level,
        "z_score": z_score,
        "on_hand": oh_val,
        "on_order": oo_val,
        "backorders": bo_val,
        "inventory_position": resolved_ip,
        "time_unit": time_unit,
    }

    variable_definitions = {
        "d": f"Average demand per {time_unit[:-1]}",
        "sigma_d": f"Demand standard deviation per {time_unit[:-1]}",
        "T": f"Fixed review period ({time_unit})",
        "L": f"Replenishment lead time ({time_unit})",
        "sigma_L": f"Lead-time standard deviation ({time_unit})",
        "P": f"Protection period (T + L, in {time_unit})",
        "mu_{T+L}": "Demand during protection period (d * (T + L))",
        "sigma_{T+L}": "Standard deviation of demand during protection period",
        "z": "Safety factor for target cycle service level",
        "SS": "Safety stock (units)",
        "S": "Target order-up-to inventory level (units)",
        "IP": "Inventory position (on_hand + on_order - backorders)",
        "Q": "Replenishment order quantity (max(0, S - IP))",
    }

    formula: Dict[str, str] = {
        "protection_period": "P = T + L",
        "demand_during_protection_period": "mu_{T+L} = d * (T + L)",
        "protection_period_std": "sigma_{T+L} = sqrt((T + L) * sigma_d^2 + d^2 * sigma_L^2)",
        "safety_stock": "SS = z * sigma_{T+L}" if service_criterion != "explicit_safety_stock" else "SS = given safety stock",
        "target_inventory": "S = mu_{T+L} + SS",
        "inventory_position": "IP = on_hand + on_order - backorders",
        "order_quantity": "Q = max(0, S - IP)",
    }

    formula_substituted: Dict[str, str] = {
        "protection_period": f"P = {_fmt(t_rev)} + {_fmt(lt)} = {_fmt(protection_period)} {time_unit}",
        "demand_during_protection_period": (
            f"mu_{{T+L}} = {_fmt(d_bar)} * {_fmt(protection_period)} = {_fmt(demand_during_protection)} units"
        ),
    }
    if service_criterion == "explicit_safety_stock" and protection_std == 0.0:
        formula_substituted["safety_stock"] = f"SS = {_fmt(ss_val)} units (provided directly)"
    elif service_criterion == "deterministic":
        formula_substituted["protection_period_std"] = "sigma_{T+L} = 0 (deterministic)"
        formula_substituted["safety_stock"] = "SS = 0 units"
    else:
        formula_substituted["protection_period_std"] = (
            f"sigma_{{T+L}} = sqrt({_fmt(protection_period)} * {_fmt(sigma_d)}^2 + "
            f"{_fmt(d_bar)}^2 * {_fmt(sigma_l)}^2) = {_fmt(protection_std)} units"
        )
        formula_substituted["safety_stock"] = (
            f"SS = {_fmt(z)} * {_fmt(protection_std)} = {_fmt(ss_val)} units"
            if z is not None
            else f"SS = {_fmt(ss_val)} units"
        )
    formula_substituted["target_inventory"] = (
        f"S = {_fmt(demand_during_protection)} + {_fmt(ss_val)} = {_fmt(target_inventory)} units"
    )
    if resolved_ip is not None and order_qty_now is not None:
        formula_substituted["inventory_position"] = ip_formula_note
        formula_substituted["order_quantity"] = (
            f"Q = max(0, {_fmt(target_inventory)} - {_fmt(resolved_ip)}) = {_fmt(order_qty_now)} units"
        )

    intermediate_calculations = {
        "demand_regime": demand_regime,
        "protection_period": protection_period,
        "demand_during_protection_period": demand_during_protection,
        "protection_period_variance": protection_variance,
        "protection_period_std": protection_std,
        "service_criterion": service_criterion,
        "z_score": z,
        "cycle_service_level": csl,
        "inventory_position": resolved_ip,
        "expected_cycle_stock": expected_cycle_stock,
        "expected_average_inventory": expected_average_inventory,
        "expected_shortage_per_cycle": expected_shortage_per_cycle,
        "implied_fill_rate": implied_fill_rate,
    }
    final_answer = {
        "protection_period": protection_period,
        "safety_stock": ss_val,
        "rounded_safety_stock": rounded_ss,
        "target_inventory": target_inventory,
        "rounded_target_inventory": rounded_s,
        "order_quantity": order_qty_now,
        "rounded_order_quantity": rounded_q,
    }

    verification_checks = [
        {
            "name": "protection_period_identity",
            "passed": math.isclose(protection_period, t_rev + lt, rel_tol=1e-9, abs_tol=1e-9),
            "detail": f"Protection period T + L = {protection_period:g} {time_unit}.",
        },
        {
            "name": "target_inventory_identity",
            "passed": math.isclose(target_inventory, demand_during_protection + ss_val, rel_tol=1e-8, abs_tol=1e-8),
            "detail": f"Target inventory S ({target_inventory:.2f}) == mu_{{T+L}} ({demand_during_protection:.2f}) + SS ({ss_val:.2f}).",
        },
    ]
    if service_criterion == "deterministic":
        verification_checks.append(
            {
                "name": "deterministic_safety_stock",
                "passed": math.isclose(ss_val, 0.0, abs_tol=1e-9),
                "detail": f"Deterministic regime ({demand_regime}): SS = {ss_val:.2f}.",
            }
        )
    elif z is not None and csl is not None:
        verification_checks.append(
            {
                "name": "normal_cdf_consistency",
                "passed": math.isclose(norm_cdf(z), csl, rel_tol=1e-6, abs_tol=1e-6),
                "detail": f"Phi(z={z:.4f}) = {norm_cdf(z):.4f} matches CSL = {csl:.4f}.",
            }
        )
    if order_qty_now is not None and resolved_ip is not None:
        q_expected = max(0.0, target_inventory - resolved_ip)
        verification_checks.append(
            {
                "name": "replenishment_order_identity",
                "passed": math.isclose(order_qty_now, q_expected, rel_tol=1e-8, abs_tol=1e-8),
                "detail": f"Order quantity Q ({order_qty_now:.2f}) == max(0, S - IP).",
            }
        )

    return {
        "model": "fixed_period_review",
        "inputs_extracted": inputs_extracted,
        "inputs_validated": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_substituted": formula_substituted,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Flat convenience fields
        "demand_regime": demand_regime,
        "time_unit": time_unit,
        "avg_demand": d_bar,
        "demand_std": sigma_d,
        "review_period": t_rev,
        "lead_time": lt,
        "lead_time_std": sigma_l,
        "protection_period": protection_period,
        "protection_interval": protection_period,
        "service_criterion": service_criterion,
        "z_score": z,
        "cycle_service_level": csl,
        "demand_during_protection_period": demand_during_protection,
        "protection_demand_mean": demand_during_protection,
        "protection_demand_std": protection_std,
        "safety_stock": ss_val,
        "rounded_safety_stock": rounded_ss,
        "target_inventory": target_inventory,
        "order_up_to_level": target_inventory,
        "rounded_order_up_to_level": rounded_s,
        "inventory_position": resolved_ip,
        "order_quantity": order_qty_now,
        "rounded_order_quantity": rounded_q,
        "expected_cycle_stock": expected_cycle_stock,
        "expected_average_inventory": expected_average_inventory,
        "expected_shortage_per_cycle": expected_shortage_per_cycle,
        "implied_fill_rate": implied_fill_rate,
    }


def calculate_fixed_period_review(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Backwards-compatible alias for calculate_periodic_review()."""
    return calculate_periodic_review(*args, **kwargs)


# ---------------------------------------------------------------------------
# 5. Newsvendor (Single-Period Stochastic Model)
# ---------------------------------------------------------------------------

def _resolve_newsvendor_costs(
    underage_cost: Optional[float],
    overage_cost: Optional[float],
    selling_price: Optional[float],
    unit_cost: Optional[float],
    salvage_value: Optional[float],
    goodwill_penalty: Optional[float],
    assumptions: List[str],
) -> Tuple[float, float, Optional[float], Optional[float], float, float]:
    g = 0.0 if goodwill_penalty is None else _require_non_negative("goodwill_penalty", goodwill_penalty)
    s_val = 0.0 if salvage_value is None else _to_finite_float("salvage_value", salvage_value)

    p_val: Optional[float] = _require_positive("selling_price", selling_price) if selling_price is not None else None
    c_val: Optional[float] = _require_positive("unit_cost", unit_cost) if unit_cost is not None else None

    if underage_cost is not None and overage_cost is not None:
        cu = _require_positive("underage_cost", underage_cost)
        co = _require_positive("overage_cost", overage_cost)
        assumptions.append("Using directly supplied underage cost (Cu) and overage cost (Co).")
    elif p_val is not None and c_val is not None:
        if salvage_value is None:
            assumptions.append("No salvage_value provided; assumed salvage_value = 0.0.")
        if goodwill_penalty is None:
            assumptions.append("No goodwill_penalty provided; assumed goodwill_penalty = 0.0.")
        cu = p_val - c_val + g
        co = c_val - s_val
        if cu <= 0.0:
            raise ValueError(
                f"Underage cost (selling_price - unit_cost + goodwill_penalty = {cu:.4f}) must be positive."
            )
        if co <= 0.0:
            raise ValueError(
                f"Overage cost (unit_cost - salvage_value = {co:.4f}) must be positive."
            )
        assumptions.append(
            f"Derived Cu = p - c + g = {_fmt(cu)} and Co = c - s = {_fmt(co)}."
        )
    else:
        raise ValueError(
            "Provide either ('underage_cost' and 'overage_cost') or ('selling_price' and 'unit_cost')."
        )

    return cu, co, p_val, c_val, s_val, g


def calculate_newsvendor(
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
    selling_price: Optional[float] = None,
    unit_cost: Optional[float] = None,
    salvage_value: Optional[float] = None,
    goodwill_penalty: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    demand_min: Optional[float] = None,
    demand_max: Optional[float] = None,
    demand_distribution: Optional[Sequence[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """Calculate underage cost, overage cost, critical ratio, and recommended order quantity."""
    assumptions: List[str] = [
        "Single-period perishable setting: replenishment occurs once before demand is realized.",
    ]

    cu, co, p_val, c_val, s_val, g_val = _resolve_newsvendor_costs(
        underage_cost=underage_cost,
        overage_cost=overage_cost,
        selling_price=selling_price,
        unit_cost=unit_cost,
        salvage_value=salvage_value,
        goodwill_penalty=goodwill_penalty,
        assumptions=assumptions,
    )

    critical_ratio = cu / (cu + co)
    prev_cum_prob = 0.0

    demand_dist_seq: Optional[Sequence[Tuple[float, float]]] = None
    explicit_dist_name: Optional[str] = None
    if isinstance(demand_distribution, str):
        explicit_dist_name = demand_distribution.strip().lower()
        if explicit_dist_name not in ("normal", "uniform", "discrete"):
            raise ValueError(
                f"Unsupported demand_distribution string '{demand_distribution}'. Use 'normal', 'uniform', or discrete pairs."
            )
    elif demand_distribution is not None:
        demand_dist_seq = demand_distribution

    if demand_dist_seq is not None or explicit_dist_name == "discrete":
        if not demand_dist_seq:
            raise ValueError("demand_distribution cannot be empty.")
        pairs = sorted(
            ((_require_non_negative("demand_value", d), _require_non_negative("probability", prob))
             for d, prob in demand_dist_seq),
            key=lambda x: x[0],
        )
        total_prob = sum(prob for _, prob in pairs)
        if not math.isclose(total_prob, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(
                f"Discrete demand probabilities must sum to 1.0 (got {total_prob})."
            )

        cum_prob = 0.0
        optimal_q = pairs[-1][0]
        csl_at_opt = 1.0
        for d_val, prob in pairs:
            if cum_prob + prob + 1e-12 >= critical_ratio:
                prev_cum_prob = cum_prob
                cum_prob += prob
                optimal_q = d_val
                csl_at_opt = cum_prob
                break
            cum_prob += prob

        mean_d = sum(d_val * prob for d_val, prob in pairs)
        exp_sales = sum(min(d_val, optimal_q) * prob for d_val, prob in pairs)
        exp_leftover = sum(max(optimal_q - d_val, 0.0) * prob for d_val, prob in pairs)
        exp_lost_sales = sum(max(d_val - optimal_q, 0.0) * prob for d_val, prob in pairs)
        exp_mismatch = cu * exp_lost_sales + co * exp_leftover
        z_star: Optional[float] = None
        dist_type = "discrete"
        assumptions.append("Discrete demand distribution: Q* is the smallest demand level with F(Q) >= critical ratio.")

    elif demand_min is not None and demand_max is not None:
        a = _to_finite_float("demand_min", demand_min)
        b = _to_finite_float("demand_max", demand_max)
        if b <= a:
            raise ValueError(f"Uniform demand requires demand_min < demand_max, got [{a}, {b}].")
        optimal_q = a + critical_ratio * (b - a)
        mean_d = 0.5 * (a + b)
        exp_leftover = ((optimal_q - a) ** 2) / (2.0 * (b - a))
        exp_lost_sales = ((b - optimal_q) ** 2) / (2.0 * (b - a))
        exp_sales = mean_d - exp_lost_sales
        exp_mismatch = cu * exp_lost_sales + co * exp_leftover
        csl_at_opt = critical_ratio
        z_star = None
        dist_type = "uniform"
        assumptions.append(f"Demand follows Uniform({a:g}, {b:g}).")

    elif demand_mean is not None and demand_std is not None:
        mu = _require_non_negative("demand_mean", demand_mean)
        sigma = _require_positive("demand_std", demand_std)
        z_star = norm_inv_cdf(critical_ratio)
        raw_q = mu + z_star * sigma
        optimal_q = max(0.0, raw_q)
        effective_z = (optimal_q - mu) / sigma
        loss_z = standard_normal_loss(effective_z)

        mean_d = mu
        exp_lost_sales = sigma * loss_z
        exp_sales = mu - exp_lost_sales
        exp_leftover = optimal_q - exp_sales
        exp_mismatch = cu * exp_lost_sales + co * exp_leftover
        csl_at_opt = norm_cdf(effective_z)
        dist_type = "normal"
        assumptions.append(f"Demand follows Normal(mean={mu:g}, std={sigma:g}).")

    else:
        raise ValueError(
            "Provide Normal ('demand_mean', 'demand_std'), Uniform ('demand_min', 'demand_max'), "
            "or Discrete ('demand_distribution') parameters."
        )

    fill_rate = (exp_sales / mean_d) if mean_d > 0.0 else None

    expected_profit: Optional[float] = None
    if p_val is not None and c_val is not None:
        expected_profit = (
            p_val * exp_sales
            + s_val * exp_leftover
            - c_val * optimal_q
            - g_val * exp_lost_sales
        )

    rounded_q = int(round(optimal_q))

    inputs_extracted = {
        "underage_cost": cu,
        "overage_cost": co,
        "selling_price": p_val,
        "unit_cost": c_val,
        "salvage_value": s_val,
        "goodwill_penalty": g_val,
        "distribution": dist_type,
        "demand_mean": float(demand_mean) if demand_mean is not None else None,
        "demand_std": float(demand_std) if demand_std is not None else None,
        "demand_min": float(demand_min) if demand_min is not None else None,
        "demand_max": float(demand_max) if demand_max is not None else None,
    }

    variable_definitions = {
        "Cu": "Underage (shortage) cost per unit of unmet demand",
        "Co": "Overage (excess) cost per unit left unsold",
        "CR": "Critical ratio = Cu / (Cu + Co)",
        "Q*": "Recommended single-period order quantity",
    }
    if p_val is not None and c_val is not None:
        variable_definitions.update(
            {
                "p": "Selling price per unit",
                "c": "Purchase cost per unit",
                "s": "Salvage value per leftover unit",
                "g": "Goodwill penalty per unit of lost demand",
            }
        )
    if dist_type == "normal":
        variable_definitions.update(
            {"mu": "Mean demand", "sigma": "Demand standard deviation", "z*": "Standard normal quantile Phi^{-1}(CR)"}
        )
    elif dist_type == "uniform":
        variable_definitions.update({"a": "Uniform minimum demand", "b": "Uniform maximum demand"})

    formula: Dict[str, str] = {
        "underage_cost": "Cu = p - c + g" if (p_val is not None and c_val is not None) else "Cu (provided)",
        "overage_cost": "Co = c - s" if (p_val is not None and c_val is not None) else "Co (provided)",
        "critical_ratio": "CR = Cu / (Cu + Co)",
    }
    if dist_type == "normal":
        formula["z_score"] = "z* = Phi^{-1}(CR)"
        formula["recommended_order_quantity"] = "Q* = mu + z* * sigma"
    elif dist_type == "uniform":
        formula["recommended_order_quantity"] = "Q* = a + CR * (b - a)"
    else:
        formula["recommended_order_quantity"] = "Q* = smallest d with F(d) >= CR"

    formula_substituted: Dict[str, str] = {}
    if p_val is not None and c_val is not None:
        formula_substituted["underage_cost"] = f"Cu = {_fmt(p_val)} - {_fmt(c_val)} + {_fmt(g_val)} = ${_fmt(cu, 2)}"
        formula_substituted["overage_cost"] = f"Co = {_fmt(c_val)} - {_fmt(s_val)} = ${_fmt(co, 2)}"
    else:
        formula_substituted["underage_cost"] = f"Cu = ${_fmt(cu, 2)}"
        formula_substituted["overage_cost"] = f"Co = ${_fmt(co, 2)}"

    formula_substituted["critical_ratio"] = (
        f"CR = {_fmt(cu)} / ({_fmt(cu)} + {_fmt(co)}) = {_fmt(critical_ratio, 4)}"
    )
    if dist_type == "normal" and z_star is not None:
        formula_substituted["z_score"] = f"z* = Phi^{{-1}}({_fmt(critical_ratio, 4)}) = {_fmt(z_star, 4)}"
        formula_substituted["recommended_order_quantity"] = (
            f"Q* = {_fmt(mean_d)} + ({_fmt(z_star, 4)}) * {_fmt(float(demand_std))} = {_fmt(optimal_q)} units"
        )
    elif dist_type == "uniform":
        a_v = float(demand_min)
        b_v = float(demand_max)
        formula_substituted["recommended_order_quantity"] = (
            f"Q* = {_fmt(a_v)} + {_fmt(critical_ratio, 4)} * ({_fmt(b_v)} - {_fmt(a_v)}) = {_fmt(optimal_q)} units"
        )
    else:
        formula_substituted["recommended_order_quantity"] = (
            f"Q* = {_fmt(optimal_q)} units (F(Q*) = {_fmt(csl_at_opt, 4)} >= {_fmt(critical_ratio, 4)})"
        )

    intermediate_calculations = {
        "distribution": dist_type,
        "underage_cost": cu,
        "overage_cost": co,
        "critical_ratio": critical_ratio,
        "z_score": z_star,
        "demand_mean": mean_d,
        "expected_sales": exp_sales,
        "expected_leftover": exp_leftover,
        "expected_lost_sales": exp_lost_sales,
        "expected_mismatch_cost": exp_mismatch,
        "expected_profit": expected_profit,
        "cycle_service_level": csl_at_opt,
        "fill_rate": fill_rate,
    }
    final_answer = {
        "underage_cost": cu,
        "overage_cost": co,
        "critical_ratio": critical_ratio,
        "recommended_order_quantity": optimal_q,
        "rounded_order_quantity": rounded_q,
    }

    verification_checks = [
        {
            "name": "critical_ratio_bounds",
            "passed": math.isclose(critical_ratio, cu / (cu + co), rel_tol=1e-9) and (0.0 < critical_ratio < 1.0),
            "detail": f"Critical ratio CR = Cu/(Cu+Co) = {cu:g}/({cu:g}+{co:g}) = {critical_ratio:.4f}.",
        },
        {
            "name": "inventory_conservation",
            "passed": math.isclose(exp_sales + exp_leftover, optimal_q, rel_tol=1e-6, abs_tol=1e-6),
            "detail": f"E[sales] ({exp_sales:.2f}) + E[leftover] ({exp_leftover:.2f}) == Q* ({optimal_q:.2f}).",
        },
    ]
    if dist_type == "normal" and z_star is not None and optimal_q > 0.0:
        closed_form_mismatch = (cu + co) * float(demand_std) * norm_pdf(z_star)
        verification_checks.append(
            {
                "name": "normal_optimality_identity",
                "passed": math.isclose(exp_mismatch, closed_form_mismatch, rel_tol=1e-5, abs_tol=1e-5),
                "detail": (
                    f"Expected mismatch cost (${exp_mismatch:,.2f}) matches "
                    f"(Cu+Co)*sigma*phi(z*) (${closed_form_mismatch:,.2f})."
                ),
            }
        )
    elif dist_type == "uniform":
        a_v = float(demand_min)
        b_v = float(demand_max)
        cdf_q = (optimal_q - a_v) / (b_v - a_v)
        verification_checks.append(
            {
                "name": "uniform_optimality_identity",
                "passed": math.isclose(cdf_q, critical_ratio, rel_tol=1e-8, abs_tol=1e-8),
                "detail": f"Uniform CDF F(Q*) = {cdf_q:.4f} == CR = {critical_ratio:.4f}.",
            }
        )
    elif dist_type == "discrete":
        verification_checks.append(
            {
                "name": "discrete_fractile_optimality",
                "passed": csl_at_opt + 1e-12 >= critical_ratio and prev_cum_prob < critical_ratio + 1e-12,
                "detail": f"F(prev)={prev_cum_prob:.4f} < CR={critical_ratio:.4f} <= F(Q*)={csl_at_opt:.4f}.",
            }
        )

    return {
        "model": "newsvendor",
        "inputs_extracted": inputs_extracted,
        "inputs_validated": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_substituted": formula_substituted,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Flat convenience fields
        "distribution": dist_type,
        "underage_cost": cu,
        "overage_cost": co,
        "selling_price": p_val,
        "unit_cost": c_val,
        "salvage_value": s_val,
        "goodwill_penalty": g_val,
        "critical_ratio": critical_ratio,
        "z_score": z_star,
        "demand_mean": mean_d,
        "demand_std": float(demand_std) if demand_std is not None else None,
        "demand_min": float(demand_min) if demand_min is not None else None,
        "demand_max": float(demand_max) if demand_max is not None else None,
        "recommended_order_quantity": optimal_q,
        "optimal_order_quantity": optimal_q,
        "rounded_order_quantity": rounded_q,
        "expected_sales": exp_sales,
        "expected_leftover": exp_leftover,
        "expected_lost_sales": exp_lost_sales,
        "expected_mismatch_cost": exp_mismatch,
        "expected_profit": expected_profit,
        "cycle_service_level": csl_at_opt,
        "fill_rate": fill_rate,
    }


# ---------------------------------------------------------------------------
# Tool Registry for Agent Selection
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "eoq": {
        "tool_name": "calculate_eoq",
        "display_name": "Economic Order Quantity (EOQ)",
        "function": calculate_eoq,
        "required_params": ["demand", "ordering_cost"],
        "required_any_of": [["holding_cost", ("unit_cost", "holding_rate")]],
        "optional_params": ["unit_cost", "holding_rate", "lead_time_days", "operating_days_per_year"],
        "description": "Computes optimal fixed lot size Q = sqrt(2DS/H), order frequency, cycle length, and total inventory cost.",
    },
    "quantity_discount": {
        "tool_name": "calculate_quantity_discount",
        "display_name": "Quantity Discount (Price-Break EOQ)",
        "function": calculate_quantity_discount,
        "required_params": ["demand", "ordering_cost", "tiers"],
        "required_any_of": [["holding_cost", "holding_rate"]],
        "optional_params": ["discount_type"],
        "description": "Evaluates feasible quantities across supplier price tiers using fixed or percentage holding cost.",
    },
    "safety_stock_rop": {
        "tool_name": "calculate_rop",
        "display_name": "Safety Stock & Reorder Point (Continuous Review)",
        "function": calculate_rop,
        "required_params": ["avg_demand", "lead_time"],
        "required_any_of": [["demand_std", "lead_time_std"], ["service_level", "z_score", "fill_rate"]],
        "optional_params": ["order_quantity", "time_unit"],
        "description": "Computes safety stock and ROP across constant/variable demand and constant/variable lead time regimes.",
    },
    "fixed_period_review": {
        "tool_name": "calculate_periodic_review",
        "display_name": "Fixed-Period Review (Order-Up-To P-System)",
        "function": calculate_periodic_review,
        "required_params": ["avg_demand", "review_period", "lead_time"],
        "required_any_of": [["demand_std", "lead_time_std", "safety_stock"], ["service_level", "z_score", "safety_stock"]],
        "optional_params": [
            "safety_stock",
            "on_hand",
            "on_order",
            "backorders",
            "inventory_position",
            "time_unit",
        ],
        "description": "Computes protection period (T+L), target inventory level, safety stock, and order quantity max(0, target - IP).",
    },
    "newsvendor": {
        "tool_name": "calculate_newsvendor",
        "display_name": "Newsvendor (Single-Period Model)",
        "function": calculate_newsvendor,
        "required_params": [],
        "required_any_of": [
            [("underage_cost", "overage_cost"), ("selling_price", "unit_cost")],
            [("demand_mean", "demand_std"), ("demand_min", "demand_max"), "demand_distribution"],
        ],
        "optional_params": ["salvage_value", "goodwill_penalty"],
        "description": "Computes underage cost, overage cost, critical ratio, and recommended single-period order quantity.",
    },
}

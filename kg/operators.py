"""
Operator registry and DAG interpreter for the Saudi extreme event knowledge graph.

This module defines the vocabulary of all legal DAG operators (OPS), provides
a recursive DAG evaluator that walks a JSON-AST tree and computes results,
validates operator definitions on load, and exposes functions to load and
register operators from an operators.json file.
"""

from __future__ import annotations

import json
import math
import operator as _builtin_op
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DAGNode = Dict[str, Any]          # A node in the JSON-AST DAG
Context = Dict[str, Any]           # Variable name → value context
OpFunc = Callable[..., Any]        # Callable backing a DAG operator

# ---------------------------------------------------------------------------
# Helper: condition evaluator for the "threshold" op
# ---------------------------------------------------------------------------

_OP_MAP: Dict[str, Callable[[Any, Any], bool]] = {
    ">":  _builtin_op.gt,
    "<":  _builtin_op.lt,
    ">=": _builtin_op.ge,
    "<=": _builtin_op.le,
    "==": _builtin_op.eq,
    "!=": _builtin_op.ne,
}


def _eval_condition(val: Any, cond_str: str, ref: Any) -> bool:
    """Evaluate a comparison expression ``val <cond_str> ref``.

    Parameters
    ----------
    val : Any
        The dynamic value to test (e.g. a computed indicator).
    cond_str : str
        One of ``">", "<", ">=", "<=", "==", "!="``.
    ref : Any
        The reference value to compare against.

    Returns
    -------
    bool
    """
    cmp = _OP_MAP.get(cond_str)
    if cmp is None:
        raise ValueError(
            f"Unknown condition operator {cond_str!r}. "
            f"Expected one of: {list(_OP_MAP)}"
        )
    return cmp(val, ref)


# ---------------------------------------------------------------------------
# OPS vocabulary — all legal DAG operators
# ---------------------------------------------------------------------------

OPS: Dict[str, OpFunc] = {
    # Arithmetic -----------------------------------------------------------
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b if abs(b) > 1e-12 else None,
    "neg": lambda a: -a,

    # Power & root ---------------------------------------------------------
    "sqrt": lambda a: math.sqrt(max(a, 0.0)),
    "sqr":  lambda a: a * a,
    "pow":  lambda a, b: a ** b,

    # Aggregation ----------------------------------------------------------
    "max":  max,
    "min":  min,
    "abs":  abs,

    # Logic / conditionals -------------------------------------------------
    "threshold": lambda val, cond_str, ref: _eval_condition(val, cond_str, ref),
    "where":     lambda cond, t, f: t if cond else f,
    "and_":      lambda a, b: a and b,
    "or_":       lambda a, b: a or b,

    # Variable reference ---------------------------------------------------
    "var":  lambda name, ctx: ctx.get(name),
}

# ---------------------------------------------------------------------------
# Special formula implementations
# ---------------------------------------------------------------------------


def _vpd_formula(t_var: float, rh_var: float) -> float:
    """Compute Vapour Pressure Deficit (kPa).

    VPD = 0.6108 * exp(17.27 * T / (T + 237.3)) * (1 - RH/100)

    Parameters
    ----------
    t_var : float
        Air temperature in degrees Celsius.
    rh_var : float
        Relative humidity as a percentage (0–100).

    Returns
    -------
    float
        VPD in kPa.
    """
    es = 0.6108 * math.exp(17.27 * t_var / (t_var + 237.3))
    return es * (1.0 - rh_var / 100.0)


def _heat_index_formula(t_degc: float, rh_pct: float) -> float:
    """Simplified Rothfusz heat-index approximation.

    Converts temperature from Celsius to Fahrenheit, applies the
    regression, and returns the result in degrees Celsius with a
    low-end guard (temperatures below ~27 C / 80 F return the
    original temperature unchanged).

    HI_F = 0.5 * (T_F + 61.0 + (T_F - 68.0) * 1.2 + RH * 0.094)

    Parameters
    ----------
    t_degc : float
        Air temperature in degrees Celsius.
    rh_pct : float
        Relative humidity as a percentage (0–100).

    Returns
    -------
    float
        Heat index in degrees Celsius.
    """
    t_f = t_degc * 9.0 / 5.0 + 32.0
    hi_f = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh_pct * 0.094)

    if hi_f < 80.0:
        return t_degc

    # Convert back to Celsius
    return (hi_f - 32.0) * 5.0 / 9.0


def _sum_flags(flags: List[bool]) -> int:
    """Count how many boolean flags are ``True``.

    Parameters
    ----------
    flags : list of bool
        Flag values extracted from the context (or already evaluated).

    Returns
    -------
    int
        Number of ``True`` flags.
    """
    return sum(1 for f in flags if f)


# ---------------------------------------------------------------------------
# DAG evaluator
# ---------------------------------------------------------------------------


def evaluate_dag(dag: Any, context: Optional[Context] = None) -> Any:
    """Recursively evaluate a JSON DAG tree against a variable context.

    Leaf semantics
    --------------
    - Literal (int, float, bool, str, None) → returned as-is.
    - ``{"op": "var", "name": "<var_name>"}`` → looked up in *context*.

    Compound semantics
    ------------------
    ``{"op": "<op_name>", "left": {...}, "right": {...}}``
    or (for unary ops) ``{"op": "<op_name>", "value": {...}}``.

    Special ops (handled inline rather than through `OPS`)
    ------------------------------------------------------
    - **vpd_formula** : expects ``t_var`` and ``rh_var`` keys in *context*.
    - **heat_index_formula** : expects ``t_var`` (degC) and ``rh_var`` (%) keys.
    - **sum_flags** : expects a ``flags`` list of variable names — each name is
      looked up in *context* and its truth value counted.

    Parameters
    ----------
    dag : dict | int | float | str | bool | None
        The root of the DAG tree (or a literal leaf).
    context : dict, optional
        Variable name → value mapping used by ``"var"`` nodes and special ops.

    Returns
    -------
    Any
        The computed result (float, bool, int, etc.).

    Raises
    ------
    ValueError
        If an unknown operator is encountered, or required keys are missing,
        or an unsupported node structure is supplied.
    """
    if context is None:
        context = {}

    # --- literal leaf -----------------------------------------------------
    if isinstance(dag, (int, float, str, bool, type(None))):
        return dag

    # --- dict node --------------------------------------------------------
    if not isinstance(dag, dict):
        raise TypeError(
            f"DAG node must be a dict or literal, got {type(dag).__name__}"
        )

    op = dag.get("op")
    if op is None:
        raise ValueError("DAG dict node is missing required key 'op'")

    # -- special ops -------------------------------------------------------
    if op == "vpd_formula":
        t_var = context.get("t_var")
        rh_var = context.get("rh_var")
        if t_var is None or rh_var is None:
            raise ValueError(
                "vpd_formula requires 't_var' and 'rh_var' in context; "
                f"got t_var={t_var!r}, rh_var={rh_var!r}"
            )
        return _vpd_formula(float(t_var), float(rh_var))

    if op == "heat_index_formula":
        t_var = context.get("t_var")
        rh_var = context.get("rh_var")
        if t_var is None or rh_var is None:
            raise ValueError(
                "heat_index_formula requires 't_var' and 'rh_var' in context; "
                f"got t_var={t_var!r}, rh_var={rh_var!r}"
            )
        return _heat_index_formula(float(t_var), float(rh_var))

    if op == "sum_flags":
        flag_names: List[str] = dag.get("flags", [])
        flag_values = [bool(context.get(name)) for name in flag_names]
        return _sum_flags(flag_values)

    # -- variable reference -------------------------------------------------
    if op == "var":
        var_name = dag.get("name")
        if var_name is None:
            raise ValueError("'var' node must contain a 'name' key")
        return context.get(var_name)

    # -- regular op from OPS -----------------------------------------------
    func = OPS.get(op)
    if func is None:
        raise ValueError(
            f"Unknown DAG operator {op!r}. Known operators: {list(OPS)}"
        )

    # Determine operator arity based on presence of known keys.
    # Ternary: "where" has `cond`, `t`, `f`; "threshold" has `val`, `cond_str`, `ref`.
    if op == "where":
        cond_val = evaluate_dag(dag["cond"], context)
        t_val = evaluate_dag(dag["t"], context)
        f_val = evaluate_dag(dag["f"], context)
        return func(cond_val, t_val, f_val)

    if op == "threshold":
        val_val = evaluate_dag(dag["val"], context)
        # cond_str and ref are typically constants, not sub-DAGs
        cond_str = dag["cond_str"]
        ref_val = dag["ref"] if isinstance(dag["ref"], (int, float, str)) else evaluate_dag(dag["ref"], context)
        return func(val_val, cond_str, ref_val)

    # Generic arity detection (works for runtime-registered ops as well) -----

    # Unary: operator that takes a single "value"
    if "value" in dag:
        v = evaluate_dag(dag["value"], context)
        return func(v)

    # Binary: operator with "left" and "right"
    if "left" in dag and "right" in dag:
        left_val = evaluate_dag(dag["left"], context)
        right_val = evaluate_dag(dag["right"], context)
        return func(left_val, right_val)

    # Variadic: operator with a list of "values"
    if "values" in dag:
        evaluated = [evaluate_dag(v, context) for v in dag["values"]]
        return func(*evaluated)

    raise ValueError(
        f"Cannot determine arity or structure for operator {op!r} "
        f"from node keys: {list(dag)}"
    )


# ---------------------------------------------------------------------------
# Operator loading & validation
# ---------------------------------------------------------------------------


def load_operators(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Load operator definitions from a JSON file.

    Each entry in the JSON array is expected to have at least an ``"id"`` key.
    The result is a dict keyed by operator id for fast lookup.

    Parameters
    ----------
    json_path : str or Path
        Path to an ``operators.json`` file.

    Returns
    -------
    dict
        ``{operator_id: operator_definition, ...}``

    Raises
    ------
    FileNotFoundError
        If *json_path* does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        # The file itself is already keyed by id.
        return data

    if isinstance(data, list):
        result: Dict[str, Any] = {}
        for entry in data:
            op_id = entry.get("id")
            if op_id is None:
                raise ValueError(
                    f"Operator entry in {path.name} is missing 'id': {entry}"
                )
            if op_id in result:
                raise ValueError(
                    f"Duplicate operator id {op_id!r} in {path.name}"
                )
            result[op_id] = entry
        return result

    raise ValueError(
        f"Expected a JSON object or array at {path.name}, got {type(data).__name__}"
    )


def validate_operators(operators_dict: Dict[str, Any]) -> List[str]:
    """Validate that every operator's DAG references only known OPS entries.

    For each operator definition the function walks its ``dag`` subtree
    (recursively) and collects any operator name **not** present in the
    global ``OPS`` vocabulary (including the three special forms
    ``vpd_formula``, ``heat_index_formula``, and ``sum_flags``).

    Parameters
    ----------
    operators_dict : dict
        Operator definitions keyed by id, as returned by :func:`load_operators`.

    Returns
    -------
    list of str
        Human-readable error messages.  An empty list means all operators
        are valid.
    """
    errors: List[str] = []

    # Extend the known set with the three special-form ops that are not in OPS
    # but are handled by evaluate_dag.
    KNOWN_OPS = set(OPS) | {"vpd_formula", "heat_index_formula", "sum_flags", "var"}

    def _collect_ops(node: Any, path: str) -> None:
        """Recursively walk *node* and check every 'op' key."""
        if isinstance(node, dict):
            op = node.get("op")
            if op is not None and op not in KNOWN_OPS:
                errors.append(
                    f"Unknown operator {op!r} at {path}. Known: {sorted(KNOWN_OPS)}"
                )
            # Recurse into all children
            for key, child in node.items():
                _collect_ops(child, f"{path}.{key}")
        elif isinstance(node, list):
            for idx, child in enumerate(node):
                _collect_ops(child, f"{path}[{idx}]")

    for op_id, op_def in operators_dict.items():
        if not isinstance(op_def, dict):
            errors.append(f"Operator {op_id!r}: definition is not a dict")
            continue
        dag = op_def.get("dag")
        if dag is not None:
            _collect_ops(dag, f"operators.{op_id}.dag")
        # Also validate triggers / conditions if they contain embedded DAGs
        for sub_key in ("trigger", "condition", "formula"):
            sub_node = op_def.get(sub_key)
            if sub_node is not None:
                _collect_ops(sub_node, f"operators.{op_id}.{sub_key}")

    return errors


# ---------------------------------------------------------------------------
# Convenience: load + validate in one shot
# ---------------------------------------------------------------------------


def load_and_validate(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Load operators from a JSON file and validate them immediately.

    Parameters
    ----------
    json_path : str or Path
        Path to an ``operators.json`` file.

    Returns
    -------
    dict
        Operator definitions keyed by id.

    Raises
    ------
    ValueError
        If validation finds unknown operators.
    """
    ops = load_operators(json_path)
    errs = validate_operators(ops)
    if errs:
        raise ValueError(
            f"Operator validation failed with {len(errs)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errs)
        )
    return ops


# ---------------------------------------------------------------------------
# Operator registration (for runtime extension)
# ---------------------------------------------------------------------------


def register_operator(name: str, func: OpFunc) -> None:
    """Register (or overwrite) a single operator in the global OPS vocabulary.

    Parameters
    ----------
    name : str
        The operator name to use in DAG ``"op"`` keys.
    func : callable
        The implementation.
    """
    OPS[name] = func


def register_operators(ops_mapping: Dict[str, OpFunc]) -> None:
    """Bulk-register operators from a name→callable mapping.

    Parameters
    ----------
    ops_mapping : dict
        ``{name: callable, ...}`` pairs to insert into the global OPS.
    """
    OPS.update(ops_mapping)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def list_operators() -> List[str]:
    """Return a sorted list of all registered operator names (including OPS)."""
    return sorted(OPS)


def get_operator(name: str) -> Optional[OpFunc]:
    """Return the callable for *name*, or ``None`` if not registered."""
    return OPS.get(name)

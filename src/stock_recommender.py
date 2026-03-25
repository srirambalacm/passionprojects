from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, List


@dataclass(frozen=True)
class HistoricalPrediction:
    ticker: str
    predicted_return: float
    actual_return: float


@dataclass(frozen=True)
class CurrentPrediction:
    ticker: str
    predicted_return: float


@dataclass(frozen=True)
class StockMetrics:
    ticker: str
    sample_size: int
    hit_rate: float
    calibration_score: float
    confidence: float
    expected_return: float
    kelly_fraction: float


@dataclass(frozen=True)
class Recommendation:
    ticker: str
    decision: str
    allocation_pct: float
    allocation_amount: float
    predicted_return: float
    expected_return: float
    confidence: float
    rationale: str


def _parse_float(value: str, column: str, row_number: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid float for '{column}' in row {row_number}: {value!r}"
        ) from exc


def load_historical_predictions(csv_path: Path) -> List[HistoricalPrediction]:
    rows: List[HistoricalPrediction] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"ticker", "predicted_return", "actual_return"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"Historical CSV missing required columns: {sorted(missing)}"
            )
        for index, row in enumerate(reader, start=2):
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                raise ValueError(f"Missing ticker in row {index}")
            predicted = _parse_float(row.get("predicted_return"), "predicted_return", index)
            actual = _parse_float(row.get("actual_return"), "actual_return", index)
            rows.append(HistoricalPrediction(ticker=ticker, predicted_return=predicted, actual_return=actual))
    if not rows:
        raise ValueError("Historical CSV contains no data rows")
    return rows


def load_current_predictions(csv_path: Path) -> List[CurrentPrediction]:
    rows: List[CurrentPrediction] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"ticker", "predicted_return"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Current CSV missing required columns: {sorted(missing)}")
        for index, row in enumerate(reader, start=2):
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                raise ValueError(f"Missing ticker in row {index}")
            predicted = _parse_float(row.get("predicted_return"), "predicted_return", index)
            rows.append(CurrentPrediction(ticker=ticker, predicted_return=predicted))
    if not rows:
        raise ValueError("Current CSV contains no data rows")
    return rows


def _safe_mean(values: Iterable[float], fallback: float = 0.0) -> float:
    values_list = list(values)
    return mean(values_list) if values_list else fallback


def _estimate_kelly(hit_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss
    if b <= 0:
        return 0.0
    raw = hit_rate - ((1 - hit_rate) / b)
    return max(0.0, raw)


def compute_metrics(
    historical: List[HistoricalPrediction],
    current: CurrentPrediction,
    min_history: int,
) -> StockMetrics:
    bucket = [item for item in historical if item.ticker == current.ticker]
    sample_size = len(bucket)
    if sample_size == 0:
        return StockMetrics(
            ticker=current.ticker,
            sample_size=0,
            hit_rate=0.5,
            calibration_score=0.0,
            confidence=0.0,
            expected_return=0.0,
            kelly_fraction=0.0,
        )

    correct_direction = [
        math.copysign(1, row.predicted_return) == math.copysign(1, row.actual_return)
        for row in bucket
    ]
    hit_rate = sum(correct_direction) / sample_size
    avg_abs_error = _safe_mean(
        abs(row.predicted_return - row.actual_return) for row in bucket
    )
    calibration_score = 1 / (1 + avg_abs_error)

    confidence = min(1.0, sample_size / max(1, min_history))
    directional_edge = max(0.0, (2 * hit_rate) - 1)
    expected_return = current.predicted_return * directional_edge * calibration_score

    wins = [row.actual_return for row in bucket if row.actual_return > 0]
    losses = [abs(row.actual_return) for row in bucket if row.actual_return < 0]
    avg_win = _safe_mean(wins, fallback=0.0)
    avg_loss = _safe_mean(losses, fallback=0.0)
    kelly = _estimate_kelly(hit_rate=hit_rate, avg_win=avg_win, avg_loss=avg_loss)

    return StockMetrics(
        ticker=current.ticker,
        sample_size=sample_size,
        hit_rate=hit_rate,
        calibration_score=calibration_score,
        confidence=confidence,
        expected_return=expected_return,
        kelly_fraction=kelly,
    )


def recommend_for_stock(
    current: CurrentPrediction,
    metrics: StockMetrics,
    capital: float,
    min_confidence: float,
    min_expected_return: float,
    max_position_pct: float,
    fractional_kelly: float,
) -> Recommendation:
    if current.predicted_return <= 0:
        return Recommendation(
            ticker=current.ticker,
            decision="DO_NOT_INVEST",
            allocation_pct=0.0,
            allocation_amount=0.0,
            predicted_return=current.predicted_return,
            expected_return=metrics.expected_return,
            confidence=metrics.confidence,
            rationale="Model predicts non-positive return.",
        )

    if metrics.sample_size == 0:
        return Recommendation(
            ticker=current.ticker,
            decision="DO_NOT_INVEST",
            allocation_pct=0.0,
            allocation_amount=0.0,
            predicted_return=current.predicted_return,
            expected_return=metrics.expected_return,
            confidence=0.0,
            rationale="No historical track record for this ticker.",
        )

    if metrics.confidence < min_confidence:
        return Recommendation(
            ticker=current.ticker,
            decision="DO_NOT_INVEST",
            allocation_pct=0.0,
            allocation_amount=0.0,
            predicted_return=current.predicted_return,
            expected_return=metrics.expected_return,
            confidence=metrics.confidence,
            rationale=(
                f"Confidence {metrics.confidence:.2f} below threshold {min_confidence:.2f}."
            ),
        )

    if metrics.expected_return < min_expected_return:
        return Recommendation(
            ticker=current.ticker,
            decision="DO_NOT_INVEST",
            allocation_pct=0.0,
            allocation_amount=0.0,
            predicted_return=current.predicted_return,
            expected_return=metrics.expected_return,
            confidence=metrics.confidence,
            rationale=(
                f"Expected return {metrics.expected_return:.4f} below threshold "
                f"{min_expected_return:.4f}."
            ),
        )

    size_from_kelly = max(0.0, metrics.kelly_fraction * fractional_kelly * metrics.confidence)
    allocation_pct = min(max_position_pct, size_from_kelly)
    allocation_amount = capital * allocation_pct

    if allocation_pct <= 0:
        return Recommendation(
            ticker=current.ticker,
            decision="DO_NOT_INVEST",
            allocation_pct=0.0,
            allocation_amount=0.0,
            predicted_return=current.predicted_return,
            expected_return=metrics.expected_return,
            confidence=metrics.confidence,
            rationale="Kelly sizing returned zero allocation.",
        )

    return Recommendation(
        ticker=current.ticker,
        decision="INVEST",
        allocation_pct=allocation_pct,
        allocation_amount=allocation_amount,
        predicted_return=current.predicted_return,
        expected_return=metrics.expected_return,
        confidence=metrics.confidence,
        rationale=(
            f"Positive expected return with confidence {metrics.confidence:.2f}; "
            f"allocated via fractional Kelly."
        ),
    )


def generate_recommendations(
    historical: List[HistoricalPrediction],
    current: List[CurrentPrediction],
    capital: float,
    min_history: int,
    min_confidence: float,
    min_expected_return: float,
    max_position_pct: float,
    fractional_kelly: float,
) -> List[Recommendation]:
    recommendations: List[Recommendation] = []
    for row in current:
        metrics = compute_metrics(historical=historical, current=row, min_history=min_history)
        recommendation = recommend_for_stock(
            current=row,
            metrics=metrics,
            capital=capital,
            min_confidence=min_confidence,
            min_expected_return=min_expected_return,
            max_position_pct=max_position_pct,
            fractional_kelly=fractional_kelly,
        )
        recommendations.append(recommendation)
    return _normalize_allocations(recommendations, capital=capital)


def _normalize_allocations(
    recommendations: List[Recommendation], capital: float
) -> List[Recommendation]:
    invest_rows = [row for row in recommendations if row.decision == "INVEST"]
    total_alloc = sum(row.allocation_pct for row in invest_rows)
    if total_alloc <= 1.0 or not invest_rows:
        return recommendations

    scale = 1.0 / total_alloc
    normalized: List[Recommendation] = []
    for row in recommendations:
        if row.decision != "INVEST":
            normalized.append(row)
            continue
        new_pct = row.allocation_pct * scale
        normalized.append(
            Recommendation(
                ticker=row.ticker,
                decision=row.decision,
                allocation_pct=new_pct,
                allocation_amount=capital * new_pct,
                predicted_return=row.predicted_return,
                expected_return=row.expected_return,
                confidence=row.confidence,
                rationale=(
                    f"{row.rationale} Allocation scaled to keep total capital usage <= 100%."
                ),
            )
        )
    return normalized


def write_recommendations_csv(path: Path, recommendations: List[Recommendation]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "ticker",
                "decision",
                "allocation_pct",
                "allocation_amount",
                "predicted_return",
                "expected_return",
                "confidence",
                "rationale",
            ]
        )
        for row in recommendations:
            writer.writerow(
                [
                    row.ticker,
                    row.decision,
                    f"{row.allocation_pct:.6f}",
                    f"{row.allocation_amount:.2f}",
                    f"{row.predicted_return:.6f}",
                    f"{row.expected_return:.6f}",
                    f"{row.confidence:.6f}",
                    row.rationale,
                ]
            )


def _render_text_table(recommendations: List[Recommendation]) -> str:
    headers = [
        "ticker",
        "decision",
        "alloc_pct",
        "alloc_amt",
        "pred_ret",
        "exp_ret",
        "confidence",
    ]
    rows = [
        [
            row.ticker,
            row.decision,
            f"{row.allocation_pct:.2%}",
            f"${row.allocation_amount:,.2f}",
            f"{row.predicted_return:.2%}",
            f"{row.expected_return:.2%}",
            f"{row.confidence:.2%}",
        ]
        for row in recommendations
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]

    line = " | ".join(value.ljust(width) for value, width in zip(headers, widths))
    separator = "-+-".join("-" * width for width in widths)
    rendered_rows = [
        " | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows
    ]
    return "\n".join([line, separator, *rendered_rows])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate historical stock prediction performance and generate "
            "invest/do-not-invest recommendations with position sizing."
        )
    )
    parser.add_argument("--historical", required=True, type=Path, help="Path to historical CSV.")
    parser.add_argument("--current", required=True, type=Path, help="Path to current CSV.")
    parser.add_argument(
        "--capital", type=float, default=10_000.0, help="Total capital available to allocate."
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=8,
        help="Minimum number of historical rows for full confidence.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.50,
        help="Minimum confidence score required to invest.",
    )
    parser.add_argument(
        "--min-expected-return",
        type=float,
        default=0.01,
        help="Minimum adjusted expected return required to invest.",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.20,
        help="Maximum allocation per ticker as a fraction of capital.",
    )
    parser.add_argument(
        "--fractional-kelly",
        type=float,
        default=0.50,
        help="Fraction of Kelly position size to use (0-1 is typical).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write recommendations as CSV.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output instead of table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.capital <= 0:
        raise ValueError("--capital must be > 0")
    if args.min_history <= 0:
        raise ValueError("--min-history must be > 0")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be between 0 and 1")
    if args.max_position_pct <= 0:
        raise ValueError("--max-position-pct must be > 0")
    if args.fractional_kelly < 0:
        raise ValueError("--fractional-kelly must be >= 0")

    historical = load_historical_predictions(args.historical)
    current = load_current_predictions(args.current)
    recommendations = generate_recommendations(
        historical=historical,
        current=current,
        capital=args.capital,
        min_history=args.min_history,
        min_confidence=args.min_confidence,
        min_expected_return=args.min_expected_return,
        max_position_pct=args.max_position_pct,
        fractional_kelly=args.fractional_kelly,
    )

    if args.output_csv:
        write_recommendations_csv(path=args.output_csv, recommendations=recommendations)

    if args.json:
        print(json.dumps([row.__dict__ for row in recommendations], indent=2))
    else:
        print(_render_text_table(recommendations))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

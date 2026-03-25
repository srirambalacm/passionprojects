from __future__ import annotations

import unittest

from src.stock_recommender import CurrentPrediction, HistoricalPrediction, generate_recommendations


class StockRecommenderTests(unittest.TestCase):
    def test_invest_when_history_is_strong_and_predicted_positive(self) -> None:
        historical = [
            HistoricalPrediction("AAPL", 0.05, 0.04),
            HistoricalPrediction("AAPL", 0.06, 0.03),
            HistoricalPrediction("AAPL", 0.04, 0.05),
            HistoricalPrediction("AAPL", 0.03, 0.02),
            HistoricalPrediction("AAPL", 0.07, 0.06),
            HistoricalPrediction("AAPL", 0.02, 0.01),
            HistoricalPrediction("AAPL", 0.03, 0.04),
            HistoricalPrediction("AAPL", 0.01, 0.02),
        ]
        current = [CurrentPrediction("AAPL", 0.05)]

        recs = generate_recommendations(
            historical=historical,
            current=current,
            capital=10_000,
            min_history=8,
            min_confidence=0.5,
            min_expected_return=0.005,
            max_position_pct=0.2,
            fractional_kelly=0.5,
        )

        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].decision, "INVEST")
        self.assertGreater(recs[0].allocation_pct, 0)
        self.assertGreater(recs[0].allocation_amount, 0)

    def test_do_not_invest_on_negative_prediction(self) -> None:
        historical = [HistoricalPrediction("TSLA", -0.04, -0.03)] * 10
        current = [CurrentPrediction("TSLA", -0.01)]

        recs = generate_recommendations(
            historical=historical,
            current=current,
            capital=5_000,
            min_history=8,
            min_confidence=0.1,
            min_expected_return=0.0,
            max_position_pct=0.2,
            fractional_kelly=0.5,
        )

        self.assertEqual(recs[0].decision, "DO_NOT_INVEST")
        self.assertEqual(recs[0].allocation_pct, 0)
        self.assertEqual(recs[0].allocation_amount, 0)

    def test_total_allocation_is_normalized_to_full_capital(self) -> None:
        historical = []
        for ticker in ("AAA", "BBB", "CCC"):
            historical.extend(
                [
                    HistoricalPrediction(ticker, 0.05, 0.04),
                    HistoricalPrediction(ticker, 0.05, 0.04),
                    HistoricalPrediction(ticker, 0.05, 0.04),
                    HistoricalPrediction(ticker, 0.05, 0.04),
                    HistoricalPrediction(ticker, 0.05, 0.04),
                    HistoricalPrediction(ticker, 0.05, 0.04),
                    HistoricalPrediction(ticker, 0.05, 0.04),
                    HistoricalPrediction(ticker, 0.05, 0.04),
                    HistoricalPrediction(ticker, 0.05, -0.02),
                    HistoricalPrediction(ticker, 0.05, 0.03),
                ]
            )
        current = [
            CurrentPrediction("AAA", 0.1),
            CurrentPrediction("BBB", 0.1),
            CurrentPrediction("CCC", 0.1),
        ]

        recs = generate_recommendations(
            historical=historical,
            current=current,
            capital=10_000,
            min_history=8,
            min_confidence=0.5,
            min_expected_return=0.001,
            max_position_pct=0.7,
            fractional_kelly=2.0,
        )

        total = sum(row.allocation_pct for row in recs if row.decision == "INVEST")
        self.assertLessEqual(total, 1.0000001)


if __name__ == "__main__":
    unittest.main()

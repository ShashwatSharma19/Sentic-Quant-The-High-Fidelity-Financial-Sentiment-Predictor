"""
schemas.py
----------
Production-grade Pydantic V2 data validation schemas for the
Algorithmic Trading pipeline. Import these models at ingestion
boundaries to catch corrupt data before it enters the ML pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from typing_extensions import Annotated


# ---------------------------------------------------------------------------
# Reusable annotated types
# ---------------------------------------------------------------------------

PositiveFloat = Annotated[float, Field(gt=0, description="Must be a positive number")]
BoundedScore = Annotated[
    float,
    Field(ge=-1.0, le=1.0, description="Sentiment score clamped to [-1.0, 1.0]"),
]


# ---------------------------------------------------------------------------
# PriceRow — one OHLCV candle
# ---------------------------------------------------------------------------

class PriceRow(BaseModel):
    """Validates a single row of daily OHLCV market data."""

    Date: datetime = Field(description="Trading date (YYYY-MM-DD or ISO 8601)")
    Open: PositiveFloat = Field(description="Opening price")
    High: PositiveFloat = Field(description="Intraday high price")
    Low: PositiveFloat = Field(description="Intraday low price")
    Close: PositiveFloat = Field(description="Closing price")
    Volume: int = Field(ge=0, description="Number of shares traded")

    @model_validator(mode="after")
    def high_must_dominate(self) -> "PriceRow":
        """
        Enforce OHLC consistency:
          - High  >= Low   (basic candlestick rule)
          - High  >= Close (close cannot exceed the intraday high)
        """
        if self.High < self.Low:
            raise ValueError(
                f"'High' ({self.High}) must be >= 'Low' ({self.Low}). "
                "Check the source data for this row."
            )
        if self.High < self.Close:
            raise ValueError(
                f"'High' ({self.High}) must be >= 'Close' ({self.Close}). "
                "Check the source data for this row."
            )
        return self

    # Allow pandas Timestamps, date-strings, etc. to be coerced automatically
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# SentimentRow — one processed NLP output row
# ---------------------------------------------------------------------------

class SentimentRow(BaseModel):
    """Validates a single row of FinBERT sentiment output."""

    Date: datetime = Field(description="Date the headline was published")
    Headline: str = Field(min_length=1, description="Raw news headline string")
    Score: BoundedScore = Field(
        description="Aggregated sentiment: -1 (Negative), 0 (Neutral), +1 (Positive)"
    )


# ---------------------------------------------------------------------------
# StockDataPayload — root model for a full dataset
# ---------------------------------------------------------------------------

class StockDataPayload(RootModel[List[PriceRow]]):
    """
    Validates an entire list of PriceRow records in a single call.

    Usage
    -----
    >>> payload = StockDataPayload.model_validate(df.to_dict(orient='records'))
    >>> rows: list[PriceRow] = payload.root
    """

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self):  # type: ignore[override]
        return iter(self.root)

    def __getitem__(self, index: int) -> PriceRow:
        return self.root[index]


# ---------------------------------------------------------------------------
# Quick smoke-test  (run with: python src/schemas.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pydantic import ValidationError

    print("=" * 60)
    print("Pydantic V2 Schema Smoke Tests")
    print("=" * 60)

    # --- Test 1: Valid PriceRow ---
    try:
        row = PriceRow(
            Date="2024-01-15",
            Open=150.0,
            High=155.0,
            Low=148.5,
            Close=153.0,
            Volume=1_000_000,
        )
        print(f"\n[PASS] Valid PriceRow created: {row.Date.date()} | Close={row.Close}")
    except ValidationError as e:
        print(f"\n[FAIL] Unexpected error: {e}")

    # --- Test 2: High < Low (should fail) ---
    try:
        bad_row = PriceRow(
            Date="2024-01-15",
            Open=150.0,
            High=140.0,   # <- intentionally wrong
            Low=148.5,
            Close=139.0,
            Volume=500_000,
        )
        print("\n[FAIL] Should have raised ValidationError for High < Low")
    except ValidationError as e:
        print(f"\n[PASS] Correctly rejected High < Low:\n       {e.errors()[0]['msg']}")

    # --- Test 3: Sentiment score out of range (should fail) ---
    try:
        bad_sentiment = SentimentRow(
            Date="2024-01-15",
            Headline="Stock market crashes!",
            Score=2.5,    # <- out of [-1, 1] range
        )
        print("\n[FAIL] Should have raised ValidationError for Score=2.5")
    except ValidationError as e:
        print(f"\n[PASS] Correctly rejected Score=2.5:\n       {e.errors()[0]['msg']}")

    # --- Test 4: Valid StockDataPayload (list) ---
    try:
        payload = StockDataPayload.model_validate([
            {"Date": "2024-01-15", "Open": 150.0, "High": 155.0, "Low": 148.5, "Close": 153.0, "Volume": 1000000},
            {"Date": "2024-01-16", "Open": 153.0, "High": 158.0, "Low": 152.0, "Close": 157.5, "Volume": 900000},
        ])
        print(f"\n[PASS] StockDataPayload validated {len(payload)} rows successfully.")
    except ValidationError as e:
        print(f"\n[FAIL] Unexpected error: {e}")

    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)

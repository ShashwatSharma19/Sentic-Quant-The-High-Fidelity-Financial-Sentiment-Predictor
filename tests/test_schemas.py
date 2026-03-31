"""
test_schemas.py
---------------
Pytest suite bridging continuous integration testing to 
the core Pydantic validation boundaries of the Algorithmic Trading pipeline.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.schemas import PriceRow, SentimentRow, StockDataPayload


def test_price_row_valid():
    """Ensure a standard valid row passes instantly."""
    row = PriceRow(
        Date="2024-01-15",
        Open=150.0,
        High=155.0,
        Low=148.5,
        Close=153.0,
        Volume=100_000,
    )
    assert row.Close == 153.0
    assert isinstance(row.Date, datetime)


def test_price_row_high_less_than_low():
    """A row where the daily high is strictly less than the daily low should fail."""
    with pytest.raises(ValidationError) as exc_info:
        PriceRow(
            Date="2024-01-15",
            Open=150.0,
            High=140.0,  # Invalid: Lower than Low
            Low=148.5,
            Close=145.0,
            Volume=100_000,
        )
    assert "must be >= 'Low'" in str(exc_info.value)


def test_price_row_high_less_than_close():
    """A row where the close exceeds the intraday high should fail."""
    with pytest.raises(ValidationError) as exc_info:
        PriceRow(
            Date="2024-01-15",
            Open=150.0,
            High=155.0,
            Low=150.0,
            Close=160.0,  # Invalid: Higher than High
            Volume=100_000,
        )
    assert "must be >= 'Close'" in str(exc_info.value)


def test_price_row_negative_price():
    """Negative prices are theoretically impossible for standard equities."""
    with pytest.raises(ValidationError) as exc_info:
        PriceRow(
            Date="2024-01-15",
            Open=150.0,
            High=155.0,
            Low=-148.5,  # Invalid
            Close=153.0,
            Volume=100_000,
        )
    assert "greater than 0" in str(exc_info.value)


def test_sentiment_row_out_of_bounds():
    """FinBERT probabilities/vectors must strictly map to [-1.0, 1.0]."""
    with pytest.raises(ValidationError) as exc_info:
        SentimentRow(Date="2024-01-15", Headline="Earnings call!", Score=1.5)
    assert "less than or equal to 1" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        SentimentRow(Date="2024-01-15", Headline="Lawsuit!", Score=-1.1)
    assert "greater than or equal to -1" in str(exc_info.value)


def test_payload_list_parsing():
    """Ensure the RootModel cleanly ingests structured lists of dictionaries."""
    data = [
        {"Date": "2024-01-15", "Open": 150.0, "High": 155.0, "Low": 148.5, "Close": 153.0, "Volume": 100},
        {"Date": "2024-01-16", "Open": 153.0, "High": 158.0, "Low": 152.0, "Close": 157.0, "Volume": 200},
    ]
    payload = StockDataPayload.model_validate(data)
    assert len(payload) == 2
    assert payload[1].Close == 157.0

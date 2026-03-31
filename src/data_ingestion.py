"""
data_ingestion.py
-----------------
Production-grade DataPipeline class.

Responsibilities:
  1. Fetch historical OHLCV price data via yfinance.
  2. Generate mock news headlines for the trading period.
  3. Validate price data using the Pydantic V2 StockDataPayload schema.
  4. Merge price + sentiment data using Polars (fast, zero-copy).
  5. Persist the final feature table as a Parquet file for downstream consumption.

Output
------
  data/processed/pipeline_output.parquet
  data/raw/stock_data.csv   (raw price backup)
  data/raw/news_data.csv    (raw news backup)
"""

from __future__ import annotations

import logging
import pathlib
import urllib.parse
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

import feedparser
import numpy as np
import polars as pl
import yfinance as yf
from pydantic import ValidationError

from src.schemas import StockDataPayload
from src.sentiment_analysis import SentimentAnalyzer

# ---------------------------------------------------------------------------
# Module-level logger — callers can configure handlers/level themselves
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Orchestrates data ingestion, validation, transformation, and persistence.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. "AAPL".
    start_date : str
        ISO 8601 start date string, e.g. "2020-01-01".
    end_date : str
        ISO 8601 end date string, e.g. "2024-01-01".
    output_dir : str | pathlib.Path
        Root directory to write output files. Defaults to current directory.
    """

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        output_dir: str | pathlib.Path = ".",
    ) -> None:
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = pathlib.Path(output_dir)
        self._log = logging.getLogger(f"{__name__}.{self.ticker}")
        self.sentiment_analyzer = SentimentAnalyzer(logger=self._log)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_price_data(self) -> list[dict]:
        """Pull OHLCV data from Yahoo Finance, return as a list of dicts."""
        self._log.info(
            "Fetching %s price data from %s to %s …",
            self.ticker,
            self.start_date,
            self.end_date,
        )
        stock = yf.Ticker(self.ticker)
        df_pd = stock.history(start=self.start_date, end=self.end_date)

        if df_pd.empty:
            raise ValueError(
                f"yfinance returned no data for ticker '{self.ticker}'. "
                "Check that the symbol is valid and the date range is correct."
            )

        # Normalise index to plain date string for Pydantic datetime parsing
        df_pd.index = df_pd.index.strftime("%Y-%m-%d")
        df_pd.index.name = "Date"
        df_pd = df_pd[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        df_pd["Volume"] = df_pd["Volume"].astype(int)

        # Save raw backup
        raw_path = self.output_dir / "data" / "raw" / "stock_data.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df_pd.to_csv(raw_path)
        self._log.info("Raw price data saved → %s", raw_path)

        # Convert to list-of-dicts for Pydantic
        records = df_pd.reset_index().to_dict(orient="records")
        return records

    def _fetch_real_news(self) -> list[dict]:
        """Fetch actual news headlines via Google News RSS for the ticker."""
        self._log.info("Fetching real news headlines via Google News RSS...")
        
        # URL encode ticker
        query = urllib.parse.quote_plus(f"{self.ticker} stock")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(url)
        data: list[dict] = []
        
        for entry in feed.entries:
            try:
                # parsedate_to_datetime parses RFC 2822 dates like 'Thu, 26 Mar 2026 12:00:00 GMT'
                dt = parsedate_to_datetime(entry.published)
                date_str = dt.strftime("%Y-%m-%d")
                
                # Filter strictly to our date range
                if self.start_date <= date_str <= self.end_date:
                    data.append({"Date": date_str, "Headline": entry.title})
            except Exception:
                pass
                
        # Save raw backup
        raw_path = self.output_dir / "data" / "raw" / "news_data.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if data:
            pl.DataFrame(data).write_csv(str(raw_path))
            
        self._log.info("Fetched %d real headlines.", len(data))
        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_and_validate(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Fetch price + news data, validate price rows via Pydantic, and
        return two clean Polars DataFrames: (price_df, news_df).

        Raises
        ------
        RuntimeError
            If Pydantic validation finds corrupt rows in the price data.
        ValueError
            If yfinance returns no data for the given ticker / date range.
        """
        raw_price_records = self._fetch_price_data()
        raw_news_records = self._fetch_real_news()

        # --- Pydantic validation gate ---
        try:
            payload = StockDataPayload.model_validate(raw_price_records)
            self._log.info(
                "Pydantic validation passed ✓  %d price rows validated.", len(payload)
            )
        except ValidationError as exc:
            # Log every individual field error in structured detail
            for err in exc.errors():
                self._log.error(
                    "Validation failure | row=%s | field=%s | msg=%s",
                    err.get("loc", ("?",))[0],
                    " → ".join(str(x) for x in err.get("loc", [])),
                    err.get("msg", "unknown error"),
                )
            raise RuntimeError(
                f"Price data failed Pydantic validation with {exc.error_count()} "
                "error(s). Check logs for details. Pipeline aborted."
            ) from exc

        # --- Convert validated records to Polars ---
        price_df = pl.DataFrame(
            [row.model_dump() for row in payload]
        ).with_columns(pl.col("Date").cast(pl.Date))

        if not raw_news_records:
            news_df = pl.DataFrame(schema={"Date": pl.Date, "Headline": pl.Utf8})
        else:
            news_df = (
                pl.DataFrame(raw_news_records)
                .with_columns(pl.col("Date").str.to_date())
            )

        return price_df, news_df

    def merge_and_save(self) -> pathlib.Path:
        """
        Full pipeline entry-point:
          1. fetch_and_validate()
          2. Compute a daily mean sentiment score from news (mock: random mapping).
          3. Left-join sentiment onto price data.
          4. Fill missing sentiment with 0 (Neutral).
          5. Write final DataFrame to Parquet.

        Returns
        -------
        pathlib.Path
            Absolute path to the written Parquet file.
        """
        price_df, news_df = self.fetch_and_validate()

        # --- Produce a daily sentiment score using real FinBERT ---
        if news_df.height > 0:
            headlines = news_df["Headline"].to_list()
            scores = self.sentiment_analyzer.analyze_headlines(headlines)
            
            daily_sentiment = (
                news_df.with_columns(pl.Series(name="score", values=scores))
                .group_by("Date")
                .agg(pl.col("score").mean().alias("daily_sentiment_score"))
            )
        else:
            daily_sentiment = pl.DataFrame(
                schema={"Date": pl.Date, "daily_sentiment_score": pl.Float64}
            )

        # --- Left-join so every trading day is preserved ---
        merged_df = price_df.join(daily_sentiment, on="Date", how="left").with_columns(
            pl.col("daily_sentiment_score").fill_null(0.0)
        )

        self._log.info(
            "Merged price + sentiment → shape: (%d rows × %d cols)",
            merged_df.height,
            merged_df.width,
        )

        # --- Persist as Parquet ---
        out_path = self.output_dir / "data" / "processed" / "pipeline_output.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.write_parquet(str(out_path))
        self._log.info("Pipeline output saved → %s", out_path)

        return out_path.resolve()


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime

    pipeline = DataPipeline(
        ticker="AAPL",
        start_date="2020-01-01",
        end_date=datetime.today().strftime("%Y-%m-%d"),
        output_dir=".",
    )

    output_path = pipeline.merge_and_save()

    # Quick sanity-check: read the Parquet back and print schema + head
    result = pl.read_parquet(str(output_path))
    print("\n--- Output Schema ---")
    print(result.schema)
    print("\n--- First 5 Rows ---")
    print(result.head(5))
    print(f"\n✅ Pipeline complete. {result.height} rows written to: {output_path}")

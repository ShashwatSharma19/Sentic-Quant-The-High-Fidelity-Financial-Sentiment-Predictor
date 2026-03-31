"""
sentiment_analysis.py
---------------------
Production-grade SentimentAnalyzer class.
Loads the FinBERT model once and exposes methods to score real news headlines.
"""

from __future__ import annotations

import logging

import torch
from transformers import pipeline


class SentimentAnalyzer:
    """
    Wrapper for the ProsusAI/finbert model.
    Converts financial text directly into numerical trading signals.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or logging.getLogger(__name__)
        
        # Standard ProsusAI FinBERT model
        self.model_name = "ProsusAI/finbert"
        
        # Auto-detect if CUDA (GPU) is available, otherwise fallback to CPU
        self.device = 0 if torch.cuda.is_available() else -1
        self._log.info(
            "Loading FinBERT model ('%s') on %s ...",
            self.model_name,
            "GPU" if self.device == 0 else "CPU",
        )
        
        # Initialize Hugging Face pipeline
        self.nlp = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            tokenizer=self.model_name,
            device=self.device,
        )
        
        # Fixed mapping from textual labels to numerical signal multiplier
        self.label_map = {
            "positive": 1.0,
            "negative": -1.0,
            "neutral": 0.0
        }

    def analyze_headlines(self, headlines: list[str]) -> list[float]:
        """
        Runs batch evaluation of headlines through FinBERT.
        
        Returns:
            list[float]: A list of aggregated scores where:
                         positive = +score
                         negative = -score
                         neutral  = 0.0
        """
        if not headlines:
            return []
            
        self._log.info("Analyzing %d headlines...", len(headlines))
        
        # Returns list of dicts: [{'label': 'positive', 'score': 0.89}, ...]
        results = self.nlp(headlines)
        
        scores: list[float] = []
        for res in results:
            label = res["label"]
            confidence = res["score"]
            multiplier = self.label_map.get(label, 0.0)
            
            # For this pipeline, we multiply the direction by the confidence score.
            # e.g., 'positive' with 0.9 confidence -> +0.9
            scores.append(multiplier * confidence)
            
        return scores

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List


def build_ingest_smoke_samples() -> List[Dict[str, Any]]:
    return [
        {
            "source_id": "sess-s1",
            "metadata": {
                "case_id": "ingest_case_001",
                "user_id": "user_alice",
                "conv_id": "conv_alpha",
                "session": "s1",
                "session_time": "2023-07-01T09:00:00Z",
                "period": "2023-Q3",
                "category": "travel",
                "topic": "camping",
            },
            "text": """[Session: s1]\n[Time: 2023-07-01]\nAlice: Last Friday I bought a blue ceramic bowl with white dots.\nBob: I noted it next to pets: Oliver, Luna, Bailey.\nAlice: Two weeks ago we planned camping at Voyageurs National Park.""",
        },
        {
            "source_id": "sess-s2",
            "metadata": {
                "case_id": "ingest_case_001",
                "user_id": "user_alice",
                "conv_id": "conv_alpha",
                "session": "s2",
                "session_time": "2023-07-02T18:30:00Z",
                "period": "2023-Q3",
                "category": "bio",
                "topic": "family",
            },
            "text": """[Session: s2]\n[Time: 2023-07-02]\nAlicia: My dog named Scout was adopted from a breeder.\nAlice: I have a sunflower tattoo.\nBob: activities: pottery, hiking, camping.\nBob: This is not true -> Alice owns a purple dragon.""",
        },
    ]


def build_queries() -> Dict[str, str]:
    return {
        "q1": "What unique bowl detail was mentioned?",
        "q2": "Where did they plan camping?",
        "q3": "Who has a sunflower tattoo?",
        "negative": "Did Alice own a purple dragon?",
    }


def expected_facts() -> Dict[str, str]:
    return {
        "q1": "blue ceramic bowl with white dots",
        "q2": "Voyageurs National Park",
        "q3": "sunflower tattoo",
        "negative": "purple dragon",
    }

#!/usr/bin/env python3
"""Rank correlation without a scipy dependency.

Several audits compare a candidate score against a semantic outcome and need
Spearman's rho with ties averaged. They each carried their own copy.
"""
import numpy as np


def rankdata(x):
    """Average ranks, ties shared, matching scipy's default."""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[o[j]] == x[o[i]]:
            j += 1
        r[o[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return r


def corr(x, y):
    """Pearson correlation, nan when either side is constant or too short."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return corr(rankdata(x), rankdata(y))

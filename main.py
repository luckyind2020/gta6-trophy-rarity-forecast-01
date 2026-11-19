"""Week-one trophy rarity forecaster for GTA VI on PlayStation 5.

Reads every ``<id>.in`` case file from the input folder and writes one
``<id>.out`` forecast per case to the output folder. Each case hands us the
house recorder's reads of Exophase's GTA VI trophy page up to a checkpoint,
and asks for five quantiles of every trophy's earn rate (percent of tracked
players) at the case's target moment.

How the forecast is made, per trophy (model v1):

* Turn each past read into a point on the logit scale: the horizontal
  coordinate is ``ln(1 + hours since launch midnight 2026-11-19 UTC)`` and
  the vertical coordinate is the logit of that read's earn rate. Reads with
  a missing rate are left out.
* When the logit points barely move (range under 0.1) the trophy is treated
  as flat: the median is the last rate divided by 3.5, because Exophase's
  first refresh of the night cut every rate by roughly that factor, except
  a rate at (or above) 99.99%, which stays near 99.5. The logit spread is
  1.3.
* Otherwise a recency-weighted straight line is fitted to the logit points
  (older reads count half as much every 12 hours), the slope is shrunk
  toward zero by ``n / (n + 2)``, and the line is read off at the target.
  That trend rate is mixed 75/25 with a tier prior that grows with the
  target hour (bronze 6% at 5h rising to 25% at 168h, silver 3% to 12%,
  gold 1.5% to 5%, platinum 0.4% to 0.8%), then capped at the tier's
  ceiling (bronze 90%, silver 60%, gold 35%, platinum 3%) and a floor of
  0.05%. The spread combines the fit residuals with a random-walk term
  that grows with the horizon in days, floored at 0.5 and capped at 2.5.
* Quantiles are the median's logit shifted by the normal z-scores
  (-1.645, -0.674, 0, 0.674, 1.645) times the spread, mapped back to
  percent and rounded to 4 decimals, half up.

Standard library only; no network, no randomness, so two runs of the same
case write identical bytes.
"""

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

LAUNCH_MIDNIGHT = datetime(2026, 11, 19, tzinfo=timezone.utc)
ZSCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)
FLAT_LOGIT_RANGE = 0.1
FLAT_SPREAD = 1.3
REFRESH_DIVISOR = 3.5
NEARLY_EVERYONE = 99.99
NEARLY_EVERYONE_MEDIAN = 99.5
MIN_MEDIAN = 0.05

# Tier prior: (rate at 5h after launch, rate at 168h) in percent.
TIER_PRIOR = {
    "bronze": (6.0, 25.0),
    "silver": (3.0, 12.0),
    "gold": (1.5, 5.0),
    "platinum": (0.4, 0.8),
}
# Hard ceiling on the median forecast per tier, in percent.
TIER_CEILING = {
    "bronze": 90.0,
    "silver": 60.0,
    "gold": 35.0,
    "platinum": 3.0,
}


@dataclass
class PastRead:
    """One usable past observation for a single trophy."""

    hours: float
    rate: float


def moment_from(text):
    """Parse an ISO-8601 timestamp (with a trailing Z) into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_after_launch(moment):
    """Hours between launch midnight (2026-11-19T00:00Z) and ``moment``."""
    return (moment - LAUNCH_MIDNIGHT).total_seconds() / 3600.0


def logit_of(pct):
    """Logit of a percent, with the rate first squeezed into [0.01, 99.99]."""
    squeezed = min(99.99, max(0.01, pct))
    return math.log(squeezed / (100.0 - squeezed))


def expit_of(y):
    """Inverse logit, back to a percent in (0, 100)."""
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y)) if y < 700 else 100.0
    return 100.0 * math.exp(y) / (1.0 + math.exp(y)) if y > -700 else 0.0


def round_half_up(value):
    """Round to 4 decimals, halves away from zero, via decimal arithmetic."""
    stepped = Decimal(repr(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    )
    return float(stepped)


def prior_for_tier(tier, target_hours):
    """Tier prior at the target hour, interpolated on the log-hours axis."""
    low, high = TIER_PRIOR.get(tier, (6.0, 25.0))
    start = math.log(1.0 + 5.0)
    end = math.log(1.0 + 168.0)
    here = math.log(1.0 + max(target_hours, 0.0))
    fraction = (here - start) / (end - start)
    fraction = min(1.0, max(0.0, fraction))
    return low + fraction * (high - low)


def spread_from_residuals(residuals, horizon_days):
    """Logit spread from fit residuals plus a horizon random-walk term."""
    count = len(residuals)
    mean_resid = sum(residuals) / count if count else 0.0
    variance = sum((r - mean_resid) ** 2 for r in residuals) / count if count else 0.0
    fit_spread = min(1.2, max(0.3, math.sqrt(variance)))
    total = math.sqrt(
        fit_spread ** 2 + (0.5 * math.sqrt(horizon_days)) ** 2 + 0.3 ** 2
    )
    return min(2.5, max(0.5, total))


def tidy_row(values):
    """Force a quantile row into [0, 100] and non-decreasing order.

    Rounding to 4 decimals can only tie neighbours, never invert them, so
    this is belt-and-braces: the grader rejects a row that steps down.
    """
    cleaned = []
    previous = 0.0
    for value in values:
        stepped = min(100.0, max(0.0, value))
        stepped = max(previous, stepped)
        cleaned.append(stepped)
        previous = stepped
    return cleaned


def forecast_one(tier, reads, target_hours, horizon_days):
    """Five quantiles (percent) for one trophy from its past reads."""
    if not reads:
        median = prior_for_tier(tier, target_hours)
        spread = FLAT_SPREAD
    else:
        logits = [logit_of(read.rate) for read in reads]
        if max(logits) - min(logits) < FLAT_LOGIT_RANGE:
            last_rate = reads[-1].rate
            if last_rate >= NEARLY_EVERYONE:
                median = NEARLY_EVERYONE_MEDIAN
            else:
                median = last_rate / REFRESH_DIVISOR
            spread = FLAT_SPREAD
        else:
            xs = [math.log(1.0 + read.hours) for read in reads]
            last_hours = reads[-1].hours
            weights = [
                0.5 ** ((last_hours - read.hours) / 12.0) for read in reads
            ]
            weight_sum = sum(weights)
            x_bar = sum(w * x for w, x in zip(weights, xs)) / weight_sum
            y_bar = sum(w * y for w, y in zip(weights, logits)) / weight_sum
            sxx = sum(w * (x - x_bar) ** 2 for w, x in zip(weights, xs))
            sxy = sum(
                w * (x - x_bar) * (y - y_bar)
                for w, x, y in zip(weights, xs, logits)
            )
            slope = sxy / sxx if sxx > 0 else 0.0
            slope *= len(reads) / (len(reads) + 2)
            x_target = math.log(1.0 + max(target_hours, 0.0))
            trend_rate = expit_of(y_bar + slope * (x_target - x_bar))
            median = 0.75 * trend_rate + 0.25 * prior_for_tier(
                tier, target_hours
            )
            ceiling = TIER_CEILING.get(tier, 90.0)
            median = min(ceiling, max(MIN_MEDIAN, median))
            residuals = [
                y - (y_bar + slope * (x - x_bar))
                for x, y in zip(xs, logits)
            ]
            spread = spread_from_residuals(residuals, horizon_days)
    centre = logit_of(median)
    raw = [round_half_up(expit_of(centre + z * spread)) for z in ZSCORES]
    return tidy_row(raw)


def read_cases():
    """Load every case file from the input folder, in file-name order."""
    folder = Path(os.getenv("MAKEFUN_INPUT_DIR", "/input"))
    found = []
    for path in sorted(folder.glob("*.in")):
        with open(path, encoding="utf-8") as handle:
            found.append((path.stem, json.load(handle)))
    return found


def forecast_whole_case(case):
    """Forecast every trophy of one case; rows sorted by trophy id."""
    target_hours = hours_after_launch(moment_from(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        reads = []
        for entry in case["history"]:
            rate = entry["rates"][index]
            if rate is None:
                continue
            stamp = entry.get("at") or entry["slot"]
            reads.append(
                PastRead(
                    hours=hours_after_launch(moment_from(stamp)),
                    rate=float(rate),
                )
            )
        rows.append(
            {
                "id": trophy["id"],
                "q": forecast_one(
                    trophy["tier"], reads, target_hours, horizon_days
                ),
            }
        )
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def run():
    """Read all cases, forecast each, and write one .out file per case."""
    out_folder = Path(os.getenv("MAKEFUN_OUTPUT_DIR", "/output"))
    out_folder.mkdir(parents=True, exist_ok=True)
    for stem, case in read_cases():
        forecast = forecast_whole_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        with open(out_folder / f"{stem}.out", "w", encoding="utf-8") as handle:
            handle.write(text)


run()

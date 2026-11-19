"""Builds a synthetic private equity fund dataset with realistic structure.

Real fund-level returns (Preqin, PitchBook, Burgiss, Cambridge Associates) are licensed and can't be
redistributed, so this generator encodes relationships documented in PE research instead:

- manager persistence: a GP's prior-fund performance predicts the next fund (Kaplan & Schoar, 2005)
- vintage-year cycles: funds raised into expensive, frothy markets earn less
- size: very large funds see diminishing returns
- strategy: venture has the widest dispersion, credit the narrowest
- leverage helps buyouts when rates are low and hurts when they are high

The target is a fund's net IRR (%). Every number here is simulated; swap in a licensed dataset with the
same columns to train on real funds (see README).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGIES = {  # base net IRR %, dispersion %, typical leverage (debt / EBITDA)
    "Buyout": (14.0, 7.0, 5.0),
    "Growth": (13.0, 9.0, 2.0),
    "Venture": (12.0, 18.0, 0.0),
    "Credit": (9.0, 3.5, 0.0),
    "Real Assets": (10.0, 6.0, 4.0),
}
REGIONS = ["North America", "Europe", "Asia", "Global"]
SECTORS = ["Technology", "Healthcare", "Industrials", "Consumer", "Financials", "Energy", "Diversified"]


def macro_by_vintage(rng):
    """One macro environment per vintage year (2000-2020)."""
    years = np.arange(2000, 2021)
    gdp = rng.normal(2.2, 1.2, len(years))
    gdp[years == 2008] -= 3.0
    gdp[years == 2009] -= 4.0
    gdp[years == 2020] -= 5.0
    rates = np.clip(5.5 - 0.27 * (years - 2000) + rng.normal(0, 0.6, len(years)), 0.1, 6.5)
    rates[(years >= 2009) & (years <= 2015)] = np.clip(rates[(years >= 2009) & (years <= 2015)], 0.1, 0.5)
    multiples = 8.0 + 0.18 * (years - 2000) + rng.normal(0, 0.7, len(years))   # entry EV/EBITDA
    public = rng.normal(8.0, 6.0, len(years))                                 # public market return over the fund life
    return pd.DataFrame({"vintage": years, "gdp_growth": gdp, "rate_10y": rates,
                         "entry_ev_ebitda": multiples, "public_mkt_return": public})


def build(n, seed):
    rng = np.random.default_rng(seed)
    macro = macro_by_vintage(rng)
    strategy = rng.choice(list(STRATEGIES), n, p=[0.38, 0.17, 0.25, 0.12, 0.08])
    vintage = rng.integers(2000, 2021, n)
    df = pd.DataFrame({
        "fund_id": [f"F{i:05d}" for i in range(n)],
        "vintage": vintage,
        "strategy": strategy,
        "region": rng.choice(REGIONS, n, p=[0.5, 0.25, 0.15, 0.10]),
        "sector_focus": rng.choice(SECTORS, n),
        "fund_size_musd": np.round(np.exp(rng.normal(6.3, 1.1, n)), 1),       # ~$550M median
        "fund_number": rng.integers(1, 12, n),                                 # 1 = first-time fund
    }).merge(macro, on="vintage", how="left")

    base, spread, lev = (np.array([STRATEGIES[s][i] for s in strategy]) for i in range(3))
    skill = rng.normal(0, 1, n)                                                # unobserved manager skill
    first_time = df["fund_number"] == 1
    df["prior_fund_tvpi"] = np.where(first_time, np.nan,
                                     np.round(1.55 + 0.32 * skill + rng.normal(0, 0.25, n), 2))
    df["prior_fund_irr"] = np.where(first_time, np.nan,
                                    np.round(base + 0.55 * spread * skill + rng.normal(0, 3, n), 1))
    df["leverage_debt_ebitda"] = np.round(np.clip(lev + rng.normal(0, 0.8, n) * (lev > 0), 0, 8), 2)
    df["gp_commitment_pct"] = np.round(np.clip(rng.normal(2.0, 1.0, n), 0.5, 6), 2)
    df["management_fee_pct"] = np.round(np.clip(rng.normal(1.9, 0.25, n), 1.0, 2.5), 2)
    df["portfolio_companies"] = np.clip(rng.poisson(np.where(strategy == "Venture", 28, 12)), 3, 80)
    df["top10_concentration_pct"] = np.round(np.clip(100 * 10 / df["portfolio_companies"] + rng.normal(0, 8, n), 12, 100), 1)
    df["exit_ipo_share_pct"] = np.round(np.clip(rng.normal(np.where(strategy == "Venture", 25, 10), 8, n), 0, 70), 1)

    irr = (base
           + 0.55 * spread * skill                                            # persistence
           - 0.9 * (df["entry_ev_ebitda"] - 10)                               # bought expensive
           + 0.8 * df["gdp_growth"]
           + 0.35 * df["public_mkt_return"]
           - 1.6 * np.log(df["fund_size_musd"] / 500).clip(lower=0)           # diseconomies of scale
           + np.where(strategy == "Buyout", 0.9 * (df["leverage_debt_ebitda"] - 5) * (3.0 - df["rate_10y"]) / 3, 0)
           + 0.6 * (df["gp_commitment_pct"] - 2)                              # skin in the game
           - 1.5 * (df["management_fee_pct"] - 1.9)
           + 0.06 * (df["exit_ipo_share_pct"] - 10)
           - 1.2 * first_time
           + rng.normal(0, 0.45, n) * spread)                                 # luck
    df["net_irr_pct"] = np.round(irr, 2)
    df["top_quartile"] = (df["net_irr_pct"] >= df.groupby(["vintage", "strategy"])["net_irr_pct"].transform(lambda s: s.quantile(0.75))).astype(int)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/pe_funds_synthetic.csv")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    data = build(a.n, a.seed)
    data.to_csv(a.out, index=False)
    print(f"wrote {len(data):,} synthetic funds to {a.out}")

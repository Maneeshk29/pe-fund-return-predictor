"""Trains and compares Ridge Regression, Random Forest and XGBoost for predicting PE fund net IRR.

Funds are split by vintage year (train on older vintages, test on newer ones), which is how a screening
model is used in practice: you only know the past when you evaluate a new fund.

    python src/train.py --data data/pe_funds_synthetic.csv --test-from 2016
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

TARGET = "net_irr_pct"
CATEGORICAL = ["strategy", "region", "sector_focus"]
NUMERIC = ["fund_size_musd", "fund_number", "prior_fund_tvpi", "prior_fund_irr", "leverage_debt_ebitda",
           "gp_commitment_pct", "management_fee_pct", "portfolio_companies", "top10_concentration_pct",
           "exit_ipo_share_pct", "gdp_growth", "rate_10y", "entry_ev_ebitda", "public_mkt_return"]


def features(df):
    X = df[CATEGORICAL + NUMERIC].copy()
    X["first_time_fund"] = (df["fund_number"] == 1).astype(int)            # prior-fund fields are blank for these
    X["log_fund_size"] = np.log(df["fund_size_musd"])
    return X


def preprocess():
    num = NUMERIC + ["first_time_fund", "log_fund_size"]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


MODELS = {
    "Ridge Regression": RidgeCV(alphas=np.logspace(-2, 3, 30)),
    "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=7),
    "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, random_state=7, n_jobs=4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pe_funds_synthetic.csv")
    ap.add_argument("--test-from", type=int, default=2016, help="first vintage year held out for testing")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    df = pd.read_csv(a.data)
    train, test = df[df.vintage < a.test_from], df[df.vintage >= a.test_from]
    Xtr, ytr, Xte, yte = features(train), train[TARGET], features(test), test[TARGET]
    print(f"train: {len(train):,} funds (vintages {train.vintage.min()}-{train.vintage.max()}) | "
          f"test: {len(test):,} funds (vintages {test.vintage.min()}-{test.vintage.max()})")

    rows, fitted = [], {}
    baseline = np.full(len(yte), ytr.mean())
    rows.append({"model": "Baseline (training mean)", "cv_rmse": None, "rmse": root_mean_squared_error(yte, baseline),
                 "mae": mean_absolute_error(yte, baseline), "r2": r2_score(yte, baseline)})
    for name, model in MODELS.items():
        pipe = Pipeline([("prep", preprocess()), ("model", model)])
        cv = -cross_val_score(pipe, Xtr, ytr, groups=train.vintage, cv=GroupKFold(5), scoring="neg_root_mean_squared_error")
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        fitted[name] = (pipe, pred)
        rows.append({"model": name, "cv_rmse": cv.mean(), "rmse": root_mean_squared_error(yte, pred),
                     "mae": mean_absolute_error(yte, pred), "r2": r2_score(yte, pred)})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))

    best = res.iloc[1:].sort_values("rmse").iloc[0]["model"]
    pipe, pred = fitted[best]

    # How well does the best model screen for top-quartile funds?
    test = test.assign(pred=pred)
    test["pred_rank"] = test.groupby(["vintage", "strategy"])["pred"].rank(pct=True)
    picked = test[test.pred_rank >= 0.75]
    hit_rate = picked["top_quartile"].mean()
    screen = {"best_model": best, "picked_funds": int(len(picked)), "top_quartile_hit_rate": round(float(hit_rate), 3),
              "random_hit_rate": round(float(test["top_quartile"].mean()), 3),
              "avg_irr_picked": round(float(picked[TARGET].mean()), 2), "avg_irr_all": round(float(test[TARGET].mean()), 2)}
    print(json.dumps(screen, indent=1))

    # Feature importance (gain) from XGBoost
    xgb_pipe = fitted["XGBoost"][0]
    names = xgb_pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(xgb_pipe.named_steps["model"].feature_importances_, index=[n.split("__", 1)[1] for n in names])
    imp = imp.groupby(lambda n: next((c for c in CATEGORICAL if n.startswith(c + "_")), n)).sum().sort_values()

    # Charts: model comparison, predicted vs actual, importance
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    fig.patch.set_facecolor("white")
    m = res.iloc[1:]
    axes[0].barh(m["model"], m["rmse"], color=["#1d4ed8" if n == best else "#94a3b8" for n in m["model"]])
    axes[0].axvline(res.iloc[0]["rmse"], color="#ef4444", ls="--", lw=1.2, label="baseline")
    axes[0].set_title("Test RMSE (net IRR, pct pts)\nlower is better", fontsize=11)
    for i, v in enumerate(m["rmse"]): axes[0].text(v, i, f" {v:.2f}", va="center", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].scatter(yte, pred, s=7, alpha=0.35, color="#1d4ed8")
    lim = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
    axes[1].plot(lim, lim, color="#ef4444", lw=1)
    axes[1].set(xlabel="Actual net IRR (%)", ylabel="Predicted net IRR (%)")
    axes[1].set_title(f"{best}: predicted vs actual\nR² = {r2_score(yte, pred):.2f} on {a.test_from}+ vintages", fontsize=11)
    top = imp.tail(10)
    axes[2].barh(top.index, top.values, color="#1d4ed8")
    axes[2].set_title("XGBoost feature importance (gain)", fontsize=11)
    for ax in axes: ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8.5)
    fig.suptitle("Private Equity Fund Return Predictor (synthetic data)", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "model_results.png", dpi=150)

    res.round(4).to_csv(out / "metrics.csv", index=False)
    (out / "screening.json").write_text(json.dumps(screen, indent=2))
    imp.sort_values(ascending=False).round(4).to_csv(out / "feature_importance.csv", header=["importance"])
    print(f"saved charts and metrics to {out}/")


if __name__ == "__main__":
    main()

"""Length-controlled statistical models and windowed Odds Ratios for RQ1."""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def compute_windowed_odds_ratios(
    step_records: List[Dict[str, Any]],
    windows: List[int] = [1, 3, 5],
) -> Dict[int, Dict[str, Any]]:
    """Compute lead-lag Odds Ratios: likelihood of error onset following stale vs fresh observations."""
    results = {}
    df = pd.DataFrame(step_records)
    if df.empty or "is_stale" not in df.columns or "is_incorrect" not in df.columns:
        return results

    for w in windows:
        # Check if an incorrect step occurs within next w steps
        # Group by trajectory and create forward rolling max
        df[f"error_in_next_{w}"] = (
            df.groupby("traj_id")["is_incorrect"]
            .transform(lambda s: s.iloc[::-1].rolling(window=w, min_periods=1).max().iloc[::-1])
        )

        contingency = pd.crosstab(df["is_stale"], df[f"error_in_next_{w}"])
        # Ensure 2x2 shape
        a = contingency.loc[True, 1.0] if (True in contingency.index and 1.0 in contingency.columns) else 0
        b = contingency.loc[True, 0.0] if (True in contingency.index and 0.0 in contingency.columns) else 0
        c = contingency.loc[False, 1.0] if (False in contingency.index and 1.0 in contingency.columns) else 0
        d = contingency.loc[False, 0.0] if (False in contingency.index and 0.0 in contingency.columns) else 0

        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])
        # Standard error and 95% CI
        se = np.sqrt(1/(a + 0.5) + 1/(b + 0.5) + 1/(c + 0.5) + 1/(d + 0.5))
        log_or = np.log(odds_ratio) if odds_ratio > 0 else 0
        ci_lower = np.exp(log_or - 1.96 * se)
        ci_upper = np.exp(log_or + 1.96 * se)

        results[w] = {
            "window": w,
            "odds_ratio": float(odds_ratio),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "p_value": float(p_value),
            "table": {"a": int(a), "b": int(b), "c": int(c), "d": int(d)}
        }

    return results


def fit_length_controlled_logistic_regression(step_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fit logistic regression predicting error at step t controlling for step position and run length.

    Model: logit(P(error_t)) ~ beta0 + beta1*is_stale + beta2*(step/total_steps) + beta3*total_steps + agent_dummies
    """
    df = pd.DataFrame(step_records)
    if len(df) < 20 or "is_stale" not in df.columns or "is_incorrect" not in df.columns:
        return {"error": "Insufficient step records for regression"}

    df["is_stale_int"] = df["is_stale"].astype(int)
    df["relative_step"] = df["step_id"] / df["total_steps"].clip(lower=1)
    df["outcome"] = df["is_incorrect"].astype(int)

    # Base design matrix
    X_cols = ["is_stale_int", "relative_step", "total_steps"]
    
    # Add framework dummy variables if multiple frameworks exist
    if "framework" in df.columns and df["framework"].nunique() > 1:
        framework_dummies = pd.get_dummies(df["framework"], drop_first=True, prefix="fw", dtype=float)
        X = pd.concat([df[X_cols], framework_dummies], axis=1)
    else:
        X = df[X_cols]

    X = sm.add_constant(X)
    y = df["outcome"]

    # Check variation in outcome
    if y.nunique() < 2:
        return {"error": "Outcome variable has no variation (no errors or all errors)"}

    try:
        model = sm.Logit(y, X).fit(disp=False)
        params = model.params.to_dict()
        pvalues = model.pvalues.to_dict()
        conf_int = model.conf_int().to_dict()

        beta_stale = params.get("is_stale_int", 0.0)
        p_stale = pvalues.get("is_stale_int", 1.0)
        odds_ratio_stale = float(np.exp(beta_stale))

        return {
            "converged": bool(model.mle_retvals.get("converged", True)),
            "pseudo_r2": float(model.prsquared),
            "stale_beta": float(beta_stale),
            "stale_odds_ratio": odds_ratio_stale,
            "stale_pvalue": float(p_stale),
            "summary_table": {
                k: {"coef": float(params[k]), "p_val": float(pvalues[k])}
                for k in params
            }
        }
    except Exception as e:
        return {"error": str(e)}


def fit_stale_edit_association(step_records: List[Dict[str, Any]], horizon: int = 3) -> Dict[str, Any]:
    """Does a stale edit at step t precede an annotated incorrect step soon after?

    Outcome: any step in t+1..t+horizon is annotated incorrect, so the stale
    edit must come first; the incorrect step itself is not the predictor.
    Adjusted for the step's position in its trajectory and for framework, and
    with standard errors clustered by trajectory, since steps of one run are
    not independent. The earlier windowed odds ratio pooled steps across runs
    and tested concurrent exposure, not precedence.
    """
    df = pd.DataFrame(step_records)
    need = {"traj_id", "step_index", "total_steps", "stale_edit", "is_incorrect", "framework"}
    if len(df) < 20 or not need <= set(df.columns):
        return {"error": "Insufficient step records"}
    df = df.sort_values(["traj_id", "step_index"]).reset_index(drop=True)
    ahead = []
    for _, g in df.groupby("traj_id", sort=False):
        inc = g["is_incorrect"].astype(int).tolist()
        ahead += [int(any(inc[i + 1:i + 1 + horizon])) for i in range(len(inc))]
    df["error_ahead"] = ahead
    df["stale_edit"] = df["stale_edit"].astype(int)
    df["relative_step"] = df["step_index"] / df["total_steps"].clip(lower=1)
    X = pd.concat([df[["stale_edit", "relative_step"]],
                   pd.get_dummies(df["framework"], drop_first=True, prefix="fw", dtype=float)], axis=1)
    X = sm.add_constant(X)
    y = df["error_ahead"]
    out: Dict[str, Any] = {
        "horizon": horizon,
        "n_steps": int(len(df)),
        "n_trajectories": int(df["traj_id"].nunique()),
        "n_stale_edit_steps": int(df["stale_edit"].sum()),
        "error_ahead_after_stale": float(y[df["stale_edit"] == 1].mean()) if df["stale_edit"].any() else None,
        "error_ahead_otherwise": float(y[df["stale_edit"] == 0].mean()),
    }
    if y.nunique() < 2 or df["stale_edit"].nunique() < 2:
        out["error"] = "no variation"
        return out
    if y[df["stale_edit"] == 1].nunique() < 2:
        # Every stale edit is followed by an error, or none is: the likelihood
        # has no finite maximum, and a fitted odds ratio (0 or infinity) with
        # its Wald interval and p-value would be an artefact of the optimiser.
        out["error"] = "separation: the outcome is constant among stale-edit steps"
        return out
    groups = pd.factorize(df["traj_id"])[0]
    try:
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
        lo, hi = res.conf_int().loc["stale_edit"]
        out.update({
            "odds_ratio": float(np.exp(res.params["stale_edit"])),
            "ci_lower": float(np.exp(lo)),
            "ci_upper": float(np.exp(hi)),
            "p_value": float(res.pvalues["stale_edit"]),
        })
    except Exception as exc:  # separation, singular design
        out["error"] = str(exc)
    return out


def stratified_stale_edit_associations(step_records: List[Dict[str, Any]], horizon: int = 3) -> Dict[str, Any]:
    """`fit_stale_edit_association` refit two ways, for robustness.

    Without the frameworks that never stale-edit in these records: such a
    stratum carries no information about the odds ratio and can only
    destabilise the framework terms. And within each framework that does,
    where the framework adjustment drops out. Which frameworks are dropped is
    read from the records, never named by hand.
    """
    exposed = sorted({r["framework"] for r in step_records if r.get("stale_edit")})
    present = sorted({r["framework"] for r in step_records})
    dropped = [fw for fw in present if fw not in exposed]
    return {
        "dropped_zero_exposure": dropped,
        "without_zero_exposure": fit_stale_edit_association(
            [r for r in step_records if r["framework"] in exposed], horizon=horizon),
        "by_framework": {fw: fit_stale_edit_association(
            [r for r in step_records if r["framework"] == fw], horizon=horizon) for fw in exposed},
    }

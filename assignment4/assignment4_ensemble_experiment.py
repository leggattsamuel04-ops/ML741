import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assignment4_results_grouped"
OUT.mkdir(exist_ok=True)
SEED = 11
LABELS = np.arange(10)
SCORING = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy"}
SIZES = [20, 40, 60, 80, 100, 120, 160, 200]


def forest_preprocessor(columns):
    nominal = ["proto", "service", "state"]
    numerical = [c for c in columns if c not in nominal]
    return ColumnTransformer(
        [
            ("numeric", "passthrough", numerical),
            ("nominal", Pipeline([
                ("missing", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), nominal),
        ],
        sparse_threshold=1.0,
    )


def forest_pipeline(columns):
    return Pipeline([
        ("prep", forest_preprocessor(columns)),
        ("model", RandomForestClassifier(random_state=SEED, bootstrap=True, n_jobs=1)),
    ])


def revised_tree_pipeline(columns):
    return Pipeline([
        ("prep", forest_preprocessor(columns)),
        ("model", DecisionTreeClassifier(criterion="entropy", max_depth=30,
                                         min_samples_leaf=5, min_samples_split=4,
                                         class_weight=None, random_state=SEED)),
    ])


def score(y, pred):
    return {
        "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
    }


def save_search(search, filename):
    fields = ["params", "mean_train_macro_f1", "mean_test_macro_f1",
              "std_test_macro_f1", "mean_test_balanced_accuracy", "mean_test_accuracy",
              "rank_test_macro_f1"]
    table = pd.DataFrame(search.cv_results_)[fields].sort_values("rank_test_macro_f1")
    table.to_csv(OUT / filename, index=False)
    print(filename, "best", search.best_params_, "macro-F1", round(search.best_score_, 4), flush=True)
    return table


def oob_size_study(x_select, y_select, columns, fixed, sizes=SIZES):
    rows = []
    for trees in sizes:
        model = forest_pipeline(columns).set_params(
            **fixed, model__n_estimators=trees, model__oob_score=True)
        model.fit(x_select, y_select)
        probability = model.named_steps["model"].oob_decision_function_
        usable = probability.sum(axis=1) > 0
        pred = model.named_steps["model"].classes_[probability[usable].argmax(axis=1)]
        result = score(y_select.iloc[np.flatnonzero(usable)], pred)
        rows.append({"trees": trees, "covered_rows": int(usable.sum()), **result})
        print("OOB", trees, "macro-F1", round(result["macro_f1"], 4), flush=True)
    return pd.DataFrame(rows)


def main():
    evaluation_only = "--evaluation-only" in sys.argv
    result_out = OUT
    frame = pd.read_csv(ROOT / "networkTraffic.csv", na_values="?")
    mapping = pd.read_csv(ROOT / "attack_category_map.csv")
    names = mapping["Attack Category Name"].tolist()
    x_all = frame.drop(columns=["id", "attack_cat"])
    y_all = frame["attack_cat"].astype(int)
    _, pool_idx = train_test_split(frame.index, test_size=60_000,
                                   stratify=y_all, random_state=SEED)
    x_pool, y_pool = x_all.loc[pool_idx], y_all.loc[pool_idx]
    pool_groups = pd.util.hash_pandas_object(x_pool, index=False).to_numpy()
    outer = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
                 .split(x_pool, y_pool, pool_groups))
    eval_pos, select_pos = outer[1]
    x_select, y_select = x_pool.iloc[select_pos], y_pool.iloc[select_pos]
    x_eval, y_eval = x_pool.iloc[eval_pos], y_pool.iloc[eval_pos]
    select_groups, eval_groups = pool_groups[select_pos], pool_groups[eval_pos]
    assert not np.isin(np.unique(select_groups), np.unique(eval_groups)).any()
    split_audit = {
        "seed": SEED,
        "pool_rows": len(x_pool),
        "selection_rows": len(x_select),
        "evaluation_rows": len(x_eval),
        "selection_unique_vectors": int(pd.Series(select_groups).nunique()),
        "evaluation_unique_vectors": int(pd.Series(eval_groups).nunique()),
        "selection_evaluation_exact_vector_overlap": 0,
        "selection_class_counts": {names[i]: int((y_select == i).sum()) for i in LABELS},
        "evaluation_class_counts": {names[i]: int((y_eval == i).sum()) for i in LABELS},
        "exact_feature_target_duplicate_rows_full_data": int(frame.drop(columns="id").duplicated().sum()),
        "conflicting_label_feature_groups_full_data": int(
            frame.groupby(list(x_all.columns), dropna=False)["attack_cat"].nunique().gt(1).sum()),
    }
    (OUT / "split_audit.json").write_text(json.dumps(split_audit, indent=2), encoding="utf-8")
    print("Split audit", split_audit, flush=True)

    if evaluation_only:
        selected = json.loads((OUT / "selected_parameters.json").read_text(encoding="utf-8"))
    else:
        selected = tune_forest(x_all.columns, x_select, y_select, select_groups)

    models = {
        "Revised A2 tree": revised_tree_pipeline(x_all.columns),
        "Random forest": forest_pipeline(x_all.columns).set_params(**selected),
    }
    cv5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    predictions = {name: np.full(len(y_eval), -1, dtype=int) for name in models}
    for fold, (fit_pos, hold_pos) in enumerate(cv5.split(x_eval, y_eval, eval_groups), 1):
        x_fit, x_hold = x_eval.iloc[fit_pos], x_eval.iloc[hold_pos]
        y_fit, y_hold = y_eval.iloc[fit_pos], y_eval.iloc[hold_pos]
        assert not np.isin(np.unique(eval_groups[fit_pos]), np.unique(eval_groups[hold_pos])).any()
        for name, model in models.items():
            model.fit(x_fit, y_fit)
            train_pred, hold_pred = model.predict(x_fit), model.predict(x_hold)
            for partition, labels, pred in [("train", y_fit, train_pred),
                                            ("holdout", y_hold, hold_pred)]:
                rows.append({"fold": fold, "model": name, "partition": partition,
                             "rows": len(labels), "worms": int((labels == 7).sum()),
                             **score(labels, pred)})
                if partition == "holdout":
                    predictions[name][hold_pos] = pred
                    print("CV5", fold, name, "rows", len(labels), "macro-F1",
                          round(rows[-1]["macro_f1"], 4), flush=True)

    scores = pd.DataFrame(rows)
    scores.to_csv(result_out / "cv5_scores.csv", index=False)
    summary = scores.groupby(["model", "partition"])[
        ["macro_f1", "balanced_accuracy", "accuracy"]].agg(["mean", "std"])
    summary.to_csv(result_out / "cv5_summary.csv")
    print("CV5 summary", summary.round(4), flush=True)
    paired = scores.query("partition == 'holdout'").pivot(
        index="fold", columns="model", values="macro_f1")
    delta = paired["Random forest"] - paired["Revised A2 tree"]
    (result_out / "paired_difference.json").write_text(json.dumps({
        "fold_differences": delta.tolist(), "mean": float(delta.mean()),
        "sample_sd": float(delta.std(ddof=1))}, indent=2), encoding="utf-8")

    class_rows = []
    screening_rows = []
    for name, pred in predictions.items():
        assert (pred >= 0).all()
        report = classification_report(y_eval, pred, labels=LABELS, target_names=names,
                                       output_dict=True, zero_division=0)
        for category in names:
            class_rows.append({"model": name, "category": category, **report[category]})
        matrix = confusion_matrix(y_eval, pred, labels=LABELS)
        pd.DataFrame(matrix, index=names, columns=names).to_csv(
            result_out / f"confusion_{name.lower().replace(' ', '_')}.csv")
        normal = y_eval.to_numpy() == 0
        predicted_normal = pred == 0
        screening_rows.append({
            "model": name,
            "normal_correct": int((normal & predicted_normal).sum()),
            "normal_flagged_attack": int((normal & ~predicted_normal).sum()),
            "attack_missed_as_normal": int((~normal & predicted_normal).sum()),
            "attack_flagged": int((~normal & ~predicted_normal).sum()),
        })
    pd.DataFrame(class_rows).to_csv(result_out / "per_class.csv", index=False)
    pd.DataFrame(screening_rows).to_csv(result_out / "screening_errors.csv", index=False)
    print("Screening errors", pd.DataFrame(screening_rows), flush=True)


def tune_forest(columns, x_select, y_select, select_groups):
    cv3 = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    stage_a = GridSearchCV(
        forest_pipeline(columns),
        {"model__n_estimators": [80], "model__max_depth": [None, 16],
         "model__min_samples_leaf": [1, 5, 15],
         "model__max_features": ["sqrt", 0.5],
         "model__class_weight": [None, "balanced_subsample"]},
        cv=cv3, scoring=SCORING, refit="macro_f1", return_train_score=True,
        n_jobs=2, verbose=1,
    )
    stage_a.fit(x_select, y_select, groups=select_groups)
    save_search(stage_a, "stage_a_grouped.csv")

    fixed = {key: value for key, value in stage_a.best_params_.items()
             if key != "model__n_estimators"}
    stage_b = GridSearchCV(
        forest_pipeline(columns),
        {**{k: [v] for k, v in fixed.items()},
         "model__n_estimators": SIZES},
        cv=cv3, scoring=SCORING, refit="macro_f1", return_train_score=True,
        n_jobs=2, verbose=1,
    )
    stage_b.fit(x_select, y_select, groups=select_groups)
    size_table = save_search(stage_b, "stage_b_grouped.csv")
    selected = dict(stage_b.best_params_)
    (OUT / "selected_parameters.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    oob = oob_size_study(x_select, y_select, columns, fixed)
    oob.to_csv(OUT / "oob_size.csv", index=False)
    size_plot = size_table.copy()
    size_plot["trees"] = size_plot.params.map(lambda p: p["model__n_estimators"])
    size_plot = size_plot.sort_values("trees")
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    ax.plot(size_plot.trees, 1 - size_plot.mean_test_macro_f1, marker="o",
            label="Grouped CV3")
    ax.plot(oob.trees, 1 - oob.macro_f1, marker="s", linestyle="--",
            label="Ordinary OOB")
    ax.axvline(selected["model__n_estimators"], color="0.4", linestyle=":",
               label=f"Best tested: {selected['model__n_estimators']} trees")
    ax.set(xlabel="Number of trees", ylabel="1 - macro-F1 (lower is better)",
           xticks=SIZES)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ensemble_size.png", dpi=200)
    plt.close(fig)

    return selected


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.tree import DecisionTreeClassifier


SEED = 11
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assignment4_results"
OUT.mkdir(exist_ok=True)
SCORING = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy"}


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


def assignment2_tree_preprocessor(columns):
    nominal = ["proto", "service", "state"]
    numerical = [c for c in columns if c not in nominal]
    return ColumnTransformer(
        [
            ("num", "passthrough", numerical),
            ("state", OneHotEncoder(handle_unknown="ignore"), ["state"]),
            ("proto", TargetEncoder(target_type="multiclass", smooth="auto", cv=5,
                                    shuffle=True, random_state=SEED), ["proto"]),
            ("service", Pipeline([
                ("missing", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), ["service"]),
        ]
    )


def forest_pipeline(columns):
    return Pipeline([
        ("prep", forest_preprocessor(columns)),
        ("model", RandomForestClassifier(random_state=SEED, n_jobs=1, bootstrap=True)),
    ])


def save_search(search, name):
    table = pd.DataFrame(search.cv_results_)
    columns = ["params", "mean_train_macro_f1", "mean_test_macro_f1",
               "std_test_macro_f1", "mean_test_balanced_accuracy", "mean_test_accuracy",
               "rank_test_macro_f1"]
    table[columns].sort_values("rank_test_macro_f1").to_csv(OUT / f"{name}.csv", index=False)
    print(name, "best", search.best_params_, "macro-F1", round(search.best_score_, 4), flush=True)
    return table


def score_rows(y, predictions, fold, model, partition):
    return {"model": model, "fold": fold, "partition": partition,
            "macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, predictions),
            "accuracy": accuracy_score(y, predictions)}


def main(reuse_forest_search=False):
    frame = pd.read_csv(ROOT / "networkTraffic.csv", na_values="?")
    mapping = pd.read_csv(ROOT / "attack_category_map.csv")
    labels = mapping["Mapping"].to_list()
    names = mapping["Attack Category Name"].to_list()
    y = frame["attack_cat"].astype(int)
    X = frame.drop(columns=["id", "attack_cat"])
    audit = {"rows": len(frame), "predictors_including_id": frame.shape[1] - 1,
             "missing": frame.isna().sum().loc[lambda s: s.gt(0)].to_dict(),
             "class_counts": {names[i]: int((y == labels[i]).sum()) for i in range(len(labels))},
             "nominal_cardinality": {c: int(frame[c].nunique()) for c in ["proto", "state", "service"]},
             "duplicate_rows_excluding_id": int(frame.drop(columns="id").duplicated().sum()),
             "conflicting_label_feature_groups": int(frame.groupby(list(X.columns), dropna=False)["attack_cat"].nunique().gt(1).sum())}
    (OUT / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Data audit", audit, flush=True)

    select_idx, eval_idx = train_test_split(
        X.index, train_size=20_000, test_size=40_000, stratify=y, random_state=SEED)
    X_select, y_select = X.loc[select_idx], y.loc[select_idx]
    X_eval, y_eval = X.loc[eval_idx], y.loc[eval_idx]
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    if reuse_forest_search:
        selected = json.loads((OUT / "selected_parameters.json").read_text(encoding="utf-8"))
    else:
        # fixed ensemble size but varying other parameters
        stage_a = GridSearchCV(
            forest_pipeline(X.columns),
            {"model__n_estimators": [80], "model__max_depth": [16, None],
             "model__min_samples_leaf": [1, 5, 15],
             "model__max_features": ["sqrt", 0.5],
             "model__class_weight": [None, "balanced_subsample"]},
            cv=cv3, scoring=SCORING, refit="macro_f1", return_train_score=True,
            n_jobs=2, verbose=1,
        )
        stage_a.fit(X_select, y_select)
        save_search(stage_a, "stage_a_capacity_decorrelation_weighting")

        # varied ensemble size over best other params.
        fixed = {key: [value] for key, value in stage_a.best_params_.items()
                 if key != "model__n_estimators"}
        stage_b = GridSearchCV(
            forest_pipeline(X.columns),
            {**fixed, "model__n_estimators": [40, 120, 240]},
            cv=cv3, scoring=SCORING, refit="macro_f1", return_train_score=True,
            n_jobs=2, verbose=1,
        )
        stage_b.fit(X_select, y_select)
        save_search(stage_b, "stage_b_forest_size")
        selected = dict(stage_b.best_params_)
        if stage_a.best_score_ > stage_b.best_score_:
            selected = dict(stage_a.best_params_)
        (OUT / "selected_parameters.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    original_tree = Pipeline([
        ("prep", assignment2_tree_preprocessor(X.columns)),
        ("model", DecisionTreeClassifier(random_state=SEED, criterion="entropy",
                                         max_depth=30, min_samples_leaf=5,
                                         min_samples_split=4, class_weight=None)),
    ])

    models = {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "Assignment 2 tree": original_tree,
        "Random forest": forest_pipeline(X.columns).set_params(**selected),
    }
    rows = []
    predictions = {name: np.empty(len(y_eval), dtype=int) for name in models}
    feature_importances = []
    for fold, (train_pos, test_pos) in enumerate(cv5.split(X_eval, y_eval), start=1):
        x_train, x_test = X_eval.iloc[train_pos], X_eval.iloc[test_pos]
        y_train, y_test = y_eval.iloc[train_pos], y_eval.iloc[test_pos]
        for name, model in models.items():
            if name == "Assignment 2 tree":
                prep = model.named_steps["prep"]
                encoded_train = prep.fit_transform(x_train, y_train)
                encoded_test = prep.transform(x_test)
                tree = model.named_steps["model"].fit(encoded_train, y_train)
                train_pred = tree.predict(encoded_train)
                test_pred = tree.predict(encoded_test)
            else:
                model.fit(x_train, y_train)
                train_pred = model.predict(x_train)
                test_pred = model.predict(x_test)
            rows.append(score_rows(y_train, train_pred, fold, name, "train"))
            rows.append(score_rows(y_test, test_pred, fold, name, "holdout"))
            predictions[name][test_pos] = test_pred
            print(f"CV5 {fold} {name}: holdout macro-F1 {rows[-1]['macro_f1']:.4f}", flush=True)
            if name == "Random forest":
                features = model.named_steps["prep"].get_feature_names_out()
                for feature, importance in zip(features, model.named_steps["model"].feature_importances_):
                    feature_importances.append({"fold": fold, "encoded_feature": feature,
                                                "importance": importance})

    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "cv5_fold_scores.csv", index=False)
    summary = folds.groupby(["model", "partition"])[["macro_f1", "balanced_accuracy", "accuracy"]].agg(["mean", "std"])
    summary.to_csv(OUT / "cv5_summary.csv")
    print("CV5 summary\n", summary.round(4).to_string(), flush=True)

    class_tables = []
    for name, pred in predictions.items():
        report = classification_report(y_eval, pred, labels=labels, target_names=names,
                                       zero_division=0, output_dict=True)
        table = pd.DataFrame(report).T
        table.index.name = "class"
        table.to_csv(OUT / f"per_class_{name.lower().replace(' ', '_')}.csv")
        matrix = confusion_matrix(y_eval, pred, labels=labels)
        pd.DataFrame(matrix, index=names, columns=names).to_csv(
            OUT / f"confusion_{name.lower().replace(' ', '_')}.csv")
        for class_name in names:
            class_tables.append({"model": name, "class": class_name, **report[class_name]})
    pd.DataFrame(class_tables).to_csv(OUT / "per_class_comparison.csv", index=False)

    forest_cm = confusion_matrix(y_eval, predictions["Random forest"], labels=labels,
                                 normalize="true")
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(forest_cm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    fig.colorbar(image, ax=ax, label="Within-class proportion")
    fig.tight_layout()
    fig.savefig(OUT / "forest_confusion.png", dpi=180)
    plt.close(fig)

    holdout = folds.query("partition == 'holdout'").pivot(index="fold", columns="model", values="macro_f1")
    paired = holdout["Random forest"] - holdout["Assignment 2 tree"]
    (OUT / "paired_tree_difference.json").write_text(json.dumps(
        {"fold_differences": paired.tolist(), "mean": float(paired.mean()),
         "sample_sd": float(paired.std(ddof=1))}, indent=2), encoding="utf-8")
    pd.DataFrame(feature_importances).to_csv(OUT / "forest_encoded_importances.csv", index=False)
    print("Paired forest minus tree macro-F1", paired.round(4).tolist(), flush=True)


if __name__ == "__main__":
    main()

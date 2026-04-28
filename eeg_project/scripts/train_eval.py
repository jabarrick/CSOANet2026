from typing import Dict, Tuple
import numpy as np


def _make_pipeline(name: str):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    if name == "LDA":
        clf = LinearDiscriminantAnalysis()
    elif name == "SVM":
        clf = SVC(kernel="rbf", C=1.0)
    elif name == "RF":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {name}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def train_classifiers(X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """训练并交叉验证多个分类器（含标准化 Pipeline），返回得分统计。"""
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    names = ["LDA", "SVM", "RF"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for name in names:
        pipe = _make_pipeline(name)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        results[name] = {"scores": scores, "mean": scores.mean(), "std": scores.std()}
    return results


def evaluate_best_classifier(
    X: np.ndarray, y: np.ndarray, results: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[object, np.ndarray, str, str]:
    """基于 CV 结果选择最佳模型，做独立 holdout 评估并返回模型/混淆矩阵/报告/名称。"""
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split

    best_name = max(results.keys(), key=lambda k: results[k]["mean"])
    pipe = _make_pipeline(best_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return pipe, cm, report, best_name



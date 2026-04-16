import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Machine Learning Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Logging configuration specifically for the report output
logging.basicConfig(level=logging.INFO, format='%(message)s')


def plot_model_comparison(results: dict, output_dir: str):
    """
    Generates and saves a grouped bar chart comparing all classifiers
    across Accuracy, Precision, Recall, F1-Score, and AUC.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    # Shorten model names for the legend
    short_names = {
        "K-Nearest Neighbors": "KNN",
        "Naive Bayes": "Naive Bayes",
        "Logistic Regression": "Logistic Reg.",
        "Random Forest": "Random Forest",
        "Gradient Boosting": "Grad. Boosting",
    }

    values = np.array([
        [results[m][metric] if results[m][metric] is not None else 0 for metric in metrics]
        for m in model_names
    ])

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f9f9f9")

    x = np.arange(n_metrics)
    bar_width = 0.14
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

    colors = ["#4A90D9", "#E8734A", "#5CB85C", "#9B59B6", "#F0AD4E"]

    for i, (model, offset) in enumerate(zip(model_names, offsets)):
        bars = ax.bar(
            x + offset,
            values[i],
            width=bar_width,
            label=short_names.get(model, model),
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.004,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#444444",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — CVD Prediction", fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"\nPlot saved to: {output_path}")


def execute_models(train_path: str, test_path: str):
    """
    Loads preprocessed data, builds the preprocessing pipeline,
    and evaluates 5 classifiers returning a comprehensive metric report.
    """
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logging.error("Preprocessed CSVs not found. Please run preprocess_data.py first.")
        return

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('cardio', axis=1)
    y_train = train_df['cardio']
    X_test = test_df.drop('cardio', axis=1)
    y_test = test_df['cardio']

    # 1. Feature Engineering: Column Transformer
    # Ensures no data leakage by fitting only on the training set during pipeline execution
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), ['gender']),
        (StandardScaler(), ['age', 'height', 'weight', 'ap_hi', 'ap_lo']),
        remainder='passthrough'
    )

    # 2. Optimized Hyperparameters Configuration
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(
            leaf_size=10,
            n_neighbors=67,
            metric='euclidean',
            weights='distance'
        ),
        "Naive Bayes": GaussianNB(
            var_smoothing=0.2848035868
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=10000,
            C=0.025,
            solver='saga',
            class_weight='balanced',
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=900,
            max_depth=30,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=600,
            max_depth=20,
            random_state=42
        )
    }

    # Print Report Header
    logging.info(f"Model Evaluation on {len(X_train)} training samples...\n")
    logging.info(f"{'Classifier Name':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'AUC':<6}")
    logging.info("-" * 85)

    # 3. Training and Evaluation Loop
    results = {}
    for name, clf in models.items():
        pipeline = make_pipeline(preprocessor, clf)

        # Fit on training data
        pipeline.fit(X_train, y_train)

        # Predict on testing data
        y_pred = pipeline.predict(X_test)

        # Calculate standard metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Calculate AUC if the model supports probability estimation
        auc = None
        if hasattr(clf, "predict_proba"):
            auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
            auc_str = f"{auc:.4f}"
        else:
            auc_str = "N/A"

        # Log formatted row
        logging.info(f"{name:<22} | {acc:.4f}   | {prec:.4f}    | {rec:.4f}   | {f1:.4f}   | {auc_str}")

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "AUC": auc,
        }

    # 4. Generate comparison plot
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "../outputs")
    plot_model_comparison(results, output_dir)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(base_dir, "../data/train_final.csv")
    test_csv = os.path.join(base_dir, "../data/test_final.csv")

    execute_models(train_csv, test_csv)

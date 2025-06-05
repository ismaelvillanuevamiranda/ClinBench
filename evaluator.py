# evaluator.py

import argparse
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

class Evaluator:
    def __init__(self,
                 gt_path: str,
                 predictions_dir: str,
                 output_dir: str,
                 columns: list[str],
                 transforms: dict[str, callable] = None,
                 label_orders: dict[str, list] = None,
                 plot_cm: bool = False):
        """
        Initialize the Evaluator.

        Args:
            gt_path:        Path to the ground-truth CSV file.
            predictions_dir:Directory containing model prediction CSVs.
            output_dir:     Directory where per-model and summary CSVs will be saved.
            columns:        List of feature/column names to evaluate.
            transforms:     Optional mapping of feature->function to normalize values.
            label_orders:   Optional mapping of feature->list specifying CM axis order.
            plot_cm:        If True, generate and save confusion matrix plots.
        """
        self.gt_path = gt_path
        self.pred_dir = predictions_dir
        self.out_dir = output_dir
        self.features = columns
        self.transforms = transforms or {}
        self.label_orders = label_orders or {}
        self.plot_cm = plot_cm

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load and transform ground truth once (suffix "_gold" applied in transform)
        self.gt = self._load_and_transform(gt_path, suffix="_gold")

    def _load_and_transform(self, path, suffix=""):
        """
        Load a CSV into a DataFrame and apply any specified transforms.

        Args:
            path:   Path to CSV file.
            suffix: If provided, appended to column names before transform lookup.
        Returns:
            Transformed pandas.DataFrame.
        """
        df = pd.read_csv(path)

        # Apply each transform function to its corresponding column
        for col, fn in self.transforms.items():
            target = col + suffix if suffix else col
            if target in df:
                df[target] = df[target].apply(fn)

        return df

    def _compute_metrics(self, y_true, y_pred, labels=None):
        """
        Compute standard classification metrics plus specificity and coverage.

        Args:
            y_true:   Series of true labels.
            y_pred:   Series of predicted labels.
            labels:   Specific ordering of labels for the confusion matrix.
        Returns:
            (metrics_dict, confusion_matrix, cm_label_list)
        """
        # Determine label set/order for the confusion matrix
        cm_labels = labels or sorted(set(y_true.unique()) | set(y_pred.unique()))
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

        # Compute true/false positives/negatives per class
        tn = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + cm.diagonal()
        fp = cm.sum(axis=0) - cm.diagonal()
        fn = cm.sum(axis=1) - cm.diagonal()
        tp = cm.diagonal()

        # Specificity = TN / (TN + FP) averaged across classes
        specificity = (tn / (tn + fp + 1e-10)).mean()

        # Gather metrics in a dict
        metrics = {
            "Accuracy":    accuracy_score(y_true, y_pred),
            "Precision":   precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall":      recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1":          f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "Specificity": specificity,
            "Coverage":    len(y_pred) / len(y_true)  # fraction of predictions made
        }
        return metrics, cm, cm_labels

    def _plot_cm(self, cm, labels, feature, model_name):
        """
        Plot and save a confusion matrix heatmap.

        Args:
            cm:           Confusion matrix array.
            labels:       Label ordering for axes.
            feature:      Feature name (for title/filename).
            model_name:   Model identifier (for title/filename).
        """
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"{model_name} — {feature}")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Save plot to output directory
        fpath = os.path.join(self.out_dir, f"cm_{model_name}_{feature}.png")
        plt.tight_layout()
        plt.savefig(fpath, dpi=200)
        plt.close()

    def evaluate_all(self):
        """
        Main loop: for each prediction CSV, compute metrics per feature,
        optionally plot confusion matrices, and write results.
        """
        summary_rows = []

        # Find all .csv files in prediction directory
        for csv in glob.glob(os.path.join(self.pred_dir, "*.csv")):
            model_name = os.path.splitext(os.path.basename(csv))[0]

            # Load and transform predictions (no suffix)
            pred = self._load_and_transform(csv, suffix="")

            results = {}
            # Evaluate each requested feature/column
            for feat in self.features:
                # Determine ground-truth column (with "_gold" suffix if present)
                gt_col = feat + "_gold" if feat + "_gold" in self.gt else feat

                # Skip if missing in either GT or predictions
                if gt_col not in self.gt or feat not in pred:
                    print(f"Skipping {feat} for {model_name} (missing column)")
                    continue

                y_true = self.gt[gt_col]
                y_pred = pred[feat]

                # Compute metrics and confusion matrix
                metrics, cm, labels = self._compute_metrics(
                    y_true, y_pred, self.label_orders.get(feat)
                )

                # Flatten metrics into results dict
                for m, v in metrics.items():
                    results[f"{feat}.{m}"] = round(v, 4)

                # Optionally generate confusion matrix plot
                if self.plot_cm:
                    self._plot_cm(cm, labels, feat, model_name)

            # Save per-model results to CSV
            df_res = pd.DataFrame([results])
            df_res.insert(0, "Model", model_name)
            out_csv = os.path.join(self.out_dir, f"{model_name}_results.csv")
            df_res.to_csv(out_csv, index=False)
            summary_rows.append(df_res)

        # Concatenate all per-model results into a final summary CSV
        if summary_rows:
            summary = pd.concat(summary_rows, ignore_index=True)
            summary.to_csv(os.path.join(self.out_dir, "final_summary.csv"), index=False)

        print("Done. Results in:", self.out_dir)


if __name__ == "__main__":
    # Parse command-line arguments
    p = argparse.ArgumentParser()
    p.add_argument("--gt",        required=True, help="ground truth CSV")
    p.add_argument("--preds_dir", required=True, help="folder with prediction CSVs")
    p.add_argument("--out_dir",   required=True, help="where to save results")
    p.add_argument("--features",  required=True,
                   help="comma-sep list of columns to eval")
    p.add_argument("--plot_cm",   action="store_true",
                   help="whether to save confusion matrix plots")
    args = p.parse_args()

    from lung_transforms import (
        transform_pT,
        transform_pN,
        transform_stage_value,
        transform_histological_diagnosis
    )
    transforms = {
        "pT":                   transform_pT,
        "pN":                   transform_pN,
        "tumor_stage":          transform_stage_value,
        "histologic_diagnosis": transform_histological_diagnosis,
    }

    label_orders = {
        "pT": ["T1","T2","T3","T4","Unknown"],
        "pN": ["n0","n1","n2","n3","Unknown"],
    }

    # Instantiate and run the evaluator
    evalr = Evaluator(
        gt_path=args.gt,
        predictions_dir=args.preds_dir,
        output_dir=args.out_dir,
        columns=args.features.split(","),
        transforms=transforms,
        label_orders=label_orders,
        plot_cm=args.plot_cm,
    )
    evalr.evaluate_all()

    # Usage from the command-line:
    # python evaluator.py \
    #   --gt /path/to/ground_truth.csv \
    #   --preds_dir /path/to/predictions_folder \
    #   --out_dir /path/to/eval_output \
    #   --features pT,pN,tumor_stage,histologic_diagnosis \
    #   --plot_cm

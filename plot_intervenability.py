import os

import matplotlib.pyplot as plt
import pandas as pd

NAME_STANDARD_MODEL = "PGCM"

# Global plotting style. Tune these values to increase/decrease all visual elements.
PLOT_STYLE = {
    "base_font_size": 20,
    "label_font_size": 22,
    "title_font_size": 26,
    "tick_font_size": 20,
    "legend_font_size": 18,
    "line_width": 6,
    "marker_size": 14,
    "fill_alpha": 0.2,
    "grid_alpha": 0.3,
    "figure_width_per_panel": 6,
    "figure_height": 5,
    "legend_ncol_max": 6,
    "bottom_legend_space": 0.08,
    "pdf_dpi": 300,
}

MODEL_STYLE = {
    "CRM": {"color": "tab:blue", "linestyle": "-", "zorder": 2},
    "CBM": {"color": "tab:orange", "linestyle": "-", "zorder": 3},
    "CMR": {"color": "tab:green", "linestyle": "-", "zorder": 4},
    "PGCM": {"color": "tab:red", "linestyle": "-", "zorder": 5},
    "PGCM*": {"color": "tab:purple", "linestyle": "--", "zorder": 6},
}


def apply_global_plot_style():
    plt.rcParams.update(
        {
            "font.size": PLOT_STYLE["base_font_size"],
            "axes.labelsize": PLOT_STYLE["label_font_size"],
            "axes.titlesize": PLOT_STYLE["title_font_size"],
            "xtick.labelsize": PLOT_STYLE["tick_font_size"],
            "ytick.labelsize": PLOT_STYLE["tick_font_size"],
            "legend.fontsize": PLOT_STYLE["legend_font_size"],
        }
    )

# Dataset -> model -> seed output folders.
# Each folder should contain: epoch_final_test/intervenability_results.csv
# For PGCM (without '*') we use intervenability_results_standard_pgcm.csv.
DATASET_MODEL_PATHS = {
    "ColorMNIST+": {
        "CRM": [
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CRM_SEED_1_20260124_161855",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CRM_SEED_2_20260124_161855",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CRM_SEED_3_20260124_161855",
        ],
        "CBM": [
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CBM_SEED_1_20260124_150958",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CBM_SEED_2_20260124_151750",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CBM_SEED_3_20260124_152001",
        ],
        "CMR": [
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CMR_SEED_1_20260124_162206",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CMR_SEED_2_20260124_162505",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CMR_SEED_3_20260124_162753",
        ],
        NAME_STANDARD_MODEL: [
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/outputs/run_MNIST_PLOTSTANDARD_SEED1",
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/outputs/run_MNIST_PLOTSTANDARD_SEED2",
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/outputs/run_MNIST_PLOTSTANDARD_SEED3",
        ],
        "PGCM*": [
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/outputs/run_MNIST_PLOTSTANDARD_SEED1",
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/outputs/run_MNIST_PLOTSTANDARD_SEED2",
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/outputs/run_MNIST_PLOTSTANDARD_SEED3",
        ],
    },
    "CUB": {
        "CRM": [
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/new_outputs/20260424_110632_cubEMB_crm_cub_new_1/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/new_outputs/20260424_105041_cubEMB_crm_cub_new_2/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/new_outputs/20260424_105913_cubEMB_crm_cub_new_3/outputs",
        ],
        "CBM": [
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/new_outputs/20260424_095929_cubEMB_cbm_cub_new_1/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/new_outputs/20260424_103159_cubEMB_cbm_cub_new_2/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/new_outputs/20260424_103917_cubEMB_cbm_cub_new_3/outputs",
        ],
        "CMR": [
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/new_outputs/20260424_114431_cubEMB_cmr_cub_new_1/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/new_outputs/20260424_111755_cubEMB_cmr_cub_new_2/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/new_outputs/20260424_113126_cubEMB_cmr_cub_new_3/outputs",
        ],
        NAME_STANDARD_MODEL: [
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260423_143758_cubEMB_realcosine25_s1/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260423_145805_cubEMB_realcosine25_s2/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260423_151440_cubEMB_realcosine25_s3/outputs",
        ],
        "PGCM*": [
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260423_143758_cubEMB_realcosine25_s1/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260423_145805_cubEMB_realcosine25_s2/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260423_151440_cubEMB_realcosine25_s3/outputs",
        ],
    },
    "CelebA": {
        "CRM": [
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CRM_CELEBA_SEED_1_20260124_163114",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CRM_CELEBA_SEED_2_20260124_163905",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CRM_CELEBA_SEED_3_20260124_164833",
        ],
        "CBM": [
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CBM_CELEBA_SEED_1_20260124_153710",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CBM_CELEBA_SEED_2_20260124_155405",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CBM_CELEBA_SEED_3_20260124_160534",
        ],
        "CMR": [
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CMR_CELEBA_SEED_1_20260124_180239",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CMR_CELEBA_SEED_2_20260124_201151",
            "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/run_CMR_CELEBA_SEED_3_20260124_202927",
        ],
        # NAME_STANDARD_MODEL: [  # CONCEPT
        #     "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260312_132537_celebamask_moreprotos_1/outputs",
        #     "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260416_144341_celebamask_moreprotos_2/outputs",
        #     "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260417_092136_celebamask_moreprotos_3/outputs",
        # ],
        # "PGCM*": [
        #     "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260312_132537_celebamask_moreprotos_1/outputs",
        #     "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260416_144341_celebamask_moreprotos_2/outputs",
        #     "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260417_092136_celebamask_moreprotos_3/outputs",
        # ],
        NAME_STANDARD_MODEL: [  # TASK
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260417_115658_celebamask_testunbalanced/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260417_120409_celebamask_testunbalanced/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260424_154552_celebamask_testunbalanced/outputs",
        ],
        "PGCM*": [
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260417_115658_celebamask_testunbalanced/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260417_120409_celebamask_testunbalanced/outputs",
            "/cw/dtaijupiter/NoCsBack/dtai/stefano/repos/higherordercbms/new_outputs/20260424_154552_celebamask_testunbalanced/outputs",
        ],
    },
}


def load_intervenability_data(base_path, use_standard_intervention=False):
    if use_standard_intervention:
        csv_path = os.path.join(base_path, "epoch_final_test", "intervenability_results_standard_pgcm.csv")
    else:
        csv_path = os.path.join(base_path, "epoch_final_test", "intervenability_results.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    return pd.read_csv(csv_path)


def aggregate_model_data(paths, use_standard_intervention=False):
    all_data = []

    for path in paths:
        try:
            all_data.append(load_intervenability_data(path, use_standard_intervention=use_standard_intervention))
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    grouped = combined.groupby("nb_interventions").agg(
        {
            "c_accuracies": ["mean", "std"],
            "y_accuracies": ["mean", "std"],
        }
    ).reset_index()

    # Avoid NaN std when only one run is available.
    grouped[("c_accuracies", "std")] = grouped[("c_accuracies", "std")].fillna(0.0)
    grouped[("y_accuracies", "std")] = grouped[("y_accuracies", "std")].fillna(0.0)
    return grouped


def _resolve_model_paths(model_paths, metric_name):
    resolved = {}
    for model_name, paths in model_paths.items():
        if isinstance(paths, dict):
            resolved[model_name] = paths[metric_name]
        else:
            resolved[model_name] = paths
    return resolved


def collect_all_dataset_data(dataset_model_paths):
    all_data = {}
    for dataset_name, model_paths in dataset_model_paths.items():
        dataset_data = {}
        print(f"\n=== Dataset: {dataset_name} ===")
        for model_name, paths in model_paths.items():
            print(f"Processing {dataset_name} / {model_name}...")
            data = aggregate_model_data(
                paths,
                use_standard_intervention=("pgcm" in model_name.lower() and "*" not in model_name.lower()),
            )
            if data is None:
                print("  No valid data found")
            else:
                dataset_data[model_name] = data
                print(f"  Loaded {len(paths)} configured seed path(s)")
        all_data[dataset_name] = dataset_data
    return all_data


def _plot_metric_panel(ax, dataset_name, dataset_data, metric_col, ylabel):
    if not dataset_data:
        ax.text(0.5, 0.5, f"No data for {dataset_name}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(dataset_name)
        ax.grid(True, alpha=PLOT_STYLE["grid_alpha"])
        return

    for model_name, data in dataset_data.items():
        x = data["nb_interventions"]
        y_mean = data[(metric_col, "mean")]
        y_std = data[(metric_col, "std")]

        style = MODEL_STYLE.get(model_name, {})
        color = style.get("color", None)
        linestyle = style.get("linestyle", "-")
        zorder = style.get("zorder", 1)

        ax.plot(
            x,
            y_mean,
            marker="o",
            label=model_name,
            linewidth=PLOT_STYLE["line_width"],
            markersize=PLOT_STYLE["marker_size"],
            linestyle=linestyle,
            color=color,
            zorder=zorder,
        )
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=PLOT_STYLE["fill_alpha"], color=color)

    ax.set_title(dataset_name)
    ax.set_xlabel("Number of Concept Interventions")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=PLOT_STYLE["grid_alpha"])


def plot_figure_across_datasets(
    all_dataset_data,
    metric_col,
    ylabel,
    output_pdf,
):
    dataset_names = list(all_dataset_data.keys())
    n = len(dataset_names)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(PLOT_STYLE["figure_width_per_panel"] * n, PLOT_STYLE["figure_height"]),
        squeeze=False,
    )

    for i, dataset_name in enumerate(dataset_names):
        ax = axes[0, i]
        _plot_metric_panel(
            ax,
            dataset_name,
            all_dataset_data[dataset_name],
            metric_col=metric_col,
            ylabel=ylabel,
        )

        ax.tick_params(axis="both", which="major", labelsize=PLOT_STYLE["tick_font_size"])
        ax.xaxis.label.set_size(PLOT_STYLE["label_font_size"])
        ax.yaxis.label.set_size(PLOT_STYLE["label_font_size"])
        ax.title.set_size(PLOT_STYLE["title_font_size"])

    handles, labels = [], []
    for ax in axes.flatten():
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), PLOT_STYLE["legend_ncol_max"]),
            fontsize=PLOT_STYLE["legend_font_size"],
            frameon=False,
        )

    plt.tight_layout(rect=[0, PLOT_STYLE["bottom_legend_space"], 1, 1])
    plt.savefig(output_pdf, dpi=PLOT_STYLE["pdf_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_pdf}")


def main():
    apply_global_plot_style()
    all_dataset_data = collect_all_dataset_data(DATASET_MODEL_PATHS)

    plot_figure_across_datasets(
        all_dataset_data,
        metric_col="c_accuracies",
        ylabel="Concept Accuracy",
        output_pdf="concept_accuracy_all_datasets.pdf",
    )

    plot_figure_across_datasets(
        all_dataset_data,
        metric_col="y_accuracies",
        ylabel="Task Accuracy",
        output_pdf="task_accuracy_all_datasets.pdf",
    )


if __name__ == "__main__":
    main()

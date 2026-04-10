import numpy as np
import matplotlib.pyplot as plt


def get_latex_figsize(width_scale=1.0, height_scale=None):
    """
    Calculates the figure size in inches for a LaTeX document based on its
    standard text width.
    """
    doc_textwidth_mm = 117.0
    inches_per_mm = 1 / 25.4
    doc_textwidth_in = doc_textwidth_mm * inches_per_mm

    fig_width = doc_textwidth_in * width_scale

    if height_scale is None:
        # Use the golden ratio for a pleasing aspect ratio
        golden_ratio = (np.sqrt(5) - 1.0) / 2.0
        fig_height = fig_width * golden_ratio
    else:
        fig_height = doc_textwidth_in * height_scale

    return {"width": fig_width, "height": fig_height}


def apply_publication_style(ax, title, xlabel, ylabel):
    """
    Applies the specific, consolidated styling to a Matplotlib Axes object.

    Args:
        ax (matplotlib.axes.Axes): The axes object to style.
        title (str): The title for the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    # --- Fonts and Text Sizes ---
    # ax.set_title(title, fontsize=10) # NO TITLE FOR FIGURES
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)

    # --- Grid and Ticks ---
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    # --- Legend Styling ---
    ax.legend(
        loc="lower right",
        fontsize=6,
        borderpad=0.4,  # Padding inside the legend box
        labelspacing=0.5,  # Vertical space between entries
        handletextpad=0.5,  # Space between the line/marker and the text
    )


# --- Main script to generate an example plot ---
if __name__ == "__main__":

    # 1. Get figure dimensions
    figsize = get_latex_figsize(width_scale=1.0)  # Use 100% of text width

    # 2. Create figure and axes
    fig, ax = plt.subplots(figsize=(figsize["width"], figsize["height"]))

    # 3. Create sample data
    episodes = np.arange(0, 500)
    # Simulate mean and std for two datasets
    mean1 = 450 * np.exp(-episodes / 150) + 50
    std1 = 30 * np.exp(-episodes / 200) + 20
    mean2 = 400 * np.exp(-episodes / 250) + 100
    std2 = 40 * np.exp(-episodes / 300) + 25

    # 4. Plot the data using the specific color scheme
    # --- Dataset 1 (Equivalent to "Baseline") ---
    ax.plot(episodes, mean1, label="Dataset 1", color="dodgerblue")
    ax.fill_between(episodes, mean1 - std1, mean1 + std1, color="skyblue", alpha=0.2)
    ax.axvline(
        x=180, color="dodgerblue", linestyle="--", label="Dataset 1 Avg Solve (180)"
    )

    # --- Dataset 2 (Equivalent to "Noise") ---
    ax.plot(episodes, mean2, label="Dataset 2", color="darkorange")
    ax.fill_between(episodes, mean2 - std2, mean2 + std2, color="sandybrown", alpha=0.2)
    ax.axvline(
        x=250, color="darkorange", linestyle="--", label="Dataset 2 Avg Solve (250)"
    )

    # 5. Apply the consolidated styling settings
    apply_publication_style(
        ax=ax, title="Recreated Plot Style", xlabel="Episode", ylabel="Duration"
    )

    # 6. Final adjustments and saving
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)

    # Use bbox_inches='tight' to remove extra whitespace
    plt.savefig("recreated_plot.pdf", bbox_inches="tight")

    print("Plot saved to 'recreated_plot.pdf'")
    plt.show()

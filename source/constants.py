import os
import seaborn as sns
import matplotlib.pyplot as plt


PROJECT_DIRECTORY_PATH = os.path.dirname(os.path.dirname(__file__))

deep_palette = sns.color_palette("deep")

TRIAGE_MAPPING = {
    "A": "Critical",
    "H": "High Priority",
    "V1": "Moderate Priority",
    "V2": "Scheduled"
}

COLOR_MAPPING_DEEP = {
    "Critical": deep_palette[3],
    "High Priority": deep_palette[1],
    "Moderate Priority": deep_palette[8],
    "Scheduled": deep_palette[2]
}

COLOR_MAPPING_LIGHT = [plt.cm.colors.to_rgba(color, alpha=0.65) for color in COLOR_MAPPING_DEEP.values()]

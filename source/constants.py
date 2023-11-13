import os
import seaborn as sns
import matplotlib.pyplot as plt


PROJECT_DIRECTORY_PATH = os.path.dirname(os.path.dirname(__file__))

deep_palette = sns.color_palette("deep")

TRIAGE_MAPPING = {
    "A": "A",
    "H": "H",
    "V1": "V1",
    "V2": "V2"
}

COLOR_MAPPING_DEEP = {
    "A": deep_palette[3],
    "U": deep_palette[1],
    "V1": deep_palette[8],
    "V2": deep_palette[2]
}

COLOR_MAPPING_LIGHT = [plt.cm.colors.to_rgba(color, alpha=0.65) for color in COLOR_MAPPING_DEEP.values()]

import os
import seaborn as sns
import matplotlib.pyplot


PROJECT_DIRECTORY_PATH = os.path.dirname(os.path.dirname(__file__))

# set font
matplotlib.pyplot.rcParams["font.family"] = "Palatino Linotype"
matplotlib.pyplot.rcParams["font.size"] = 12

matplotlib.pyplot.rcParams['mathtext.fontset'] = 'custom'
matplotlib.pyplot.rcParams['mathtext.rm'] = 'Palatino Linotype'
matplotlib.pyplot.rcParams['mathtext.it'] = 'Palatino Linotype:italic'
matplotlib.pyplot.rcParams['mathtext.bf'] = 'Palatino Linotype:bold'

FONT_PROPERTIES_HEADER = {
    "size": 17,
    "weight": "heavy",
    "color": "black"
}

FONT_PROPERTIES_SUB_HEADER = {
    "fontsize": 14,
    "fontweight": "light",
    "style": "italic",
    "color": "black"
}

normal_palette = sns.color_palette()
deep_palette = sns.color_palette("deep")

TRIAGE_MAPPING = {
    "A": "A",
    "H": "H",
    "V1": "V1",
    "V2": "V2"
}

COLOR_MAPPING_DEEP = {
    "A": deep_palette[3],
    "H": deep_palette[1],
    "V1": deep_palette[8],
    "V2": deep_palette[2]
}

COLOR_MAPPING_LIGHT = [matplotlib.pyplot.cm.colors.to_rgba(color, alpha=0.65) for color in COLOR_MAPPING_DEEP.values()]

COLOR_MAPPING_NORMAL = {
    "A": normal_palette[0],
    "H": normal_palette[1],
    "V1": normal_palette[2],
    "V2": normal_palette[3]
}

COLOR_MAPPING_NORMAL_LIGHT = [matplotlib.pyplot.cm.colors.to_rgba(color, alpha=0.65) for color in COLOR_MAPPING_NORMAL.values()]

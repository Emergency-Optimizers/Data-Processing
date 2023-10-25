import os
import seaborn as sns

PROJECT_DIRECTORY_PATH = os.path.dirname(os.path.dirname(__file__))

deep_palette = sns.color_palette("deep")
tab10_palette = sns.color_palette("tab10")

COLOR_MAPPING_DEEP = {
    'A': deep_palette[3],
    'H': deep_palette[1],
    'V1': deep_palette[8],
    'V2': deep_palette[2]
}

COLOR_MAPPING_TEST = {
    'A': deep_palette[1],
    'H': deep_palette[0],
    'V1': deep_palette[6],
    'V2': deep_palette[8]
}

COLOR_MAPPING_TEST_TAB10 = {
    'A': tab10_palette[1],
    'H': tab10_palette[0],
    'V1': tab10_palette[6],
    'V2': tab10_palette[8]
}

COLOR_MAPPING_GPT = {
    'A': sns.color_palette()[0],
    'H': sns.color_palette()[1],
    'V1': sns.color_palette("husl", 8)[4],
    'V2': sns.color_palette("husl", 8)[7]
}
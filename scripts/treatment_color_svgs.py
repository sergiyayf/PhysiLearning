import os
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd


def change_svg_background(svg_file, new_color):
    """Modify only line 13 in the SVG file."""
    with open(svg_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    if len(lines) >= 13:
        lines[12] = lines[12].replace('fill="white"', f'fill="{new_color}"')

    with open(svg_file, 'w', encoding='utf-8') as file:
        file.writelines(lines)


def process_svgs(folder, target_frames, new_color):
    """Loop through SVGs and change background for specified frames."""
    for frame in target_frames:
        svg_file = os.path.join(folder, f"snapshot{frame:08d}.svg")
        if os.path.exists(svg_file):
            change_svg_background(svg_file, new_color)
            print(f"Updated background in {svg_file}")
        else:
            print(f"File {svg_file} not found.")


if __name__ == "__main__":
    os.chdir('/home/saif/Projects/PhysiLearning')
    # Example usage:
    svg_directory = "./data/GRAPE_important_data/Best_agent_fig_2/"  # Change to your SVG folder path

    df = pd.read_hdf(f'./Evaluations/1402_pcs_evals/run_4.h5', key=f'run_0')
    treat = np.array(df['Treatment'])[::2]
    target_frames = np.where(treat == 1)[0]
    #target_frames = np.arange(62)  # Frames where background should be changed
    new_bg_color = "#d0d2d3"  # Desired background color

    process_svgs(svg_directory, target_frames, new_bg_color)

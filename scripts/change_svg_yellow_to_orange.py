import os
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
import re


def change_svg_background(svg_file, new_color):
    """Modify only line 13 in the SVG file."""
    with open(svg_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    if len(lines) >= 13:
        lines[12] = lines[12].replace('fill="white"', f'fill="{new_color}"')

    with open(svg_file, 'w', encoding='utf-8') as file:
        file.writelines(lines)

def modify_sensitive_cells(svg_file):
    """Map sensitive cell colors from (0,0,x) to soft blue scale."""
    def map_component(x, x0, x1, y0, y1):
        return int((x - x0) / (x1 - x0) * (y1 - y0) + y0)

    tree = ET.parse(svg_file)
    root = tree.getroot()

    ns = {'svg': 'http://www.w3.org/2000/svg'}
    ET.register_namespace('', ns['svg'])

    for g in root.findall('.//{http://www.w3.org/2000/svg}g'):
        if g.get('type') == 'sensitive':
            for circle in g.findall('{http://www.w3.org/2000/svg}circle'):
                fill = circle.get('fill')
                if fill and fill.startswith("rgb"):
                    match = re.match(r'rgb\((\d+),(\d+),(\d+)\)', fill)
                    if match:
                        r, g_val, b = map(int, match.groups())

                        # Only transform shades of blue (0,0,x)
                        if r == 0 and g_val == 0:
                            new_r = map_component(b, 80, 255, 49, 134)
                            new_g = map_component(b, 80, 255, 84, 169)
                            new_b = map_component(b, 80, 255, 139, 224)
                            circle.set('fill', f"rgb({new_r},{new_g},{new_b})")
                stroke = circle.get('stroke')
                if stroke and stroke.startswith("rgb"):
                    match = re.match(r'rgb\((\d+),(\d+),(\d+)\)', stroke)
                    if match:
                        r, g_val, b = map(int, match.groups())
                        # Only transform shades of blue (0,0,x)
                        if r == 0 and g_val == 0:
                            new_r = map_component(b, 80, 255, 49, 134)
                            new_g = map_component(b, 80, 255, 84, 169)
                            new_b = map_component(b, 80, 255, 139, 224)
                            circle.set('stroke', f"rgb({new_r},{new_g},{new_b})")

    tree.write(svg_file)

def modify_resistant_cells(svg_file):
    """Halve the green component of the RGB fill for resistant cells."""

    def map_component(x, x0, x1, y0, y1):
        return int((x - x0) / (x1 - x0) * (y1 - y0) + y0)
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # SVG uses namespaces; we may need to handle them
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    ET.register_namespace('', ns['svg'])  # Keep output clean

    for g in root.findall('.//{http://www.w3.org/2000/svg}g'):
        if g.get('type') == 'resistant':
            for circle in g.findall('{http://www.w3.org/2000/svg}circle'):
                fill = circle.get('fill')
                if fill and fill.startswith("rgb"):
                    match = re.match(r'rgb\((\d+),(\d+),(\d+)\)', fill)
                    if match:
                        r, g_val, b = map(int, match.groups())

                        # Apply only to colors like (x,x,0)
                        if r == g_val and b == 0:
                            new_r = map_component(r, 80, 255, 165, 255)
                            new_g = map_component(r, 80, 255, 80, 190)
                            new_b = map_component(r, 0, 255, 0, 85)
                            circle.set('fill', f"rgb({new_r},{new_g},{new_b})")
                stroke = circle.get('stroke')
                if stroke and stroke.startswith("rgb"):
                    match = re.match(r'rgb\((\d+),(\d+),(\d+)\)', stroke)
                    if match:
                        r, g_val, b = map(int, match.groups())
                        # Apply only to colors like (x,x,0)
                        if r == g_val and b == 0:
                            new_r = map_component(r, 80, 255, 165, 255)
                            new_g = map_component(r, 80, 255, 80, 190)
                            new_b = map_component(r, 0, 255, 0, 85)
                            circle.set('stroke', f"rgb({new_r},{new_g},{new_b})")
    # for g in root.findall('.//{http://www.w3.org/2000/svg}g'):
    #     if g.get('type') == 'resistant':
    #         for circle in g.findall('{http://www.w3.org/2000/svg}circle'):
    #             fill = circle.get('fill')
    #             if fill and fill.startswith("rgb"):
    #                 match = re.match(r'rgb\((\d+),(\d+),(\d+)\)', fill)
    #                 if match:
    #                     r, g_val, b = map(int, match.groups())
    #                     new_g = max(0, g_val // 2)  # Half the green value
    #                     new_rgb = f"rgb({r},{new_g},{b})"
    #                     circle.set('fill', new_rgb)
    #             stoke = circle.get('stroke')
    #             if stoke and stoke.startswith("rgb"):
    #                 match = re.match(r'rgb\((\d+),(\d+),(\d+)\)', stoke)
    #                 if match:
    #                     r, g_val, b = map(int, match.groups())
    #                     new_g = max(0, g_val // 2)
    #                     new_rgb = f"rgb({r},{new_g},{b})"
    #                     circle.set('stroke', new_rgb)


    tree.write(svg_file)

def convert_time_to_generations(svg_file):
    """Convert simulation time from days/hours/minutes to generations."""
    tree = ET.parse(svg_file)
    root = tree.getroot()

    ns = {'svg': 'http://www.w3.org/2000/svg'}
    ET.register_namespace('', ns['svg'])

    for text_elem in root.findall('.//{http://www.w3.org/2000/svg}text'):
        if text_elem.text and "Current time:" in text_elem.text:
            match = re.search(r'(\d+)\s+days,\s+(\d+)\s+hours', text_elem.text)
            if match:
                days = int(match.group(1))
                hours = int(match.group(2))

                generations = days + (hours / 24)
                # Round to 2 decimal places to keep things clean
                generation_str = f"{generations:.2f} generations"

                text_elem.text = f"Current time: {generation_str}"

    tree.write(svg_file)

def process_svgs(folder, target_frames, new_color):
    """Loop through SVGs and apply background and resistant cell modifications."""
    for frame in target_frames:
        svg_file = os.path.join(folder, f"snapshot{frame:08d}.svg")
        if os.path.exists(svg_file):
            #change_svg_background(svg_file, new_color)
            modify_resistant_cells(svg_file)
            modify_sensitive_cells(svg_file)
            convert_time_to_generations(svg_file)
            print(f"Updated {svg_file}")
        else:
            print(f"File {svg_file} not found.")


if __name__ == "__main__":
    os.chdir('/home/saif/Projects/PhysiLearning')

    svg_directory = "./data/GRAPE_important_data/test/"

    df = pd.read_hdf('./Evaluations/1402_pcs_evals/run_4.h5', key='run_0')
    treat = np.array(df['Treatment'])[::2]
    target_frames = np.arange(0,1000)

    new_bg_color = "#d0d2d3"

    process_svgs(svg_directory, target_frames, new_bg_color)

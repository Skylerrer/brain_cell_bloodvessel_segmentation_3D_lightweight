"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file can be used to write your own scripts to execute the segmentation or quantification pipelines without the GUI
########################################################################################################################

import time
from pathlib import Path

from trpseg.segmentation.blood_vessel_wall_seg import blood_vessel_wall_segmentation
from trpseg.trpseg_util.utility import count_segmentation_pixels, get_file_list_from_directory_filter


def test_vessel_wall_segmentation(input_folder_path, output_folder_path, resolution=[2.0,0.325,0.325]):

    input_file_paths = get_file_list_from_directory_filter(input_folder_path)
    blood_vessel_wall_segmentation(input_file_paths, output_folder_path, resolution)

    total_num_pixels = count_segmentation_pixels(output_folder_path, isTif=None, channel_one=False)
    total_volume = total_num_pixels * resolution[0] * resolution[1] * resolution[2]

    time_str = time.strftime("%H%M%S_")
    out_name = time_str + "total_vessel_wall_volume(from object pixel count).txt"
    volume_file_path = Path(output_folder_path) / out_name

    out_str = "Total fenestrated blood vessel wall volume (computed from summing up object pixel volumes) [" + u"\u03bc" + "m^3]: " + str(total_volume)

    with open(volume_file_path, 'w', encoding="utf-8") as f:
        f.write(out_str)



def main():

    test_vessel_wall_segmentation()



if __name__ == "__main__":
    main()

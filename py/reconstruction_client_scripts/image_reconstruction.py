from image_reconstruction_utils import *

if __name__ == '__main__':
    # currently only supports one ply creation
    # enter Timestamp for which ply should be created.
    # AHat and PV image needs to exist for the Timestamp.
    parent_dir = "C:/Users/jonathan_pc/Desktop/projectx/py/25_3_22/"
    Timestamp = 0
    create_projection_ply_for_timestamp(Timestamp, parent_dir)

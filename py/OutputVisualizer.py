import cv2
import numpy as np
import os
#import vedo
import open3d as o3d
from PIL import Image
from pathlib import Path


def match_timestamp(target, all_timestamps):
    return all_timestamps[np.argmin([abs(x - target) for x in all_timestamps])]


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def add_suffix_per_key(timestamp, key: str):
    if key == "PV" :
        return timestamp + '.png'

    if key == r"pinhole_projection\rgb":
        return timestamp + '_proj.png'

    if str(key).startswith("VLC"):
        return timestamp + '.pgm'


def load_images_timestamps_from_folder(parent_folder, flags):
    timestamps = []
    timestamp_dict = {}
    for key in flags:
        if(flags[key]):
            timestamp_dict[key] = []
            print("==========" + key + "==========")
            child_folder = parent_folder + key
            for filename in os.listdir(child_folder):
                if filename.endswith('.png'):           #rgb file from PV sensor or depth sensor
                    removed_png = os.path.splitext(os.path.basename(filename))[0]
                    removed_proj = removed_png.split("_")[0]
                    timestamps.append(removed_proj)
                    timestamp_dict[key].append(removed_proj)

                    continue

                if filename.endswith('.pgm'):           #pgm file from LL,LF,RF,RR sensors
                    timestamps.append(os.path.splitext(os.path.basename(filename))[0])
                    timestamp_dict[key].append(os.path.splitext(os.path.basename(filename))[0])

    unique_timestamps = list(set(timestamps))
    unique_timestamps.sort()
    return unique_timestamps, timestamp_dict


def timestamp_in_folder(output_dir_path, timestamp):
    if is_in_folder(timestamp + '.png',output_dir_path):
        return timestamp + '.png'

    if is_in_folder(timestamp + '_proj.png', output_dir_path):
        return timestamp + '_proj.png'

    if is_in_folder(timestamp + '.pgm', output_dir_path):
        return timestamp + '.pgm'

    return None


def visualizer_display(output_dir_path,flags):

    image_dict = {}
    timestamps, timestamp_dict = load_images_timestamps_from_folder(output_dir_path, flags)
    #for image_timestamp in timestamps:                                          #iterate over all timestamps
    for image_timestamp in timestamp_dict["PV"]:  # iterate over all timestamps
        frame_list = []
        for key in flags:                                                       #check all folders
            #print("key is:" + key)
            child_folder = output_dir_path + key
            if flags[key]:                                                      #folder exists
                #file_name = timestamp_in_folder(child_folder, image_timestamp)
                closest_stamp = match_timestamp(int(image_timestamp), [int(i) for i in timestamp_dict[key]])
                new_file_name = add_suffix_per_key(str(closest_stamp), key)
                #if(file_name != None):                                          #check folder for current timestamp
                if(new_file_name != None):
                    image_dict[key] = cv2.imread(os.path.join(child_folder, new_file_name))

                    if str(key).startswith("VLC"):
                        image_dict[key] = cv2.resize(image_dict[key], (428, 428))
                        image_dict[key] = cv2.rotate(image_dict[key], cv2.ROTATE_90_CLOCKWISE)

                    if (key == "PV"):
                        image_dict[key] = cv2.resize(image_dict[key], (720, 428))

        for key in image_dict:
            frame_list.append(image_dict[key])

        if (len(frame_list) > 1):
            numpy_horizontal_concat = np.concatenate(frame_list, axis=1)
            cv2.imshow('Hololens2 stream visualizer', numpy_horizontal_concat)

        # cv2.imshow('Main', image)
        # cv2.imshow('Numpy Vertical', numpy_vertical)
        # cv2.imshow('Numpy Horizontal', numpy_horizontal)
        # cv2.imshow('Numpy Vertical Concat', numpy_vertical_concat)

        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

    return

def write_sensors(output_dir_path,flags):


    ### configuration ###
    PV_folder = output_dir_path + r'PV'
    pinhole_folder = output_dir_path + r'pinhole_projection\rgb'

    ### text configurations ###
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 300)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2


    if(flags["PV_stream"] == True):
        for file in os.listdir(PV_folder):
            if(file.endswith('.png')):
                img = cv2.imread(os.path.join(PV_folder, file))
                bottomLeftCornerOfText = (int((img.shape[0] - (int(img.shape[0]) * (9 / 10)))), int((img.shape[1] - (int(img.shape[1]) / 10))))
                cv2.putText(img, 'RGB sensor',
                            (10,img.shape[0]-20),
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                if not os.path.exists(output_dir_path+r"\visalizerOutput\PV"):
                    os.makedirs(output_dir_path+r"\visalizerOutput\PV")
                cv2.imwrite(output_dir_path+r"\visalizerOutput\PV\\"+file,img)

    if(flags["long_depth"] == True): # Long Throw
        for file in os.listdir(pinhole_folder):
            if (file.endswith('.png')):
                img = cv2.imread(os.path.join(pinhole_folder, file))
                bottomLeftCornerOfText = (
                int((img.shape[0] - (int(img.shape[0]) * (9 / 10)))), int((img.shape[1] - (int(img.shape[1]) / 10))))
                cv2.putText(img, 'Long throw sensor',
                            (10, img.shape[0] - 20),
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                if not os.path.exists(output_dir_path + r"\visalizerOutput\pinhole_projection\rgb"):
                    os.makedirs(output_dir_path + r"\visalizerOutput\pinhole_projection\rgb")
                cv2.imwrite(output_dir_path + r"\visalizerOutput\pinhole_projection\rgb\\" + file, img)

    return

def is_in_folder(filename,folder):
    for i in os.listdir(folder):
        if(filename == i):
            return True
    return False

### streams flags

pinhole_flag = False

flags = {
    "PV": True,
    r"pinhole_projection\rgb": False,  #flag for both long throw and ahat
    "VLC LF": True,
    "VLC LL": True,
    "VLC RR": True,
    "VLC RF": True

}

output_dir_path = os.path.dirname(os.path.realpath(__file__))+'\\OutputFolder\\2021-12-07-095641\\'
visualizer_output_dir_path = os.path.dirname(os.path.realpath(__file__))+'\\OutputFolder\\2021-12-07-095641\\visalizerOutput\\'


#write_sensors(output_dir_path,flags)

visualizer_display(output_dir_path, flags)


 #    # print("Load a ply point cloud, print it, and render it")
 #    # pcd = o3d.io.read_point_cloud(r"C:\Users\eviatarsegev\Desktop\ProjectX\HoloLens2Dataset\toolbox\OutputFolder\2021-11-12-031638\2021-11-12-031638\Depth Long Throw\132811893987798775.ply")
 #    # o3d.visualization.draw_geometries([pcd])

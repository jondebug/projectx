import cv2
import numpy as np
import os
import vedo
import open3d as o3d
from PIL import Image
from pathlib import Path



def load_images_timestamps_from_folder(w_path,flags):
    PV_images = dict()
    LF_images = dict()
    RF_images = dict()
    LL_images = dict()
    RR_images = dict()
    eye_hands_images = dict()
    if(flags["PV"] == True):
        for file in os.listdir(w_path+'PV'):
            if(file.endswith(".png")):
                filename = os.path.splitext(os.path.basename(file))[0]
                img = cv2.imread(os.path.join(w_path+'PV', file))
                PV_images[filename] = img
    if (flags["front_left"] == True):
        for file in os.listdir(w_path+'VLC LF new'):
            if(file.endswith(".pgm")):
                filename = os.path.splitext(os.path.basename(file))[0]
                img = cv2.imread(os.path.join(w_path+'VLC LF new', file))
                LF_images[filename] = img
    if (flags["front_right"] == True):
        for file in os.listdir(w_path+'VLC RF new'):
            if(file.endswith(".pgm")):
                filename = os.path.splitext(os.path.basename(file))[0]
                img = cv2.imread(os.path.join(w_path + 'VLC RF new', file))
                RF_images[filename] = img
    if (flags["right_right"] == True):
        for file in os.listdir(w_path+'VLC RR new'):
            if(file.endswith(".pgm")):
                filename = os.path.splitext(os.path.basename(file))[0]
                img = cv2.imread(os.path.join(w_path + 'VLC RR new', file))
                RR_images[filename] = img
    if (flags["left_left"] == True):
        for file in os.listdir(w_path+'VLC LL new'):
            if(file.endswith(".pgm")):
                filename = os.path.splitext(os.path.basename(file))[0]
                img = cv2.imread(os.path.join(w_path + 'VLC LL new', file))
                LL_images[filename] = img
    if (flags["hand_eye"] == True):
        for file in os.listdir(w_path+'eye_hands'):
            if(file.endswith(".png")):
                filename = os.path.splitext(os.path.basename(file))[0]
                img = cv2.imread(os.path.join(w_path + 'eye_hands', file))
                eye_hands_images[filename] = img
    return PV_images,LF_images,RF_images,LL_images,RR_images,eye_hands_images
def visualize(w_path):
    print(os.path.join(w_path,"PV"))
    flags = {
        "PV": Path(os.path.join(w_path,"PV")).exists(),
        "long_depth": Path(os.path.join(w_path,"Depth Long Throw")).exists(),
        "AHaT_depth": Path(os.path.join(w_path, "Depth AHaT")).exists(),
        "front_left": Path(os.path.join(w_path, "VLC LF new")).exists(),
        "front_right": Path(os.path.join(w_path, "VLC RF new")).exists(),
        "right_right": Path(os.path.join(w_path, "VLC RR new")).exists(),
        "left_left": Path(os.path.join(w_path, "VLC LL new")).exists(),
        "hand_eye" : Path(os.path.join(w_path, "eye_hands")).exists()
    }
    print(flags)
    original_path = str(w_path)
    w_path = str(w_path)+"\\"
    PV_images,LF_images,RF_images,LL_images,RR_images,eye_hands_images = load_images_timestamps_from_folder(str(w_path),flags)
    all_images = {**PV_images,**LF_images,**RF_images,**LL_images,**RR_images,**eye_hands_images}

    pv_img = False
    lf_img = False
    rr_img = False
    ll_img = False
    rf_img = False
    eh_img = False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(original_path +'.avi', fourcc, 15, (1520, 428))

    for timestamp in sorted(all_images.keys()):
        timestamp = str(timestamp)
        if(all_images.get(timestamp) is not None):
            if(flags["PV"] == True):
                temp = PV_images.get(timestamp)
                if(temp is not None):
                    pv_image = PV_images[timestamp]
                    pv_img = True

            if (flags["front_left"] == True):
                if(LF_images.get(timestamp) is not None):
                    LF_image = LF_images[timestamp]
                    LF_image = cv2.resize(LF_image, (428, LF_image.shape[0]))
                    LF_image_rotate = cv2.rotate(LF_image, cv2.ROTATE_90_CLOCKWISE)
                    lf_img = True

            if (flags["front_right"] == True):
                if(RF_images.get(timestamp) is not None):
                    RF_image = RF_images[timestamp]
                    RF_image = cv2.resize(RF_image, (428, RF_image.shape[0]))
                    RF_image_rotate = cv2.rotate(RF_image,cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rf_img = True
            if (flags["right_right"] == True):
                if(RR_images.get(timestamp) is not None):
                    RR_image = RR_images[timestamp]
                    RR_image = cv2.resize(RR_image, (428, RR_image.shape[0]))
                    RR_image = cv2.rotate(RR_image, cv2.ROTATE_90_CLOCKWISE)
                    rr_img = True
            if (flags["left_left"] == True):
                if(LL_images.get(timestamp) is not None):
                    LL_image = LL_images[timestamp]
                    LL_image = cv2.resize(LL_image, (428, LL_image.shape[0]))
                    LL_image = cv2.rotate(LL_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    ll_img = True
            if (flags["hand_eye"] == True):
                temp = eye_hands_images.get(timestamp)
                if (temp is not None):
                    eye_hand_image = eye_hands_images[timestamp]
                    eh_img = True

            all_flags = int(pv_img) + int(lf_img) + int(rf_img) + int(ll_img) + int(rr_img) + int(eh_img)
            if(flags["PV"] and flags["front_right"] and flags["front_left"] and all_flags>3):
                numpy_horizontal_concat1 = np.concatenate((pv_image, eye_hand_image ), axis=1)
                #numpy_horizontal_concat2 = np.concatenate((eye_hand_image, LL_image, RR_image), axis=1)
                #numpy_vertical_concat = np.concatenate((numpy_horizontal_concat1, numpy_horizontal_concat2), axis=0)
                video.write(numpy_horizontal_concat1)
            #     cv2.imshow('Hololens2 stream visualizer', numpy_vertical_concat)
            #
            # if cv2.waitKey(20) & 0xFF == ord('q'):
            #     break

w_path = r"C:\Users\jonathan_pc\Desktop\project_A_hololense_3D\git_repo\HoloLens2ForCV\Samples\StreamRecorder\StreamRecorderConverter\2022-01-04-124017_jonathan_drawer_desktop_"
visualize(w_path)
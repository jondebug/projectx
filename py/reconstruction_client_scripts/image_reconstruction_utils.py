from utils import *
import struct

VIDEO_STREAM_HEADER_FORMAT = "@qIIII20f"
RM_EXTRINSICS_HEADER_FORMAT = "@16f"
RM_STREAM_HEADER_FORMAT = "@qIIII16f"


def create_projection_ply_for_timestamp(timestamp, parent_dir):
    # read relevant file:
    pv_hdr, pv_img = read_pv_info_from_files(timestamp, parent_dir)
    depth_hdr, depth_img = read_depth_info_from_files(timestamp, parent_dir)
    lut_header, lut = read_extrinsics_info_from_files(timestamp, parent_dir)

    # start the projection:
    create_ply(pv_hdr, pv_img, depth_hdr, depth_img, lut_header, lut, timestamp, parent_dir)

    return


def read_pv_info_from_files(timestamp, parent_dir):
    str_timestamp = str(timestamp)
    child_folder_path = parent_dir + "PV/"
    pv_header_format = VIDEO_STREAM_HEADER_FORMAT

    f_header = open(child_folder_path + "pv_header_" + str_timestamp, "r")
    data = struct.unpack(pv_header_format, f_header.read())
    header = data  # header = self.header_data(*data)

    f_data = open(child_folder_path + "pv_data_" + str_timestamp, "r")
    image_data = f_data.read()

    pv_img = np.frombuffer(image_data, dtype=np.uint8).reshape(
        (header.ImageHeight, header.ImageWidth, header.PixelStride))  # TODO use this to recreate image from binary fil
    pv_header = header

    return pv_header, pv_img


def read_depth_info_from_files(timestamp, parent_dir):
    str_timestamp = str(timestamp)
    child_folder_path = parent_dir + "Depth AHat/"
    ahat_header_format = RM_STREAM_HEADER_FORMAT

    f_header = open(child_folder_path + "ahat_header_" + str_timestamp, "r")
    data = struct.unpack(ahat_header_format, f_header.read())
    header = data  # header = self.header_data(*data)

    f_data = open(child_folder_path + "pv_data_" + str_timestamp, "r")
    depth_data = f_data.read()

    depth_img = np.frombuffer(depth_data, dtype=np.uint16).reshape(
        (header.ImageHeight, header.ImageWidth))
    depth_header = header

    return depth_header, depth_img


def read_extrinsics_info_from_files(timestamp, parent_dir):
    lut_file_name = str(timestamp) + "_1"  # lets start off with using the first lut.

    child_folder_path = parent_dir + "ahat_lut_extrinsics/"

    extr_header_format = RM_EXTRINSICS_HEADER_FORMAT

    f_header = open(child_folder_path + "extr_header_" + lut_file_name, "r")
    data = struct.unpack(extr_header_format, f_header.read())
    header = data  # header = self.header_data(*data)

    f_data = open(child_folder_path + "extr_lut_data_" + lut_file_name, "r")
    lut_data = f_data.read()
    lut = np.frombuffer(lut_data, dtype=np.float32).reshape((512 * 512, 3))

    lut_header = header

    return lut_header, lut


def create_ply(pv_hdr, pv_img, depth_hdr, depth_img, lut_header, lut, timestamp, parent_dir):
    # Get xyz points in camera space
    points = get_points_in_cam_space(depth_img, lut)

    rig2cam_transform = np.array([
        lut_header.rig2camTransformM11, lut_header.rig2camTransformM12, lut_header.rig2camTransformM13,
        lut_header.rig2camTransformM14,
        lut_header.rig2camTransformM21, lut_header.rig2camTransformM22, lut_header.rig2camTransformM23,
        lut_header.rig2camTransformM24,
        lut_header.rig2camTransformM31, lut_header.rig2camTransformM32, lut_header.rig2camTransformM33,
        lut_header.rig2camTransformM34,
        lut_header.rig2camTransformM41, lut_header.rig2camTransformM42, lut_header.rig2camTransformM43,
        lut_header.rig2camTransformM44]).reshape(4, 4)

    pv2world_transform = np.array([
        pv_hdr.PVtoWorldtransformM11, pv_hdr.PVtoWorldtransformM12, pv_hdr.PVtoWorldtransformM13,
        pv_hdr.PVtoWorldtransformM14,
        pv_hdr.PVtoWorldtransformM21, pv_hdr.PVtoWorldtransformM22, pv_hdr.PVtoWorldtransformM23,
        pv_hdr.PVtoWorldtransformM24,
        pv_hdr.PVtoWorldtransformM31, pv_hdr.PVtoWorldtransformM32, pv_hdr.PVtoWorldtransformM33,
        pv_hdr.PVtoWorldtransformM34,
        pv_hdr.PVtoWorldtransformM41, pv_hdr.PVtoWorldtransformM42, pv_hdr.PVtoWorldtransformM43,
        pv_hdr.PVtoWorldtransformM44]).reshape(4, 4)

    # depth_hdr = ahat_receiver.latest_header
    rig2world_transform = np.array([
        depth_hdr.rig2worldTransformM11, depth_hdr.rig2worldTransformM12, depth_hdr.rig2worldTransformM13,
        depth_hdr.rig2worldTransformM14,
        depth_hdr.rig2worldTransformM21, depth_hdr.rig2worldTransformM22, depth_hdr.rig2worldTransformM23,
        depth_hdr.rig2worldTransformM24,
        depth_hdr.rig2worldTransformM31, depth_hdr.rig2worldTransformM32, depth_hdr.rig2worldTransformM33,
        depth_hdr.rig2worldTransformM34,
        depth_hdr.rig2worldTransformM41, depth_hdr.rig2worldTransformM42, depth_hdr.rig2worldTransformM43,
        depth_hdr.rig2worldTransformM44]).reshape(4, 4)

    xyz, cam2world_transform = cam2world(points, rig2cam_transform, rig2world_transform)

    focal_length = [pv_hdr.fx, pv_hdr.fy]
    principal_point = [pv_hdr.ox, pv_hdr.oy]

    # Project from depth to pv going via world space
    rgb, depth = project_on_pv(
        xyz, pv_img, pv2world_transform,
        focal_length, principal_point)

    # Project depth on virtual pinhole camera and save corresponding
    # rgb image inside <workspace>/pinhole_projection folder
    # Create virtual pinhole camera
    scale = 1
    width = 320 * scale
    height = 288 * scale
    proj_focal_length = 200 * scale
    intrinsic_matrix = np.array([[proj_focal_length, 0, width / 2.],
                                 [0, proj_focal_length, height / 2.],
                                 [0, 0, 1.]])
    rgb_proj, depth = project_on_depth(
        points, rgb, intrinsic_matrix, width, height)

    #cv2.imshow('test rgb', (rgb_proj).astype(np.uint8))
    depth = (depth * DEPTH_SCALING_FACTOR).astype(np.uint16)
    #cv2.imshow('test depth', (depth).astype(np.uint16))

    # Save colored point clouds as ply files
    output_path = parent_dir + "script_outputs/" + str(timestamp) + ".ply"
    save_ply(output_path, points, rgb, pv2world_transform)

    cv2.imwrite(output_path + "rgb.png", rgb_proj)
    cv2.imwrite(output_path + "depth.png", (depth).astype(np.uint16))

    return

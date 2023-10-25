import open3d as o3d
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import random
from tqdm import tqdm


images_df = pd.read_pickle("data/images.pkl")
train_df = pd.read_pickle("data/train.pkl")
points3D_df = pd.read_pickle("data/points3D.pkl")
point_desc_df = pd.read_pickle("data/point_desc.pkl")


def load_point_cloud(points3D_df):
    pointcloud_points = points3D_df["XYZ"].to_list()
    pointcloud_colors = np.divide(points3D_df["RGB"].to_list(), 255)

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np.array(pointcloud_points).reshape(-1, 3))
    pointcloud.colors = o3d.utility.Vector3dVector(np.array(pointcloud_colors).reshape(-1, 3))

    return pointcloud


def load_axes():
    lineset_points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    lineset_lines = [[0, 1], [0, 2], [0, 3]]  # X, Y, Z
    lineset_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # R, G, B

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.array(lineset_points).reshape(-1, 3))
    lineset.lines = o3d.utility.Vector2iVector(np.array(lineset_lines).reshape(-1, 2))
    lineset.colors = o3d.utility.Vector3dVector(np.array(lineset_colors).reshape(-1, 3))

    return lineset


def load_cube():
    box = o3d.geometry.TriangleMesh.create_box()

    return box


def get_transform_mat(R_euler, t, scale):
    R_mat = R.from_euler("xyz", R_euler, degrees=True).as_matrix()
    t_mat = t.reshape(3, 1)
    scale_mat = np.eye(3) * scale

    transform_mat = np.concatenate([scale_mat @ R_mat, t_mat], axis=1)

    return transform_mat


def update_cube():
    global vis, cube, cube_vertices, R_euler, t, scale

    transform_mat = get_transform_mat(R_euler, t, scale)
    transform_vertices = (transform_mat @ np.concatenate([cube_vertices.transpose(), np.ones([1, cube_vertices.shape[0]])], axis=0)).transpose()

    cube.vertices = o3d.utility.Vector3dVector(transform_vertices)
    cube.compute_vertex_normals()
    cube.paint_uniform_color([1, 0.706, 0])
    vis.update_geometry(cube)


def toggle_key_shift(vis, action, mods):
    global shift_pressed
    if action == 1:  # key down
        shift_pressed = True
    elif action == 0:  # key up
        shift_pressed = False
    return True


def update_tx(vis):
    global t, shift_pressed
    t[0] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_ty(vis):
    global t, shift_pressed
    t[1] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_tz(vis):
    global t, shift_pressed
    t[2] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_rx(vis):
    global R_euler, shift_pressed
    R_euler[0] += -1 if shift_pressed else 1
    update_cube()


def update_ry(vis):
    global R_euler, shift_pressed
    R_euler[1] += -1 if shift_pressed else 1
    update_cube()


def update_rz(vis):
    global R_euler, shift_pressed
    R_euler[2] += -1 if shift_pressed else 1
    update_cube()


def update_scale(vis):
    global scale, shift_pressed
    scale += -0.05 if shift_pressed else 0.05
    update_cube()


def get_cube_vertices():
    global vis, cube, cube_vertices, R_euler, t, scale, shift_pressed

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # load point cloud
    pointcloud = load_point_cloud(points3D_df)
    vis.add_geometry(pointcloud)

    # load axes
    axes = load_axes()
    vis.add_geometry(axes)

    # load cube
    cube = load_cube()
    cube_vertices = np.asarray(cube.vertices).copy()
    vis.add_geometry(cube)

    R_euler = np.array([0, 0, 0]).astype(float)
    t = np.array([0, 0, 0]).astype(float)
    scale = 1
    update_cube()

    # just set a proper initial camera view
    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_transform_mat = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_extrinsic_mat = np.concatenate([initial_transform_mat, np.array([[0, 0, 0, 1]])], axis=0)
    setattr(vc_cam, "extrinsic", initial_extrinsic_mat)
    vc.convert_from_pinhole_camera_parameters(vc_cam)

    # set key callback
    shift_pressed = False
    vis.register_key_action_callback(340, toggle_key_shift)
    vis.register_key_action_callback(344, toggle_key_shift)
    vis.register_key_callback(ord("A"), update_tx)
    vis.register_key_callback(ord("S"), update_ty)
    vis.register_key_callback(ord("D"), update_tz)
    vis.register_key_callback(ord("Z"), update_rx)
    vis.register_key_callback(ord("X"), update_ry)
    vis.register_key_callback(ord("C"), update_rz)
    vis.register_key_callback(ord("V"), update_scale)

    print("[Keyboard usage]")
    print("Translate along X-axis\tA / Shift+A")
    print("Translate along Y-axis\tS / Shift+S")
    print("Translate along Z-axis\tD / Shift+D")
    print("Rotate    along X-axis\tZ / Shift+Z")
    print("Rotate    along Y-axis\tX / Shift+X")
    print("Rotate    along Z-axis\tC / Shift+C")
    print("Scale                 \tV / Shift+V")

    vis.run()

    vis.destroy_window()

    return np.asarray(cube.vertices)


def get_surface_points(cube):
    n = 9  # Number of Equal Parts
    surface_points = []
    x_vector = cube[1] - cube[0]
    y_vector = cube[4] - cube[0]
    z_vector = cube[2] - cube[0]

    # Plane z = 0
    base = cube[0]
    for y in range(n):
        for x in range(n):
            surface_points.append([base + (x / n) * x_vector + (y / n) * y_vector, [255, 0, 0]])

    # Plane y = 0
    base = cube[0]
    for z in range(n):
        for x in range(n):
            surface_points.append([base + (x / n) * x_vector + (z / n) * z_vector, [0, 255, 0]])

    # Plane x = 0
    base = cube[0]
    for z in range(n):
        for y in range(n):
            surface_points.append([base + (y / n) * y_vector + (z / n) * z_vector, [0, 0, 255]])

    # Plane z = z_vector
    base = cube[2]
    for y in range(n):
        for x in range(n):
            surface_points.append([base + (x / n) * x_vector + (y / n) * y_vector, [255, 255, 0]])

    # Plane y = y_vector
    base = cube[4]
    for z in range(n):
        for x in range(n):
            surface_points.append([base + (x / n) * x_vector + (z / n) * z_vector, [255, 0, 255]])

    # Plane x = x_vector
    base = cube[1]
    for z in range(n):
        for y in range(n):
            surface_points.append([base + (y / n) * y_vector + (z / n) * z_vector, [0, 255, 255]])

    return surface_points


def average(x):
    return list(np.mean(x, axis=0))


def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID", "XYZ", "RGB", "DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc


def DLT(X, U, K, distCoeffs, s):
    U = cv2.undistortImagePoints(U, K, distCoeffs).reshape(U.shape[0], 2)
    f_x, f_y, c_x, c_y = K[0][0], K[1][1], K[0][2], K[1][2]

    A = []
    for i in range(s):
        x, y, z, u, v = X[i][0], X[i][1], X[i][2], U[i][0], U[i][1]
        A.append([x * f_x, y * f_x, z * f_x, f_x, 0, 0, 0, 0, x * c_x - u * x, y * c_x - u * y, z * c_x - u * z, c_x - u,])
        A.append([0, 0, 0, 0, x * f_y, y * f_y, z * f_y, f_y, x * c_y - v * x, y * c_y - v * y, z * c_y - v * z, c_y - v,])

    U1, S1, Vh1 = np.linalg.svd(np.matrix(A))

    x_bar = np.array(Vh1[-1])[0]

    R_bar = np.matrix([[x_bar[0], x_bar[1], x_bar[2]],
                       [x_bar[4], x_bar[5], x_bar[6]],
                       [x_bar[8], x_bar[9], x_bar[10]],])

    U2, S2, Vh2 = np.linalg.svd(R_bar)

    R = np.matmul(U2, Vh2)

    beta = 1 / (S2.sum() / 3)

    if beta * (X[0][0] * x_bar[8] + X[0][1] * x_bar[9] + X[0][2] * x_bar[10] + x_bar[11]) < 0:
        R = -R
        beta = -beta

    t = np.matrix([[beta * x_bar[3]], [beta * x_bar[7]], [beta * x_bar[11]]])

    return R, t


def RANSAC(points3D, points2D, K, distCoeffs):
    s = np.max([int(np.floor(np.array([points3D.shape[0] * 0.01])).item()), 6])
    N = 100
    d = 15
    T = points2D.shape[0] / 2

    max_inliers = 0
    min_mean = 0
    for n in range(N):
        idx = random.sample(range(points3D.shape[0]), s)
        points3D_sam = np.array([points3D[i] for i in idx])
        points2D_sam = np.array([points2D[i] for i in idx])

        R, t = DLT(points3D_sam, points2D_sam, K, distCoeffs, s)

        parameter = np.matmul(K, np.concatenate((R, t), axis=1))
        X = np.concatenate((points3D, np.ones((points3D.shape[0], 1))), axis=1).transpose()

        lambdaU = np.matmul(parameter, X)
        points2D_hat = []
        for lambdau, lambdav, lambda1 in lambdaU.transpose().tolist():
            points2D_hat.append([lambdau / lambda1, lambdav / lambda1])

        inliers = 0
        error_list = []
        for i in range(points2D.shape[0]):
            error = np.linalg.norm(points2D[i] - points2D_hat[i])
            error_list.append(error)
            if error < d:
                inliers += 1
        mean = np.mean(np.array(error_list))

        if n == 0 or (inliers > np.max([T, max_inliers]) and mean < min_mean):
            ideal_R = R
            ideal_t = t
            min_mean = mean
            max_inliers = inliers

    return ideal_R, ideal_t


def PnPsolver(query, model, IntrinsicMatrix, distCoeffs):
    kp_query, desc_query = query
    kp_model, desc_model = model

    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(desc_query, desc_model, k=2)

    gmatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            gmatches.append(m)

    points2D = np.empty((0, 2))
    points3D = np.empty((0, 3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D, kp_query[query_idx]))
        points3D = np.vstack((points3D, kp_model[model_idx]))

    return RANSAC(points3D, points2D, IntrinsicMatrix, distCoeffs)


def main():
    cube_vertices = get_cube_vertices()

    surface_points = get_surface_points(cube_vertices)

    fps = 15
    frame_size = cv2.imread("./data/frames/valid_img5.jpg").shape
    dim1 = frame_size[1]
    dim2 = frame_size[0]
    videowriter = cv2.VideoWriter("AR_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (dim1, dim2))

    IntrinsicMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
    distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    sorted_image_df_valid = images_df[images_df.NAME.str.startswith("valid")]
    new_IMAGE_ID = []
    for name in sorted_image_df_valid.NAME:
        new_IMAGE_ID.append(int(name[9:][:-4]))
    sorted_image_df_valid.insert(0, "new_IMAGE_ID", new_IMAGE_ID)
    sorted_image_df_valid = sorted_image_df_valid.sort_values(by="new_IMAGE_ID")

    for id in tqdm(sorted_image_df_valid["IMAGE_ID"]):
        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == id]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve DLT
        rmat, tvec = PnPsolver((kp_query, desc_query), (kp_model, desc_model), IntrinsicMatrix, distCoeffs)

        unsorted_points = surface_points
        T_inv = np.linalg.inv(np.matrix(np.concatenate((np.concatenate((rmat, tvec), axis=1), np.array([[0, 0, 0, 1]]),), axis=0,)))
        apex = np.matmul(T_inv, np.array([0, 0, 0, 1]).transpose()).tolist()[0][:3]

        sorted_points = []
        for point, color in unsorted_points:
            sorted_points.append([point, np.linalg.norm(point - apex), color])
        sorted_points.sort(key=lambda point: point[1], reverse=True)

        parameter = np.matmul(IntrinsicMatrix, np.concatenate((rmat, tvec), axis=1))
        image = cv2.imread("./data/frames/" + sorted_image_df_valid.loc[sorted_image_df_valid["IMAGE_ID"] == id]["NAME"].tolist()[0])
        for point3D, _, color in sorted_points:
            X = np.concatenate((point3D, [1]), axis=None).transpose()
            lambdaU = np.matmul(parameter, X).tolist()
            point2D = np.int_(np.round(np.divide([lambdaU[0][0], lambdaU[0][1]], lambdaU[0][2])))
            if point2D[0] >= 0 and point2D[0] < dim1 and point2D[1] >= 0 and point2D[1] < dim2:
                image = cv2.circle(image, tuple(point2D), 5, tuple(color), thickness=cv2.FILLED, lineType=cv2.LINE_AA,)

        videowriter.write(image)

    videowriter.release()


if __name__ == "__main__":
    main()

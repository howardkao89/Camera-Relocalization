from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
import random
import open3d as o3d
from tqdm import tqdm


images_df = pd.read_pickle("data/images.pkl")
train_df = pd.read_pickle("data/train.pkl")
points3D_df = pd.read_pickle("data/points3D.pkl")
point_desc_df = pd.read_pickle("data/point_desc.pkl")


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


def visualization(points3D_df, T_inv_list):
    pointcloud_points = points3D_df["XYZ"].to_list()
    pointcloud_colors = np.divide(points3D_df["RGB"].to_list(), 255)

    lineset_points = []
    lineset_lines = []
    lineset_colors = []

    trianglemesh_vertices = []
    trianglemesh_triangles = []

    for T_inv in T_inv_list:
        apex = np.matmul(T_inv, np.array([0, 0, 0, 1]).transpose()).tolist()[0][:3]
        vertex1 = np.matmul(T_inv, np.array([0.25, 0.25, 1, 1]).transpose()).tolist()[0][:3]
        vertex2 = np.matmul(T_inv, np.array([-0.25, 0.25, 1, 1]).transpose()).tolist()[0][:3]
        vertex3 = np.matmul(T_inv, np.array([-0.25, -0.25, 1, 1]).transpose()).tolist()[0][:3]
        vertex4 = np.matmul(T_inv, np.array([0.25, -0.25, 1, 1]).transpose()).tolist()[0][:3]

        lineset_points.extend([apex, vertex1, vertex2, vertex3, vertex4])
        trianglemesh_vertices.extend([vertex1, vertex2, vertex3, vertex4])

    for i in range(0, 5 * len(T_inv_list), 5):
        if i != 5 * (len(T_inv_list) - 1):
            lineset_lines.append([i, i + 5])
            lineset_colors.append([0, 1, 0])

        lineset_lines.extend([[i, i + 1], [i, i + 2], [i, i + 3], [i, i + 4],  [i + 1, i + 2], [i + 2, i + 3], [i + 3, i + 4], [i + 4, i + 1],])
        lineset_colors.extend([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],])

    for j in range(0, 4 * len(T_inv_list), 4):
        trianglemesh_triangles.extend([[j, j + 1, j + 2], [j, j + 2, j + 3], [j, j + 3, j + 1], [j + 1, j + 3, j + 2],])

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np.array(pointcloud_points).reshape(-1, 3))
    pointcloud.colors = o3d.utility.Vector3dVector(np.array(pointcloud_colors).reshape(-1, 3))

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.array(lineset_points).reshape(-1, 3))
    lineset.lines = o3d.utility.Vector2iVector(np.array(lineset_lines).reshape(-1, 2))
    lineset.colors = o3d.utility.Vector3dVector(np.array(lineset_colors).reshape(-1, 3))

    trianglemesh = o3d.geometry.TriangleMesh()
    trianglemesh.vertices = o3d.utility.Vector3dVector(np.array(trianglemesh_vertices).reshape(-1, 3))
    trianglemesh.triangles = o3d.utility.Vector3iVector(np.array(trianglemesh_triangles).reshape(-1, 3))
    trianglemesh.paint_uniform_color(np.divide([218, 226, 241], 255))

    o3d.visualization.draw_geometries([lineset, trianglemesh, pointcloud])


def main():
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

    T_inv_list = []
    error_rotation_list = []
    error_translation_list = []

    for id in tqdm(sorted_image_df_valid["IMAGE_ID"]):
        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == id]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve DLT
        rmat, tvec = PnPsolver((kp_query, desc_query), (kp_model, desc_model), IntrinsicMatrix, distCoeffs)

        T_inv_list.append(np.linalg.inv(np.matrix(np.concatenate((np.concatenate((rmat, tvec), axis=1), np.array([[0, 0, 0, 1]]),), axis=0,))))
        rotq = R.from_matrix(rmat).as_quat()
        tvec = np.array(tvec.tolist()).reshape(1, 3)

        # Get camera pose groudtruth
        ground_truth = sorted_image_df_valid.loc[sorted_image_df_valid["IMAGE_ID"] == id]
        rotq_gt = np.array(ground_truth[["QX", "QY", "QZ", "QW"]].values)
        tvec_gt = np.array(ground_truth[["TX", "TY", "TZ"]].values)

        R_e = (R.from_quat(rotq_gt) * R.from_quat(rotq).inv()).as_quat()[0]
        error_rotation_list.append(2 * np.min(np.array([np.arccos([R_e[3]]), np.pi - np.arccos([R_e[3]])])))
        error_translation_list.append(np.linalg.norm(tvec_gt - tvec))

    error_rotation = np.median(error_rotation_list)
    print(f"Rotation Error: {error_rotation}")
    error_translation = np.median(error_translation_list)
    print(f"Translation Error: {error_translation}")

    visualization(points3D_df, T_inv_list)


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm


def image_loader(dir_path:str, scale:float=2):
    # load camera intrinsic parameters (K matrix)
    with open("K.txt", "r") as f:
        lines = f.readlines()
        K = np.float32([i.strip().split(" ") for i in lines])
        
    img_list = [os.path.join(dir_path, i)for i in os.listdir(dir_path)]
    print(img_list)

    # Downscale instrinsic parameters
    K[0, 0] /= scale
    K[1, 1] /= scale
    K[0, 2] /= scale
    K[1, 2] /= scale

    return img_list, K


def downscale_image(img, scale=2):
    for _ in range(1, int(scale / 2) + 1):
        img = cv2.pyrDown(img)
    
    return img


def triangulation(point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2):
    pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T)
    print(f"point cloud from triangulation:\n {pt_cloud}")
    return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])


def pnp(obj_point, image_point, K, dist_coeff, rot_vector, initial):
    if initial == 1:
        obj_point = obj_point[:, 0 ,:]
        image_point = image_point.T
        rot_vector = rot_vector.T 

    _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
    rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

    if inlier is not None:
        image_point = image_point[inlier[:, 0]]
        obj_point = obj_point[inlier[:, 0]]
        rot_vector = rot_vector[inlier[:, 0]]

    return rot_matrix, tran_vector, image_point, obj_point, rot_vector


def reprojection_error(obj_points, image_points, transform_matrix, K, homogenity):
    '''
    Calculates the reprojection error ie the distance between the projected points and the actual points.
    returns total error, object points
    '''
    rot_matrix = transform_matrix[:3, :3]
    tran_vector = transform_matrix[:3, 3]
    rot_vector, _ = cv2.Rodrigues(rot_matrix)
    if homogenity == 1:
        obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
    image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
    image_points_calc = np.float32(image_points_calc[:, 0, :])
    total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
    
    return total_error / len(image_points_calc), obj_points
   

def optimal_reprojection_error(self, obj_points):
    '''
    calculates of the reprojection error during bundle adjustment
    returns error 
    '''
    transform_matrix = obj_points[0:12].reshape((3,4))
    K = obj_points[12:21].reshape((3,3))
    rest = int(len(obj_points[21:]) * 0.4)
    p = obj_points[21:21 + rest].reshape((2, int(rest/2))).T
    obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:])/3), 3))
    rot_matrix = transform_matrix[:3, :3]
    tran_vector = transform_matrix[:3, 3]
    rot_vector, _ = cv2.Rodrigues(rot_matrix)
    image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
    image_points = image_points[:, 0, :]
    error = [ (p[idx] - image_points[idx])**2 for idx in range(len(p))]

    return np.array(error).ravel()/len(p)


def bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error):
    '''
    Bundle adjustment for the image and object points
    returns object points, image points, transformation matrix
    '''
    opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
    opt_variables = np.hstack((opt_variables, opt.ravel()))
    opt_variables = np.hstack((opt_variables, _3d_point.ravel()))

    values_corrected = least_squares(self.optimal_reprojection_error, opt_variables, gtol = r_error).x
    K = values_corrected[12:21].reshape((3,3))
    rest = int(len(values_corrected[21:]) * 0.4)
    return values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:])/3), 3)), values_corrected[21:21 + rest].reshape((2, int(rest/2))).T, values_corrected[0:12].reshape((3,4))


def to_ply(point_clouds, colors):
    out_points = point_clouds.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    print(f"out_colors shape: {out_colors.shape}, out_points shape: {out_points.shape}")
    verts = np.hstack([out_points, out_colors])

    mean = np.mean(verts[:, :3], axis=0)
    scaled_verts = verts[:, :3] - mean
    dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist) + 300)

    verts = verts[indx]
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''
    
    with open('res.ply', 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


def correspondences(img_points_1, img_points_2, img_points_3):
    cr_points_1 = []
    cr_points_2 = []

    for i in range(img_points_1.shape[0]):
        a = np.where(img_points_2 == img_points_1[i, :])
        if a[0].size != 0:
            cr_points_1.append(i)
            cr_points_2.append(a[0][0])

    mask_array_1 = np.ma.array(img_points_2, mask=False)
    mask_array_1.mask[cr_points_2] = True
    mask_array_1 = mask_array_1.compressed()
    mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

    mask_array_2 = np.ma.array(img_points_3, mask=False)
    mask_array_2.mask[cr_points_2] = True
    mask_array_2 = mask_array_2.compressed()
    mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)

    return np.array(cr_points_1), np.array(cr_points_2), mask_array_1, mask_array_2


def find_features(image_0, image_1):
    '''
    Feature detection using the sift algorithm and KNN
    return keypoints(features) of image1 and image2
    '''

    sift = cv2.SIFT_create()
    key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
    key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_0, desc_1, k=2)
    feature = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            feature.append(m)

    return np.float32([key_points_0[m.queryIdx].pt for m in feature]), np.float32([key_points_1[m.trainIdx].pt for m in feature])


def run(img_dir:str,apply_bundle_adjustment:boolean=False):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    image_list, K = image_loader(img_dir)
    pose_array = K.ravel()

    transform_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    transform_matrix_1 = np.empty((3, 4))

    pose_0 = np.matmul(K, transform_matrix_0)
    pose_1 = np.empty((3, 4)) 
    total_points = np.zeros((1, 3))
    total_colors = np.zeros((1, 3))

    image_0 = downscale_image(cv2.imread(image_list[0]))
    image_1 = downscale_image(cv2.imread(image_list[1]))

    feature_0, feature_1 = find_features(image_0, image_1)
    essential_matrix, em_mask = cv2.findEssentialMat(feature_0, feature_1, K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
    feature_0 = feature_0[em_mask.ravel() == 1]
    feature_1 = feature_1[em_mask.ravel() == 1]

    _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, feature_0, feature_1, K)
    feature_0 = feature_0[em_mask.ravel() > 0]
    feature_1 = feature_1[em_mask.ravel() > 0]

    transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
    transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], tran_matrix.ravel())

    pose_1 = np.matmul(K, transform_matrix_1)

    feature_0, feature_1, points_3d = triangulation(pose_0, pose_1, feature_0, feature_1)
    error, points_3d = reprojection_error(points_3d, feature_1, transform_matrix_1, K, homogenity = 1)
        #ideally error < 1
    _, _, feature_1, points_3d, _ = pnp(points_3d, feature_1, K, np.zeros((5, 1), dtype=np.float32), feature_0, initial=1)
    total_images = len(image_list) - 2 
    pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))
    threshold = 0.5

    for i in tqdm(range(total_images)):
        image_2 = downscale_image(cv2.imread(image_list[i + 2]))
        features_cur, features_2 = find_features(image_1, image_2)

        if i != 0:
            feature_0, feature_1, points_3d = triangulation(pose_0, pose_1, feature_0, feature_1)
            feature_1 = feature_1.T
            points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
            points_3d = points_3d[:, 0, :]

        cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = correspondences(feature_1, features_cur, features_2)
        cm_points_2 = features_2[cm_points_1]
        cm_points_cur = features_cur[cm_points_1]

        rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = pnp(points_3d[cm_points_0], cm_points_2, K, np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial = 0)
        print(rot_matrix.shape)
        print(tran_matrix.shape)
        transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
        pose_2 = np.matmul(K, transform_matrix_1)

        error, points_3d = reprojection_error(points_3d, cm_points_2, transform_matrix_1, K, homogenity = 0)

        cm_mask_0, cm_mask_1, points_3d = triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
        error, points_3d = reprojection_error(points_3d, cm_mask_1, transform_matrix_1, K, homogenity = 1)
        print("Reprojection Error: ", error)
        pose_array = np.hstack((pose_array, pose_2.ravel()))

        if apply_bundle_adjustment:
            points_3d, cm_mask_1, transform_matrix_1 = bundle_adjustment(points_3d, cm_mask_1, transform_matrix_1, K, threshold)
            pose_2 = np.matmul(K, transform_matrix_1)
            error, points_3d = reprojection_error(points_3d, cm_mask_1, transform_matrix_1, K, homogenity = 0)
            print("Bundle Adjusted error: ",error)
            total_points = np.vstack((total_points, points_3d))
            points_left = np.array(cm_mask_1, dtype=np.int32)
            color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
            total_colors = np.vstack((total_colors, color_vector))
        else:
            total_points = np.vstack((total_points, points_3d[:, 0, :]))
            points_left = np.array(cm_mask_1, dtype=np.int32)
            color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
            total_colors = np.vstack((total_colors, color_vector)) 

        transform_matrix_0 = np.copy(transform_matrix_1)
        pose_0 = np.copy(pose_1)
        plt.scatter(i, error)
        plt.pause(0.05)

        image_0 = np.copy(image_1)
        image_1 = np.copy(image_2)
        feature_0 = np.copy(features_cur)
        feature_1 = np.copy(features_2)
        pose_1 = np.copy(pose_2)
        cv2.imshow(image_list[0].split('\\')[-2], image_2)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()
    to_ply(total_points, total_colors)

run("images/monument")
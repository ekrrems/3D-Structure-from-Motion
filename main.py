# import cv2
# import numpy as np
# import os
# from scipy.optimize import least_squares
# from tomlkit import boolean
# from tqdm import tqdm
# import matplotlib.pyplot as plt


# def downscale(K, scale_factor:float):
#     """
#     Downscale the Image Mtrix (K)
#     """
#     K[0,0] /= scale_factor
#     K[1, 1] /= scale_factor
#     K[0, 2] /= scale_factor
#     K[1, 2] /= scale_factor

#     return K


# def triangulation(point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2):
#     """
#     Create a 3D Point CLoud using 2d point vecotr and projection matrices
#     """
#     pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T)

#     return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])


# def PnP(obj_point, image_point, K, dist_coeff, rot_vector, initial):
#     if initial == 1:
#         obj_point = obj_point[:, 0 ,:]
#         image_point = image_point.T
#         rot_vector = rot_vector.T 
#     _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)

#     # Converts a rotation matrix to a rotation vector or vice versa
#     rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

#     if inlier:
#         image_point = image_point[inlier[:, 0]]
#         obj_point = obj_point[inlier[:, 0]]
#         rot_vector = rot_vector[inlier[:, 0]]

#     return rot_matrix, tran_vector, image_point, obj_point, rot_vector


# def reprojection_error(obj_points, image_points, transform_matrix, K, homogenity):
#     rot_matrix = transform_matrix[:3, :3]
#     tran_vector = transform_matrix[:3, 3]
#     rot_vector, _ = cv2.Rodrigues(rot_matrix)
#     if homogenity == 1:
#         obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
#     image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
#     image_points_calc = np.float32(image_points_calc[:, 0, :])
#     total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
    
#     return total_error / len(image_points_calc), obj_points


# def optimal_reprojection_error(self, obj_points):
#         '''
#         calculates of the reprojection error during bundle adjustment
#         returns error 
#         '''
#         transform_matrix = obj_points[0:12].reshape((3,4))
#         K = obj_points[12:21].reshape((3,3))
#         rest = int(len(obj_points[21:]) * 0.4)
#         p = obj_points[21:21 + rest].reshape((2, int(rest/2))).T
#         obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:])/3), 3))
#         rot_matrix = transform_matrix[:3, :3]
#         tran_vector = transform_matrix[:3, 3]
#         rot_vector, _ = cv2.Rodrigues(rot_matrix)
#         image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
#         image_points = image_points[:, 0, :]
#         error = [ (p[idx] - image_points[idx])**2 for idx in range(len(p))]
#         return np.array(error).ravel()/len(p)


# def bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error):
#     '''
#     Bundle adjustment for the image and object points
#     returns object points, image points, transformation matrix
#     '''
#     opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
#     opt_variables = np.hstack((opt_variables, opt.ravel()))
#     opt_variables = np.hstack((opt_variables, _3d_point.ravel()))

#     values_corrected = least_squares(self.optimal_reprojection_error, opt_variables, gtol = r_error).x
#     K = values_corrected[12:21].reshape((3,3))
#     rest = int(len(values_corrected[21:]) * 0.4)
#     return values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:])/3), 3)), values_corrected[21:21 + rest].reshape((2, int(rest/2))).T, values_corrected[0:12].reshape((3,4))


# def downscale_img(img, scale_factor):
#     for _ in range(1, int(scale_factor/2)+1):
#         img = cv2.pyrDown(img)

#     return img


# def init(img_dir:str, scale_factor:float=1):
#     with open("K.txt", "r") as f:
#         lines = f.readlines()
#         K = np.float32([i.strip().split(" ") for i in lines])
#         print(K)
    
#     image_list = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]
#     print(image_list)
#     path = os.getcwd()
#     factor = scale_factor
#     downscale()
    

# def to_ply(path, point_clouds, colors):
#     out_points = point_clouds.reshape(-1, 3) * 200
#     out_colors = colors.reshape(-1, 3)
#     print(f"out_colors shape: {out_colors.shape}, out_points shape: {out_points.shape}")
#     verts = np.hstack([out_points, out_colors])

#     mean = np.mean(verts[:, :3], axis=0)
#     scaled_verts = verts[:, :3] - mean
#     dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
#     indx = np.where(dist < np.mean(dist) + 300)

#     verts = verts[indx]

# init("images")



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


def downscale_image(img, scale):
    for _ in range(1, int(scale / 2) + 1):
        img = cv2.pyrDown(img)
    
    return img


def triangulation(point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2):
    pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T)
    print(f"point cloud from triangulation:\n {pt_cloud}")
    return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])


def pnp(obj_point, img_point, K, dist_coeff, rot_vector, initial):
    if initial == 1:
        obj_point = obj_point[:, 0 ,:]
        image_point = image_point.T
        rot_vector = rot_vector.T 

    _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, img_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
    rot_matrix = cv2.Rodrigues(rot_vector_calc)

    if inlier:
        image_point = image_point[inlier[:, 0]]
        obj_point = obj_point[inlier[:, 0]]
        rot_vector = rot_vector[inlier[:, 0]]

    return rot_matrix, tran_vector, image_point, obj_point, rot_vector


def reprojection_error(obj_points, image_points, transform_matrix, K, homogenity) ->tuple:
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
   

def optimal_reprojection_error(self, obj_points) -> np.array:
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


def to_ply(path, point_clouds, colors):
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

image_loader("images")
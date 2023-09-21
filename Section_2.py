import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


"""
**Homogeneous representation of points.** A point x = (x, y)T lies on the line l =
(a, b, c)T if and only if ax + by + c = 0. This may be written in terms of an inner
product of vectors representing the point as (x, y, 1)(a, b, c)T = (x, y, 1)l = 0; that is
the point (x, y)T in IR2 is represented as a 3-vector by adding a final coordinate of 1.
Note that for any non-zero constant k and line l the equation (kx, ky, k)l = 0 if and
only if (x, y, 1)l = 0. It is natural, therefore, to consider the set of vectors (kx, ky, k)T
for varying values of k to be a representation of the point (x, y)T in IR2. Thus, just as
with lines, points are represented by homogeneous vectors. An arbitrary homogeneous
vector representative of a point is of the form x = (x1, x2, x3)T, representing the point
(x1/x3, x2/x3)T in IR2. Points, then, as homogeneous vectors are also elements of IP2.
One has a simple equation to determine when a point lies on a line, namely
Result 2.1. The point x lies on the line l if and only if xTl = 0.
"""


"""
Degrees of freedom (dof). It is clear that in order to specify a point two values must
be provided, namely its x- and y-coordinates. In a similar manner a line is specified
by two parameters (the two independent ratios {a : b : c}) and so has two degrees
of freedom. For example, in an inhomogeneous representation, these two parameters
could be chosen as the gradient and y intercept of the line.
Intersection of lines. Given two lines l = (a, b, c)T and l

Result 2.2. The intersection of two lines l and l
The intersection of two lines is the point x = l x l
"""

# Create Correspondences 
folder_path = r"images"
img_paths = [os.path.join(folder_path, i) for i in os.listdir(folder_path)]
img1 = cv2.imread(img_paths[0])
img2 = cv2.imread(img_paths[1])

W = int(img1.shape[1]/4)
H = int(img1.shape[0]/4)
img1 = cv2.resize(img1, (W,H))
img2 = cv2.resize(img2, (W,H))

# Blur the images
img1 = cv2.GaussianBlur(img1, (7,7), 0)
img2 = cv2.GaussianBlur(img2, (7,7), 0)

detector = cv2.SIFT_create()
keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
#Lowe's Ratio Test
good_matches = []
matches1_2 = bf.knnMatch(descriptors1, descriptors2, k=2)
for m,n in matches1_2[:10]:
    if m.distance < 0.55 * n.distance:
        good_matches.append(m)


# for match in good_matches[:5]:
#     print(f"""
# Distance : {match.distance}
# queryIdx = {match.queryIdx}
# trainIdx : {match.trainIdx}
# imgIdx : {match.imgIdx}

# """)

camera_matrix = np.array([
    [2759.48, 0, 1520.69],
    [0, 2764.16, 1006.81],
    [0, 0, 1]
])

matching_points1 = [keypoints1[m.queryIdx] for m in good_matches]
matching_points2 = [keypoints2[m.trainIdx] for m in good_matches]

pts1 = np.float32([i.pt for i in matching_points1])
pts2 = np.float32([i.pt for i in matching_points2])

#print(pts1.shape)
H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
print(f" {H}")

matched_image1_2 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None)


cv2.imshow("match", matched_image1_2)



# draw the matches 
#outImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#cv2.imshow("out", outImg)

# Create A matrix to find the Homography matrix

#cv2.imshow("example image1", img1)
#cv2.imshow("example image2", img2)

cv2.waitKey()
cv2.destroyAllWindows()



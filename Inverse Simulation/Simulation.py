import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
import json
import math
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment
with open("roi_config.json", "r") as f:
    rois = json.load(f)

with open("intrinsics_transition.json", "r") as f:
    camera_calib_data = json.load(f)

R_wc = np.array(camera_calib_data["R_wc"])
t_wc = np.array(camera_calib_data["t_wc"])
rvec_cw = np.array(camera_calib_data["rvec_cw"])
tvec_cw = np.array(camera_calib_data["tvec_cw"])
K = np.array(camera_calib_data["K"])
D = np.array(camera_calib_data["D"])

# The diameter of usable mirror. Given 1 inch mirror: 25.4mm. Clear aperture from spec sheet: 22.9mm.
# 3mm diameter beam. 22.9 - (3/2) = 21.4 mm
mirror_lengths = [21.4, 21.4, 21.4, 21.4]

# Set up the laser
laser_start = (0, 100)
laser_angle = 0  # Initial laser angle in degrees

#Quad Cell Locations
qc_1 = np.array([-100, 137])
qc_2 = np.array([-300, 190])

THRESHOLD = 200     # Pixel intensity threshold for reflection point detection
EPS = 7.0           # DBSCAN groups pixels that are within EPS pixels of each other
MIN_SEP = 15        # minimum separation threshold to separate an ambiguous refl pt into two refl pts

lsr_height = 4.087 # inches

EXIT_TARGET = -0.265    # aligned exit angle
SIGMA_PX = 3            # px (tune)
SIGMA_EXIT = 8          # units of simulation_identifier (tune)
SIGMA_REFL = 3          # px (tune)

BIG_PEN = 50.0     # px penalty converted to residual via /SIGMA_REFL

# M1y, M2y, M3y, M4y = 109, 73, 69, 120 # simulation units (mm)

# Function to calculate the endpoints of a mirror given center, length, and angle
def calculate_mirror_endpoints(center, length, angle):
    half_length = length / 2
    angle_rad = np.radians(angle)
    start = (
        center[0] - half_length * np.cos(angle_rad),
        center[1] - half_length * np.sin(angle_rad),
    )
    end = (
        center[0] + half_length * np.cos(angle_rad),
        center[1] + half_length * np.sin(angle_rad),
    )
    return start, end

# Function to find the intersection of two lines
def find_intersection(p1, p2, p3, p4, eps=1e-9):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    r = np.array([x2 - x1, y2 - y1], dtype=float)
    s = np.array([x4 - x3, y4 - y3], dtype=float)

    rxs = r[0]*s[1] - r[1]*s[0]
    if abs(rxs) < eps:
        return None  # parallel

    qp = np.array([x3 - x1, y3 - y1], dtype=float)

    t = (qp[0]*s[1] - qp[1]*s[0]) / rxs
    u = (qp[0]*r[1] - qp[1]*r[0]) / rxs

    if t >= -eps:
        return (x1 + t*r[0], y1 + t*r[1]), u

    return None

# Function to calculate the reflection of a laser beam
# This is used for the optimization
def reflect_laser_ordered(laser_start, laser_angle, mirror):
    laser_angle_rad = np.radians(laser_angle)
    laser_far_end = (
        laser_start[0] + np.cos(laser_angle_rad) * 1000,
        laser_start[1] + np.sin(laser_angle_rad) * 1000,
    )

    mirror_start, mirror_end = mirror

    result = find_intersection(
        laser_start, laser_far_end,
        mirror_start, mirror_end
    )

    if result is None:
        return None, None, False

    intersection, u = result

    # Determine if inside segment
    inside = (0 <= u <= 1)

    # Reflection math (same as your original)
    mirror_vector = np.array([
        mirror_end[0] - mirror_start[0],
        mirror_end[1] - mirror_start[1]
    ])
    mirror_unit = mirror_vector / np.linalg.norm(mirror_vector)
    normal_vector = np.array([-mirror_unit[1], mirror_unit[0]])

    incident_vector = np.array([
        intersection[0] - laser_start[0],
        intersection[1] - laser_start[1]
    ])

    reflection_vector = (
        incident_vector - 2 * np.dot(incident_vector, normal_vector) * normal_vector
    )

    reflected_end = (
        intersection[0] + reflection_vector[0],
        intersection[1] + reflection_vector[1]
    )

    return intersection, reflected_end, inside

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Simulate laser reflections with length calculation
# Gives us the laser path and total laser length
def simulate_laser_with_length(laser_start, laser_angle, mirrors, max_reflections=36, exit_dist=1000):
    current_position = laser_start
    current_angle = laser_angle
    laser_path = [laser_start]

    mirror_index = 0  # M1 first

    for _ in range(max_reflections):
        intersection, reflected_end, inside = reflect_laser_ordered(
            current_position,
            current_angle,
            mirrors[mirror_index]
        )

        # If no intersection with mirror line (rare degeneracy): exit
        if intersection is None:
            laser_far_end = (
                current_position[0] + np.cos(np.radians(current_angle)) * exit_dist,
                current_position[1] + np.sin(np.radians(current_angle)) * exit_dist,
            )
            laser_path.append(laser_far_end)
            break

        # If it intersects the infinite line but misses the segment: exit (do NOT append intersection)
        if not inside:
            laser_far_end = (
                current_position[0] + np.cos(np.radians(current_angle)) * exit_dist,
                current_position[1] + np.sin(np.radians(current_angle)) * exit_dist,
            )
            laser_path.append(laser_far_end)
            break

        # Otherwise: valid reflection, keep it
        laser_path.append(intersection)

        # Update ray
        current_position = intersection
        current_angle = np.degrees(np.arctan2(
            reflected_end[1] - intersection[1],
            reflected_end[0] - intersection[0],
        ))

        mirror_index = (mirror_index + 1) % len(mirrors)

    # Total path length (sum all segments in laser_path)
    total_length = sum(
        calculate_distance(laser_path[i], laser_path[i + 1])
        for i in range(len(laser_path) - 1)
    )

    return laser_path, total_length

def extend_line(p1, p2):
    # Calculate the length of the line
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Extend the line in both directions
    new_p1 = (p1[0] - 0.73*dx, p1[1] - 0.73*dy)  # Extend p1 backwards
    new_p2 = (p2[0] + 0.73*dx, p2[1] + 0.73*dy)  # Extend p2 forwards
    return new_p1, new_p2

def create_orthogonal_line_at_endpoint(endpoint, other_endpoint, length=44):
    """Create an orthogonal line of the specified length at a given endpoint."""
    # Calculate the direction vector of the original line
    dx = other_endpoint[0] - endpoint[0]
    dy = other_endpoint[1] - endpoint[1]
    
    # Get orthogonal direction
    orthogonal_dx = -dy
    orthogonal_dy = dx
    magnitude = np.sqrt(orthogonal_dx**2 + orthogonal_dy**2)
    unit_dx = orthogonal_dx / magnitude
    unit_dy = orthogonal_dy / magnitude

    # Compute the two endpoints of the orthogonal line
    ortho_p1 = (endpoint[0] + unit_dx * length, endpoint[1] + unit_dy * length)
    ortho_p2 = (endpoint[0] - unit_dx * length, endpoint[1] - unit_dy * length)
    return ortho_p1, ortho_p2

def select_furthest_orthogonal_line(endpoint, ortho_p1, ortho_p2, reference_x=100):
    """Select the orthogonal line endpoint furthest away from reference_x."""
    # Calculate distances from reference_x for each orthogonal endpoint
    dist_ortho_p1 = abs(ortho_p1[0] - reference_x)
    dist_ortho_p2 = abs(ortho_p2[0] - reference_x)
    
    # Return the endpoint further from reference_x
    if dist_ortho_p1 > dist_ortho_p2:
        return (endpoint, ortho_p1)
    else:
        return (endpoint, ortho_p2)

def process_mirrors(mirrors):
    doubled_lines = []
    orthogonal_lines = []
    
    for p1, p2 in mirrors:
        # Double the length of the original line
        extended_p1, extended_p2 = extend_line(p1, p2)
        doubled_lines.append((extended_p1, extended_p2))

        # Create orthogonal lines at the endpoints of the doubled line
        ortho_p1_a, ortho_p1_b = create_orthogonal_line_at_endpoint(extended_p1, extended_p2)
        ortho_p2_a, ortho_p2_b = create_orthogonal_line_at_endpoint(extended_p2, extended_p1)
        
        # Select only the orthogonal line furthest from x=100
        orthogonal_lines.append(select_furthest_orthogonal_line(extended_p1, ortho_p1_a, ortho_p1_b))
        orthogonal_lines.append(select_furthest_orthogonal_line(extended_p2, ortho_p2_a, ortho_p2_b))

    return doubled_lines, orthogonal_lines

def simulation(m1cx, m1cy, m2cx, m2cy, m3cx, m3cy, m4cx, m4cy, m1a, m2a, m3a, m4a):

    mirrors = []

    # MIRROR CONFIGURATION
    mirror_centers = [(m1cx, m1cy), (m2cx, m2cy), (m3cx, m3cy), (m4cx, m4cy)]
    mirror_angles = [m1a, m2a, m3a, m4a]  # degrees

    for center, length, angle in zip(mirror_centers, mirror_lengths, mirror_angles):
        mirrors.append(calculate_mirror_endpoints(center, length, angle))

    # Initialize plot
    plt.figure(figsize=(12, 10))
    plt.scatter(*laser_start, color='red', label="Laser Source", linewidth=1)

    # Piezo mount outline visualizer
    doubled_lines, orthogonal_lines = process_mirrors(mirrors)

    # Draw mirrors
    for mirror in mirrors:
        plt.plot([mirror[0][0], mirror[1][0]],
                 [mirror[0][1], mirror[1][1]],
                 color='black', linewidth=3)

    # Draw mirror mount outlines
    for mirror in doubled_lines:
        plt.plot([mirror[0][0], mirror[1][0]],
                 [mirror[0][1], mirror[1][1]],
                 linewidth=1, color='black')

    for mirror in orthogonal_lines:
        plt.plot([mirror[0][0], mirror[1][0]],
                 [mirror[0][1], mirror[1][1]],
                 linewidth=1, color='black')

    # --- Laser simulation ---
    max_reflections = 36
    current_position = laser_start
    current_angle = laser_angle
    reflection_count = 0
    mirror_index = 0  # start with M1

    for i in range(max_reflections):

        intersection, reflected_end, inside = reflect_laser_ordered(
            current_position,
            current_angle,
            mirrors[mirror_index])

        # If ray never even intersects mirror plane
        if intersection is None:
            plt.plot(
                [current_position[0],
                 current_position[0] + np.cos(np.radians(current_angle)) * 1000],
                [current_position[1],
                 current_position[1] + np.sin(np.radians(current_angle)) * 1000],
                'g--')
            break

        # If intersection exists but outside mirror segment → beam exits system
        if not inside:
            plt.plot(
                [current_position[0],
                 current_position[0] + np.cos(np.radians(current_angle)) * 1000],
                [current_position[1],
                 current_position[1] + np.sin(np.radians(current_angle)) * 1000],
                'g--')
            break

        # Valid reflection
        plt.plot(
            [current_position[0], intersection[0]],
            [current_position[1], intersection[1]],
            'r-', linewidth=1)

        # Update ray state
        current_position = intersection
        current_angle = np.degrees(np.arctan2(
            reflected_end[1] - intersection[1],
            reflected_end[0] - intersection[0]))

        reflection_count += 1

        # Move to next mirror (M1→M2→M3→M4→repeat)
        mirror_index = (mirror_index + 1) % len(mirrors)

    # Compute full path + length
    laser_path, total_length = simulate_laser_with_length(
        laser_start, laser_angle, mirrors)

    # --- Exit distance calculation ---
    a = laser_path[-2]
    b = laser_path[-1]
    x = 0

    slope = (b[1] - a[1]) / (b[0] - a[0])
    y = a[1] + slope * (x - a[0])
    x_point = (x, y)

    distance = np.sqrt((x_point[0] - a[0])**2 + (x_point[1] - a[1])**2)

    # --- Check clipping with M4 ---
    a = np.array(a)
    b = np.array(b)
    m = np.array([m4cx, m4cy])

    v = b - a
    d = np.array([np.cos(np.deg2rad(m4a)), np.sin(np.deg2rad(m4a))])

    A = np.column_stack((v, -d))
    t, s = np.linalg.solve(A, m - a)

    if 0 <= t <= 1:
        p = a + t * v
        dist = np.linalg.norm(p - m)
    else:
        print("Beam does not intersect M4 region")
        dist = np.inf

    if dist >= 14.3:
        print("NOT CLIPPED, room to spare:", dist - 14.3, "mm")
    else:
        print("CLIPPED,", dist - 14.3, "mm too much")

    print("Laser Path:", laser_path)
    print("Total Laser Length:", total_length + distance, "mm")
    print("Total Number of Reflection (N_R) =", reflection_count)

    # Plot settings
    plt.xlim(-310, 250)
    plt.ylim(-10, 210)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Laser Reflection with Multiple Mirrors")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.grid(True, linewidth=0.3)
    plt.plot([qc_1[0], qc_1[0]], [qc_1[1] - 2, qc_1[1] + 2], linewidth=4, label='QC1')
    plt.plot([qc_2[0], qc_2[0]], [qc_2[1] - 2, qc_2[1] + 2], linewidth=4, label='QC2')
    plt.legend(prop={'size': 8})
    plt.show()

def simulation_reflec(m1cx, m1cy, m2cx, m2cy, m3cx, m3cy, m4cx, m4cy,
    m1a, m2a, m3a, m4a, expected_reflections=7
):
    mirror_centers = [(m1cx, m1cy), (m2cx, m2cy),
                      (m3cx, m3cy), (m4cx, m4cy)]
    mirror_angles = [m1a, m2a, m3a, m4a]

    mirrors = []
    for center, length, angle in zip(
        mirror_centers, mirror_lengths, mirror_angles
    ):
        mirrors.append(calculate_mirror_endpoints(center, length, angle))

    current_position = laser_start
    current_angle = laser_angle

    path = []

    for i in range(expected_reflections):
        mirror_index = i % 4
        mirror = mirrors[mirror_index]

        intersection, reflected_end, inside = reflect_laser_ordered(
            current_position, current_angle, mirror
        )

        if intersection is None:
            # rare degeneracy fallback
            path.append({
                "pt": None,
                "mirror": mirror_index,
                "inside": False
            })
            break

        path.append({
            "pt": intersection,
            "mirror": mirror_index,
            "inside": inside
        })

        current_position = intersection
        current_angle = np.degrees(np.arctan2(
            reflected_end[1] - intersection[1],
            reflected_end[0] - intersection[0],
        ))

    return path

# Tells us the exit angle, total laser length, and quadcell displacement
def simulation_identifier(m1cx, m1cy, m2cx, m2cy, m3cx, m3cy, m4cx, m4cy, m1a, m2a, m3a, m4a):
    mirrors = []

    # MIRROR CONFIGURATION
    mirror_centers = [(m1cx, m1cy), (m2cx, m2cy), (m3cx, m3cy), (m4cx, m4cy)]
    mirror_angles = [m1a, m2a, m3a, m4a] #in degrees

    for center, length, angle in zip(mirror_centers, mirror_lengths, mirror_angles):
        mirrors.append(calculate_mirror_endpoints(center, length, angle))

    laser_path, total_length = simulate_laser_with_length(laser_start, laser_angle, mirrors)

    #Indicate where to cut off laser distance calculation (x=?)
    a = np.array(laser_path[-2])
    b = np.array(laser_path[-1])

    dx = b[0] - a[0]
    dy = b[1] - a[1]

    # exit slope
    if abs(dx) < 1e-9:
        exit_slope = np.inf
    else:
        exit_slope = dy / dx

    # intercept
    y_int = a[1] - exit_slope * a[0]

    # cut-off point at x=0
    x = 0
    y = exit_slope * x + y_int
    x_point = np.array([x, y])

    distance = np.linalg.norm(x_point - a)

    # evaluate beam height at diagnostic x positions
    def y_at(x):
        return exit_slope * x + y_int

    y191 = y_at(-191)
    y200 = y_at(-200)
    y300 = y_at(-300)
    y595 = y_at(-595)

    print(f"Exit slope: {exit_slope}")
    print(f"Total length: {total_length + distance}")
    print(f"y191 error: {y191 - 160}")
    print(f"y200 error: {y200 - 163.5}")
    print(f"y300 error: {y300 - 190}")
    print(f"y595 error: {y595 - 268.3}")

    return (
        exit_slope,
        total_length + distance,
        y191 - 160,
        y300 - 190,
        y595 - 268.3
    )

# TRANSITION FUNCTIONS

# Given a pixel coordinate and its known height (u,v,H_in), this function returns the real-life coordinates (inches)
def pixel_to_world_on_plane(u, v, H_in=0.0, override_cam_height=None):
    pts = np.array([[[u, v]]], dtype=np.float64)
    rays_norm = cv.fisheye.undistortPoints(pts, K, D)  # pinhole model
    x, y = rays_norm[0,0]
    d_cam = np.array([x, y, 1.0], dtype=np.float64)

    # normalize direction
    d_cam /= np.linalg.norm(d_cam)

    d_w = R_wc @ d_cam

    C_w = t_wc.reshape(3).copy()
    if override_cam_height is not None:
        C_w[2] = float(override_cam_height)

    lam = (H_in - C_w[2]) / d_w[2]
    Pw = C_w + lam * d_w
    return float(Pw[0]), float(Pw[1])

# This is the opposite of pixel_to_world_on_plane.
# Given a real-life coordinate point (inches), this function returns the corresponding pixel coordinate
def world_to_pixel(X, Y, Z):
    obj = np.array([[[X, Y, Z]]], dtype=np.float64)  # (1,1,3)
    img_proj, _ = cv.fisheye.projectPoints(obj, rvec_cw, tvec_cw, K, D)
    u, v = img_proj.reshape(2)
    return float(u), float(v)

# ArUcos

# Returns the pixel coordinates of the detected ArUco points
def camera_arucos(img_path):
    # --- Config ---
    dict_name = "DICT_4X4_100"
    allowed_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
    resize_factor = 1.0  # Set this to 1.0 to match the original's actual run
    
    img_bgr = cv.imread(img_path)
    if img_bgr is None: return []
    
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    if resize_factor != 1.0:
        gray = cv.resize(gray, None, fx=resize_factor, fy=resize_factor, interpolation=cv.INTER_CUBIC)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    aruco_dict = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, dict_name))
    try:
        params = cv.aruco.DetectorParameters()
    except AttributeError:
        params = cv.aruco.DetectorParameters_create()

    # --- THE MISSING CRITICAL PARAMS ---
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 3  # <--- Crucial for detection density
    params.minMarkerPerimeterRate = 0.01   # <--- Crucial for small markers
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX

    try:
        detector = cv.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray_eq)
    except AttributeError:
        corners, ids, _ = cv.aruco.detectMarkers(gray_eq, aruco_dict, parameters=params)

    # Temporary list to hold (id, (x, y)) tuples
    found_markers = []
    
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            marker_id = int(i)
            if marker_id in allowed_ids:
                pts = c.reshape(4, 2)
                center = pts.mean(axis=0)
                
                # Scale back to original resolution
                if resize_factor != 1.0:
                    center = center / resize_factor
                
                found_markers.append((marker_id, tuple(center)))

    # --- Sorting Logic ---
    # Sort by the first element of the tuple (the ID)
    found_markers.sort(key=lambda x: x[0])

    # Extract only the coordinates from the sorted list
    sorted_centers = [coords for marker_id, coords in found_markers]

    return sorted_centers

# LASER REFLECTION POINTS

# Performs Principal Component Analysis (PCA) to distinguish laser reflection points that are elliptical
def pca_elongation(points_xy):
    """
    points_xy: (N,2) array of [x,y] in patch coords.
    returns (ratio, major_sigma, minor_sigma, angle_rad)
    """
    pts = points_xy.astype(float)
    pts -= pts.mean(axis=0, keepdims=True)

    C = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(C)          # vals sorted ascending
    minor, major = np.sqrt(vals[0] + 1e-9), np.sqrt(vals[1] + 1e-9)
    ratio = major / minor
    angle = np.arctan2(vecs[1,1], vecs[0,1])  # direction of major axis
    return ratio, major, minor, angle


def split_cluster_k2(points_xy, n_iter=20):
    """
    Very small k-means for k=2 on points_xy.
    Returns centers (2,2) in patch coords.
    """
    pts = points_xy.astype(float)

    # init: pick two farthest points (good for peanuts)
    d2 = ((pts[:,None,:] - pts[None,:,:])**2).sum(axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    c1, c2 = pts[i].copy(), pts[j].copy()

    for _ in range(n_iter):
        d1 = ((pts - c1)**2).sum(axis=1)
        d2 = ((pts - c2)**2).sum(axis=1)
        m1 = d1 <= d2
        m2 = ~m1
        if m1.sum() == 0 or m2.sum() == 0:
            break
        new_c1 = pts[m1].mean(axis=0)
        new_c2 = pts[m2].mean(axis=0)
        if np.allclose(new_c1, c1) and np.allclose(new_c2, c2):
            break
        c1, c2 = new_c1, new_c2

    return np.vstack([c1, c2])

def postprocess_split_peanuts(clusters, radius_split=50.0, elong_split=5, min_sep=MIN_SEP):
    """
    Splits clusters that look like two touching spots.
    Returns a new cluster list (some clusters replaced by two subclusters).
    """
    new_clusters = []
    next_label = 1000  # labels for split children

    for c in clusters:
        pts = np.array(c["points"], dtype=float)   # patch coords [x,y]
        if len(pts) < 20:
            new_clusters.append(c)
            continue

        ratio, major, minor, _ = pca_elongation(pts)

        # decide whether to split
        if (c["radius"] > radius_split) or (ratio > elong_split):
            centers2 = split_cluster_k2(pts)

            # reject split if the two centers are basically on top of each other
            dcent = np.linalg.norm(centers2[0] - centers2[1])
            print("split candidate center distance:", dcent, "min_sep:", min_sep)
            if np.linalg.norm(centers2[0] - centers2[1]) < min_sep:
                new_clusters.append(c)
                continue

            # build two child clusters based on assignment
            d1 = ((pts - centers2[0])**2).sum(axis=1)
            d2 = ((pts - centers2[1])**2).sum(axis=1)
            m1 = d1 <= d2
            m2 = ~m1

            for m, center in [(m1, centers2[0]), (m2, centers2[1])]:
                sub_pts = pts[m]
                if len(sub_pts) < 5:
                    continue
                dist = np.linalg.norm(sub_pts - center, axis=1)
                radius = float(dist.max())
                x_min, y_min = np.min(sub_pts, axis=0)
                x_max, y_max = np.max(sub_pts, axis=0)

                new_clusters.append({
                    **c,
                    "label": int(next_label),
                    "center": center.tolist(),
                    "radius": radius,
                    "size": int(len(sub_pts)),
                    "points": sub_pts.tolist(),
                    "bbox": [float(x_min), float(x_max), float(y_min), float(y_max)],
                    "density": float(len(sub_pts) / (np.pi * radius**2)) if radius > 0 else 0.0,
                    "was_split": True,
                })
                next_label += 1
        else:
            new_clusters.append(c)

    # sort biggest first like you already do
    new_clusters.sort(key=lambda x: x["size"], reverse=True)
    return new_clusters

def find_clusters_with_circles(patch, threshold=THRESHOLD, eps=EPS, min_samples=50, show=True, title=""):
    y_coords, x_coords = np.where(patch > threshold)

    if len(x_coords) == 0:
        if show:
            print("No points above threshold!")
        return []

    coordinates = np.column_stack([x_coords, y_coords])

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(coordinates)

    unique_labels = set(labels)
    clusters = []

    for label in unique_labels:
        if label == -1:
            continue
        mask = labels == label
        cluster_points = coordinates[mask]

        center = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - center, axis=1)
        radius = np.max(distances)

        x_min, y_min = np.min(cluster_points, axis=0)
        x_max, y_max = np.max(cluster_points, axis=0)

        clusters.append({
            'label': int(label),
            'center': center.tolist(),      # [x, y] in PATCH coords
            'radius': float(radius),
            'size': int(len(cluster_points)),
            'points': cluster_points.tolist(),
            'bbox': [float(x_min), float(x_max), float(y_min), float(y_max)],
            'density': float(len(cluster_points) / (np.pi * radius**2)) if radius > 0 else 0.0
        })

    clusters.sort(key=lambda x: x['size'], reverse=True)

    clusters = postprocess_split_peanuts(clusters, radius_split=30.0, elong_split=2, min_sep=MIN_SEP)

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        axes[0].imshow(patch, cmap='gray')
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(clusters), 1)))

        for i, cluster in enumerate(clusters):
            color = colors[i]
            center = cluster['center']
            radius = cluster['radius']

            circle = plt.Circle(center, radius, color=color, fill=False, linewidth=2, alpha=0.7)
            axes[0].add_patch(circle)
            axes[0].scatter(center[0], center[1], color=color, s=100, marker='x', linewidths=2)
            axes[0].text(center[0], center[1], f'C{i}', color='white', fontsize=12, fontweight='bold',
                         ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))

        axes[0].set_title(title or f'Found {len(clusters)} Cluster(s)')
        axes[0].axis('equal')

        axes[1].axis('off')
        if clusters:
            summary = "CLUSTER CENTERS:\n\n"
            for i, cluster in enumerate(clusters):
                summary += f"Cluster {i} (Label {cluster['label']}):\n"
                summary += f"  Center: ({cluster['center'][0]:.1f}, {cluster['center'][1]:.1f})\n"
                summary += f"  Radius: {cluster['radius']:.1f} px\n"
                summary += f"  Size: {cluster['size']} points\n"
                summary += f"  Density: {cluster['density']:.3f} pts/px²\n"
                summary += f"  BBox: [{cluster['bbox'][0]:.0f}-{cluster['bbox'][1]:.0f}, {cluster['bbox'][2]:.0f}-{cluster['bbox'][3]:.0f}]\n\n"
            axes[1].text(0.05, 0.95, summary, fontfamily='monospace',
                         verticalalignment='top', fontsize=10)

        plt.tight_layout()
        plt.show()

    return clusters


def clusters_in_roi(gray, roi, threshold=THRESHOLD, eps=EPS, min_samples=35, show=True):
    x1, y1, x2, y2 = roi
    patch = gray[y1:y2, x1:x2]

    clusters = find_clusters_with_circles(
        patch, threshold=threshold, eps=eps, min_samples=min_samples,
        show=show, title=f"ROI {roi} | threshold={threshold}, eps={eps}"
    )

    # Add full-image centers to each cluster dict
    for c in clusters:
        cx, cy = c["center"]  # patch coords
        c["center_full"] = [cx + x1, cy + y1]

    return clusters

def process_all_rois(gray_img, rois, threshold, eps=EPS, min_samples=35, show=False):
    results = {}
    for name, roi in rois.items():
        clusters = clusters_in_roi(
            gray_img, roi,
            threshold=threshold,
            eps=eps,
            min_samples=min_samples,
            show=show
        )
        results[name] = clusters
    return results

def reflec_pts_cam(gray_img, eps=EPS, min_samples=35, show=False):
    all_clusters = process_all_rois(
        gray_img,
        rois=rois,
        threshold=THRESHOLD,
        eps=eps,
        min_samples=min_samples,
        show=show
    )

    grouped = {k: [] for k in rois.keys()}

    # ---- Collect and group centers ----
    for clusters in all_clusters.values():
        for c in clusters:
            x, y = c["center_full"]
            for name, (x0, y0, x1, y1) in rois.items():
                if x0 <= x <= x1 and y0 <= y <= y1:
                    grouped[name].append([float(x), float(y)])
                    break

    # ---- Enforce expected reflection-count pattern ----
    # M1, M2, M3 have same count; M4 has one less
    base = max(len(grouped.get("M1", [])),
               len(grouped.get("M2", [])),
               len(grouped.get("M3", [])))

    expected = {
        "M1": base,
        "M2": base,
        "M3": base,
        "M4": max(0, base - 1),
    }

    for name, need in expected.items():
        pts = grouped[name]
        if len(pts) == 0:
            # If nothing detected, you can either leave empty or insert a dummy.
            # I'd leave empty so you notice it.
            continue

        # Duplicate last point until count matches expected
        while len(pts) < need:
            pts.append(pts[-1])

        # If too many, trim extras (keeps your residual length stable)
        if len(pts) > need:
            grouped[name] = pts[:need]

    return grouped

# Inverse Problem

def sim_to_pt(loc_x, loc_y):
    # Calibration constants from your original function
    calib_irl = [-2.65720102, -0.94574237]
    calib_sim = [-160, -109]

    # 1. Reverse the negation
    # Since: loc_x = -(diff_x + calib_sim[0])
    # Then:  -loc_x - calib_sim[0] = diff_x
    diff_x = -loc_x - calib_sim[0]
    diff_y = -loc_y - calib_sim[1]

    # 2. Reverse the scaling (25.4) and the IRL offset subtraction
    # Since: diff_x = (x - calib_irl[0]) * 25.4
    # Then:  x = (diff_x / 25.4) + calib_irl[0]
    x = (diff_x / 25.4) + calib_irl[0]
    y = (diff_y / 25.4) + calib_irl[1]

    return x, y

def get_mount_corners(x, y, z, theta_deg, 
                      s_half=1.3/2, 
                      shift_dist=0.045):
    """
    Calculates the 3 corners of a mirror mount given the center (x,y,z) 
    and rotation theta.

    Parameters
    ----------
    s_half : float
        Half side length of mirror face.
        Default = 1.3/2 (standard mirrors).
    shift_dist : float
        Distance (in inches) to shift center along mirror normal toward origin.
        Default = 0.045. Set to 0 for mirrors that do not require shift.
    """

    # --------------------------
    # 0. Shift center along normal toward origin (if shift_dist > 0)
    # --------------------------
    theta = np.radians(theta_deg)
    center = np.array([float(x), float(y), float(z)])

    # Mirror in-plane direction (x-y plane)
    u = np.array([np.cos(theta), np.sin(theta), 0.0])

    # Normal to mirror face (perpendicular in x-y plane)
    n = np.array([-np.sin(theta), np.cos(theta), 0.0])
    n = n / np.linalg.norm(n)

    if shift_dist != 0:
        if np.linalg.norm(center - shift_dist*n) < np.linalg.norm(center + shift_dist*n):
            center = center - shift_dist*n
        else:
            center = center + shift_dist*n

    # --------------------------
    # 1. Define mirror geometry
    # --------------------------
    v = np.array([0.0, 0.0, 1.0])

    corners = [
        center + s_half*u + s_half*v,
        center + s_half*u - s_half*v,
        center - s_half*u + s_half*v,
        center - s_half*u - s_half*v
    ]

    # --------------------------
    # 2. Quadrant-based filtering
    # --------------------------
    if x < 0 and y < 0:
        ref = np.array([-3.0, 0.0, 5.0])
        config = {'first': (2, True), 'third': (1, True)}
    elif x < 0 and y >= 0:
        ref = np.array([-4.0, 0.0, 0.0])
        config = {'first': (1, False), 'third': (2, False)}
    elif x >= 0 and y >= 0:
        ref = np.array([4.0, 0.0, 0.0])
        config = {'first': (2, False), 'third': (1, False)}
    else:
        ref = np.array([3.0, 0.0, 5.0])
        config = {'first': (1, True), 'third': (2, True)}

    distances = [np.linalg.norm(c - ref) for c in corners]
    discard_idx = np.argmin(distances)
    remaining = [corners[i] for i in range(4) if i != discard_idx]

    idx_f, rev_f = config['first']
    out1 = sorted(remaining, key=lambda c: c[idx_f], reverse=rev_f)[0]

    others = [c for c in remaining if not np.array_equal(c, out1)]
    idx_t, rev_t = config['third']
    out3 = sorted(others, key=lambda c: c[idx_t], reverse=rev_t)[0]

    out2 = [c for c in others if not np.array_equal(c, out3)][0]

    return [out1, out2, out3]

# THE OPTIMIZATION PROCESS

def sim_to_px_reflec(x, y): # For reflection points
    sim_M_IRL = sim_to_pt(x, y)
    pixel_point = world_to_pixel(sim_M_IRL[0], sim_M_IRL[1], lsr_height)
    return pixel_point

def sim_to_px(x, y, a, mirror_id):  # For ArUcos
    sim_M_IRL = sim_to_pt(x, y)

    Xw = sim_M_IRL[0]
    Yw = sim_M_IRL[1]

    # -----------------------------------
    # Mirror-specific geometry selection
    # -----------------------------------
    if mirror_id == 4:
        s_half = 1.49 / 2
        shift_dist = 0.0
    else:
        s_half = 1.3 / 2
        shift_dist = 0.045

    sim_M_corners = get_mount_corners(
        Xw, Yw, lsr_height, a,
        s_half=s_half,
        shift_dist=shift_dist
    )

    sim_M_corner_1 = world_to_pixel(*sim_M_corners[0])
    sim_M_corner_2 = world_to_pixel(*sim_M_corners[1])
    sim_M_corner_3 = world_to_pixel(*sim_M_corners[2])

    return sim_M_corner_1, sim_M_corner_2, sim_M_corner_3

def aruco_pixel_residuals(theta, img_path):
    # ---- cache camera ArUco detection by image path ----
    if not hasattr(aruco_pixel_residuals, "_aruco_cache"):
        aruco_pixel_residuals._aruco_cache = {}

    if img_path not in aruco_pixel_residuals._aruco_cache:
        aruco_pixel_residuals._aruco_cache[img_path] = camera_arucos(img_path)

    camera_aruco_coords = aruco_pixel_residuals._aruco_cache[img_path]
    # ----------------------------------------------------

    M1x, M2x, M3x, M4x, M1y, M2y, M3y, M4y, M1a, M2a, M3a, M4a = theta

    M1_px = sim_to_px(M1x, M1y, M1a, mirror_id=1)
    M2_px = sim_to_px(M2x, M2y, M2a, mirror_id=2)
    M3_px = sim_to_px(M3x, M3y, M3a, mirror_id=3)
    M4_px = sim_to_px(M4x, M4y, M4a, mirror_id=4)

    M_all_px = np.array([M1_px, M2_px, M3_px, M4_px]).reshape(-1, 2)

    residuals = (M_all_px - camera_aruco_coords).reshape(-1)
    return residuals

# The residuals (differences) between the measured and simulated components (ArUcos, Reflection points, ...)
def residuals(theta, img_path_light, reflec_cam):

    M1x, M2x, M3x, M4x, M1y, M2y, M3y, M4y, M1a, M2a, M3a, M4a = theta

    # ---- ArUco residuals ----
    r_aruco_px = aruco_pixel_residuals(theta, img_path_light)
    r_aruco = r_aruco_px / SIGMA_PX

    # ---- Exit residual ----
    g = simulation_identifier(M1x, M1y, M2x, M2y, M3x, M3y, M4x, M4y,
        M1a, M2a, M3a, M4a)

    r_exit_angle  = np.array([(g[0] - EXIT_TARGET) / SIGMA_EXIT], dtype=float)
    r_exit_height = np.array([(g[2]) / SIGMA_EXIT], dtype=float)

    # ---- Reflection simulation (structured) ----
    refl_sim = simulation_reflec(M1x, M1y, M2x, M2y, M3x, M3y, M4x, M4y,
        M1a, M2a, M3a, M4a)

    mirror_centers_world = [(M1x, M1y), (M2x, M2y), (M3x, M3y), (M4x, M4y),]

    r_refl = []

    for mirror_index, name in enumerate(["M1", "M2", "M3", "M4"]):

        meas_pts = reflec_cam[name]

        sim_for_mirror = [
            rec for rec in refl_sim
            if rec["mirror"] == mirror_index]

        half_length = mirror_lengths[mirror_index] / 2
        cx, cy = mirror_centers_world[mirror_index]

        residuals_mirror = []

        if len(meas_pts) > 0:

            # Convert sim points to pixel for matching
            sim_pts_px = []
            inside_flags = []

            for rec in sim_for_mirror:
                if rec["pt"] is None:
                    sim_pts_px.append([np.nan, np.nan])
                    inside_flags.append(False)
                else:
                    u, v = sim_to_px_reflec(*rec["pt"])
                    sim_pts_px.append([u, v])
                    inside_flags.append(rec["inside"])

            sim_pts_px = np.asarray(sim_pts_px, float)
            meas_pts   = np.asarray(meas_pts, float)

            # Hungarian matching
            if len(meas_pts) > 1:
                dists = np.linalg.norm(
                    meas_pts[:, None, :] - sim_pts_px[None, :, :],
                    axis=2
                )
                row_ind, col_ind = linear_sum_assignment(dists)
            else:
                row_ind = np.array([0])
                col_ind = np.array([0])

            for r_idx, s_idx in zip(row_ind, col_ind):

                if inside_flags[s_idx]:
                    du = meas_pts[r_idx, 0] - sim_pts_px[s_idx, 0]
                    dv = meas_pts[r_idx, 1] - sim_pts_px[s_idx, 1]
                    residuals_mirror.extend([du / SIGMA_REFL,
                                             dv / SIGMA_REFL])
                else:
                    # Smooth world-space miss penalty
                    xw, yw = sim_for_mirror[s_idx]["pt"]
                    dx = xw - cx
                    dy = yw - cy
                    r = np.sqrt(dx*dx + dy*dy)
                    overshoot = max(0.0, r - half_length)
                    residuals_mirror.append(overshoot / SIGMA_REFL)

        r_refl.append(np.asarray(residuals_mirror, float))

    r_refl_pts = np.concatenate(r_refl) if r_refl else np.array([])

    return np.concatenate([r_aruco, r_exit_angle, r_exit_height, r_refl_pts])

# OVERLAYING SIMULATED MEASUREMENTS OVER THE ACTUAL MEASUREMENTS

def group_aruco_centers_by_mirror(centers12):
    """
    centers12: list of 12 (x,y) points sorted by ArUco ID (0..11)
    Assumes 3 per mirror in order: M1(0-2), M2(3-5), M3(6-8), M4(9-11)
    """
    centers12 = list(centers12) if centers12 is not None else []
    out = {"M1": [], "M2": [], "M3": [], "M4": []}
    if len(centers12) >= 12:
        out["M1"] = centers12[0:3]
        out["M2"] = centers12[3:6]
        out["M3"] = centers12[6:9]
        out["M4"] = centers12[9:12]
    else:
        # graceful fallback
        for i, p in enumerate(centers12):
            m = ["M1","M2","M3","M4"][min(i//3, 3)]
            out[m].append(p)
    return out

def sim_aruco_pts_by_mirror(M1x, M2x, M3x, M4x, M1y, M2y, M3y, M4y, M1a, M2a, M3a, M4a):
    """
    Uses sim.sim_to_px(...) which returns 3 pixel points per mirror (mount corners).
    NOTE: uses fixed y values from Simulation.py (sim.M1y, sim.M2y, ...)
    """
    return {
        "M1": list(sim_to_px(M1x, M1y, M1a, mirror_id=1)),
        "M2": list(sim_to_px(M2x, M2y, M2a, mirror_id=2)),
        "M3": list(sim_to_px(M3x, M3y, M3a, mirror_id=3)),
        "M4": list(sim_to_px(M4x, M4y, M4a, mirror_id=4)),
    }

def sim_reflection_pts_by_mirror(
    M1x, M2x, M3x, M4x,
    M1y, M2y, M3y, M4y,
    M1a, M2a, M3a, M4a
):
    path = simulation_reflec(M1x, M1y, M2x, M2y, M3x, M3y, M4x, M4y,
        M1a, M2a, M3a, M4a)

    grouped = {"M1": [], "M2": [], "M3": [], "M4": []}

    mirror_names = ["M1", "M2", "M3", "M4"]

    for rec in path:

        if rec["pt"] is None:
            continue

        xw, yw = rec["pt"]
        u, v = sim_to_px_reflec(xw, yw)

        mname = mirror_names[rec["mirror"]]
        grouped[mname].append([float(u), float(v)])

    return grouped

def overlay_reflections_and_aruco(
    img_bgr,
    reflec_meas_by_mirror=None,
    aruco_meas_by_mirror=None,
    reflec_sim_by_mirror=None,
    aruco_sim_by_mirror=None,
    title="ArUcos + Reflections Overlay",
):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    mirror_colors = {"M1": "cyan", "M2": "yellow", "M3": "lime", "M4": "magenta"}

    # --- measured reflections (red open circles) ---
    if reflec_meas_by_mirror:
        all_meas = []
        for pts in reflec_meas_by_mirror.values():
            for p in pts:
                all_meas.append((float(p[0]), float(p[1])))
        if all_meas:
            rx, ry = zip(*all_meas)
            ax.scatter(rx, ry, s=120, facecolors="none", edgecolors="red",
                       linewidths=2, label="Reflections (measured)")

    # --- simulated reflections (red x) ---
    if reflec_sim_by_mirror:
        all_sim = []
        for pts in reflec_sim_by_mirror.values():
            for p in pts:
                all_sim.append((float(p[0]), float(p[1])))
        if all_sim:
            sx, sy = zip(*all_sim)
            ax.scatter(sx, sy, s=90, marker="x", linewidths=2,
                       c="red", label="Reflections (sim)")

    # --- measured arucos (colored square) ---
    if aruco_meas_by_mirror:
        for mname, pts in aruco_meas_by_mirror.items():
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(
                xs, ys,
                marker="s",
                s=260,
                linewidths=3,
                facecolors='none',   # hollow
                edgecolors=mirror_colors.get(mname, "white"),
                label=f"{mname} ArUco (measured)"
            )

    # --- simulated arucos (colored x) ---
    if aruco_sim_by_mirror:
        for mname, pts in aruco_sim_by_mirror.items():
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, marker="x", s=90, linewidths=2,
                       c=mirror_colors.get(mname, "white"),
                       label=f"{mname} ArUco (sim)")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, labelspacing=1., framealpha=0.9)
    ax.set_xlim(0, img_bgr.shape[1])
    ax.set_ylim(img_bgr.shape[0], 0)  # image coords (origin top-left)
    plt.tight_layout()
    plt.show()
    return fig, ax

def group_aruco_centers_by_mirror(centers12):
    """
    centers12: list of 12 (x,y) points sorted by ArUco ID (0..11)
    Assumes 3 per mirror in order: M1(0-2), M2(3-5), M3(6-8), M4(9-11)
    """
    centers12 = list(centers12) if centers12 is not None else []
    out = {"M1": [], "M2": [], "M3": [], "M4": []}

    if len(centers12) >= 12:
        out["M1"] = centers12[0:3]
        out["M2"] = centers12[3:6]
        out["M3"] = centers12[6:9]
        out["M4"] = centers12[9:12]
    else:
        for i, p in enumerate(centers12):
            m = ["M1", "M2", "M3", "M4"][min(i // 3, 3)]
            out[m].append(p)

    return out

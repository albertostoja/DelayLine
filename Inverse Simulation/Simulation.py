import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
import json
import itertools
import math
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution
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
qc_1 = np.array([-191, 158.24]) # Optimized for
qc_2 = np.array([-300, 185.75]) # Optimized for

# Calculating OPD
OPD_x_start = 102.1 # This is the x-coordinate of where the mirror would be in the delay line arm of the M-Z if there was no delay line
exit_angle_mean = -0.2523840245705327 # Mean exit angle from ArUco + Refl pts optimizations of 12 images
OPD_cutoff_slope = -1/exit_angle_mean # Slope for 90/10 BS
OPD_end_point = np.array([-233.95478804,  169.4891394]) # Simulated point where the OPD path would end
OPD_cutoff_second_pt = np.array([OPD_end_point[0] + 100, OPD_end_point[1] + 100*OPD_cutoff_slope]) # Another point that lies on the line of OPD_end_point w/ slope: OPD_cutoff_slope
OPD_cutoff_points = np.array([[-233.95478804,  169.4891394],[OPD_cutoff_second_pt[0], OPD_cutoff_second_pt[0]]]) # Line where the OPD calculation would end

THRESHOLD = 200     # Pixel intensity threshold for reflection point detection
EPS = 7.0           # DBSCAN groups pixels that are within EPS pixels of each other
MIN_SEP = 15        # minimum separation threshold to separate an ambiguous refl pt into two refl pts

lsr_height = 4.087 # inches

EXIT_TARGET = -0.265    # aligned exit angle
SIGMA_PX = 3            # px (tune)
SIGMA_EXIT = 8          # units of simulation_identifier (tune)
SIGMA_REFL = 3          # px (tune)
SIGMA_OPD = 0.01
SIGMA_QC = 0.01
SIGMA_MIRROR_CENTER = 0.15

DEFAULT_PEN = 50.0     # px penalty converted to residual via /SIGMA_REFL

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
        return None, None, False, None

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

    return intersection, reflected_end, inside, u

def trace_reflections(laser_start, laser_angle, mirrors, max_reflections=36):
    current_position = laser_start
    current_angle = laser_angle
    mirror_index = 0

    reflection_data = []

    for _ in range(max_reflections):
        intersection, reflected_end, inside, u = reflect_laser_ordered(
            current_position,
            current_angle,
            mirrors[mirror_index]
        )

        if intersection is None or not inside:
            break

        reflection_data.append({
            "mirror_index": mirror_index,
            "point": intersection,
            "u": u
        })

        current_position = intersection
        current_angle = np.degrees(np.arctan2(
            reflected_end[1] - intersection[1],
            reflected_end[0] - intersection[0]
        ))

        mirror_index = (mirror_index + 1) % len(mirrors)

    return reflection_data

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
    reflection_count = 0

    for _ in range(max_reflections):
        intersection, reflected_end, inside, _ = reflect_laser_ordered(
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
        reflection_count += 1

        # Update ray
        current_position = intersection
        current_angle = np.degrees(np.arctan2(
            reflected_end[1] - intersection[1],
            reflected_end[0] - intersection[0],
        ))

        mirror_index = (mirror_index + 1) % len(mirrors)

    # Delay line length (sum all segments in laser_path (except last one))
    delay_line_length = sum(
        calculate_distance(laser_path[i], laser_path[i + 1])
        for i in range(len(laser_path) - 2)
    )

    OPD_end_point_calc = find_intersection(laser_path[-2], laser_path[-1], OPD_cutoff_points[0], OPD_cutoff_points[1])

    if OPD_end_point_calc is None:
        total_path_length = 0
    else:
        last_line_OPD = calculate_distance(laser_path[-2], OPD_end_point_calc[0])
        total_path_length = delay_line_length + last_line_OPD + OPD_x_start

    return laser_path, total_path_length, reflection_count

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

def build_mirrors(M1, M2, M3, M4):
    mirrors = []
    mirror_centers = [(M1[0], M1[1]), (M2[0], M2[1]), (M3[0], M3[1]), (M4[0], M4[1])]
    mirror_angles = [M1[2], M2[2], M3[2], M4[2]]

    for center, length, angle in zip(mirror_centers, mirror_lengths, mirror_angles):
        mirrors.append(calculate_mirror_endpoints(center, length, angle))

    return mirrors

def edge_penalty(u, u_min=0.2, u_max=0.8):
    if u < u_min:
        return u_min - u
    elif u > u_max:
        return u - u_max
    else:
        return 0.0

def get_reflection_count(M1, M2, M3, M4):
    mirrors = build_mirrors(M1, M2, M3, M4)
    _, _, reflection_count = simulate_laser_with_length(laser_start, laser_angle, mirrors)
    return reflection_count

def simulation(m1cx, m1cy, m2cx, m2cy, m3cx, m3cy, m4cx, m4cy, m1a, m2a, m3a, m4a):

    mirrors = []

    # MIRROR CONFIGURATION
    mirror_centers = [(m1cx, m1cy), (m2cx, m2cy), (m3cx, m3cy), (m4cx, m4cy)]
    mirror_angles = [m1a, m2a, m3a, m4a]  # degrees

    for center, length, angle in zip(mirror_centers, mirror_lengths, mirror_angles):
        mirrors.append(calculate_mirror_endpoints(center, length, angle))

    # Initialize plot
    plt.figure(figsize=(12, 10))
    #plt.scatter(*laser_start, color='red', label="Laser Source", linewidth=1)

    # Piezo mount outline visualizer
    doubled_lines, orthogonal_lines = process_mirrors(mirrors)

    # Draw mirrors
    for mirror in mirrors:
        plt.plot([mirror[0][0], mirror[1][0]],
                 [mirror[0][1], mirror[1][1]],
                 color='black', linewidth=3)

    #Draw mirror mount outlines
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

        intersection, reflected_end, inside, _ = reflect_laser_ordered(
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
    laser_path, total_length, n_reflections = simulate_laser_with_length(
        laser_start, laser_angle, mirrors)

    # --- Exit distance calculation ---
    a = laser_path[-2]
    b = laser_path[-1]

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
    print("Total Laser Length:", total_length, "mm")
    print("Total Number of Reflection (N_R) =", reflection_count)

    # Plot settings
    plt.xlim(-310, 250)
    plt.ylim(-10, 210)
    #plt.xlim(0, 180)
    #plt.ylim(50, 140)
    #plt.axhline(0, color='black', linewidth=1)
    #plt.axvline(0, color='black', linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.title("Laser Reflection with Multiple Mirrors")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.grid(True, linewidth=0.3)
    plt.plot([qc_1[0], qc_1[0]], [qc_1[1] - 2, qc_1[1] + 2], linewidth=4, label='QC1')
    plt.plot([qc_2[0], qc_2[0]], [qc_2[1] - 2, qc_2[1] + 2], linewidth=4, label='QC2')
    plt.legend(prop={'size': 8})
    plt.show()

def simulation_fig(m1cx, m1cy, m2cx, m2cy, m3cx, m3cy, m4cx, m4cy, m1a, m2a, m3a, m4a):

    mirrors = []

    # MIRROR CONFIGURATION
    mirror_centers = [(m1cx, m1cy), (m2cx, m2cy), (m3cx, m3cy), (m4cx, m4cy)]
    mirror_angles = [m1a, m2a, m3a, m4a]  # degrees

    for center, length, angle in zip(mirror_centers, mirror_lengths, mirror_angles):
        mirrors.append(calculate_mirror_endpoints(center, length, angle))

    fig, ax = plt.subplots(figsize=(12, 5.5), frameon=False)

    # Draw only the mirror faces, without the extended mount-outline helper lines.
    for mirror in mirrors:
        ax.plot([mirror[0][0], mirror[1][0]],
                [mirror[0][1], mirror[1][1]],
                color='black', linewidth=4.5, solid_capstyle='round')

    # --- Laser simulation ---
    max_reflections = 36
    current_position = laser_start
    current_angle = laser_angle
    reflection_count = 0
    mirror_index = 0  # start with M1

    for i in range(max_reflections):

        intersection, reflected_end, inside, _ = reflect_laser_ordered(
            current_position,
            current_angle,
            mirrors[mirror_index])

        if intersection is None:
            exit_end = (
                current_position[0] + np.cos(np.radians(current_angle)) * 1000,
                current_position[1] + np.sin(np.radians(current_angle)) * 1000
            )
            ax.plot([current_position[0], exit_end[0]],
                    [current_position[1], exit_end[1]],
                    color='green', linestyle='--', linewidth=2)
            break

        if not inside:
            exit_end = (
                current_position[0] + np.cos(np.radians(current_angle)) * 1000,
                current_position[1] + np.sin(np.radians(current_angle)) * 1000
            )
            ax.plot([current_position[0], exit_end[0]],
                    [current_position[1], exit_end[1]],
                    color='green', linestyle='--', linewidth=2)
            break

        start_for_plot = current_position
        if i == 0:
            start_for_plot = (
                -65,
                current_position[1] + np.tan(np.radians(current_angle)) * (-65 - current_position[0])
            )

        ax.plot([start_for_plot[0], intersection[0]],
                [start_for_plot[1], intersection[1]],
                color='red', linewidth=2)

        current_position = intersection
        current_angle = np.degrees(np.arctan2(
            reflected_end[1] - intersection[1],
            reflected_end[0] - intersection[0]))

        reflection_count += 1
        mirror_index = (mirror_index + 1) % len(mirrors)

    # Compute full path + length using the original geometry.
    laser_path, total_length, n_reflections = simulate_laser_with_length(
        laser_start, laser_angle, mirrors)

    # --- Exit distance calculation ---
    a = laser_path[-2]
    b = laser_path[-1]

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
    print("Total Laser Length:", total_length, "mm")
    print("Total Number of Reflection (N_R) =", reflection_count)

    def y_on_exit_line(x):
        if abs(v[0]) < 1e-9:
            return float(a[1])
        return float(a[1] + (v[1] / v[0]) * (x - a[0]))

    def y_on_aligned_display_line(x):
        aligned_point = np.array([167.59574381633905, 67.7389553477378])
        aligned_slope = -0.25237623762376227
        return float(aligned_point[1] + aligned_slope * (x - aligned_point[0]))

    def draw_quadcell_snapshot(source_x, source_center_y, display_x, label, color):
        source_hit_y = y_on_exit_line(source_x)
        offset_y = source_hit_y - source_center_y
        display_center_y = y_on_aligned_display_line(display_x)
        display_hit_y = display_center_y + offset_y

        # Translate a small local slice of the real exit beam into the cropped view,
        # keeping the beam's offset from the quadcell center visible.
        direction = v / np.linalg.norm(v)
        half_segment = 9
        beam_start = np.array([display_x, display_hit_y]) - direction * half_segment
        beam_end = np.array([display_x, display_hit_y]) + direction * half_segment

        frame_width = 8
        frame_height = 18
        x0 = display_x - frame_width / 2
        y0 = display_center_y - frame_height / 2
        frame = plt.Rectangle(
            (x0, y0),
            frame_width,
            frame_height,
            fill=False,
            edgecolor=color,
            linewidth=2.4,
            linestyle='--'
        )
        ax.add_patch(frame)

        ax.plot([display_x, display_x],
                [display_center_y - 6, display_center_y + 6],
                color=color, linewidth=2.2)
        ax.plot([display_x - 3, display_x + 3],
                [display_center_y, display_center_y],
                color=color, linewidth=2.2)
        ax.scatter([display_x], [display_center_y],
                   s=26, facecolors='white', edgecolors=color, linewidths=2, zorder=4)
        ax.plot([display_x, display_x],
                [display_hit_y - 4, display_hit_y + 4],
                color='red', linewidth=1.6, alpha=0.8)
        ax.plot([beam_start[0], beam_end[0]],
                [beam_start[1], beam_end[1]],
                color='red', linewidth=2, zorder=3)
        return source_hit_y

    draw_quadcell_snapshot(-191, 158.24, -10, "QC1", "tab:blue")
    draw_quadcell_snapshot(-595, 260.20, -30, "QC2", "tab:purple")

    scale_x0 = -60
    scale_y = 56
    scale_length = 50
    ax.plot([scale_x0, scale_x0 + scale_length],
            [scale_y, scale_y],
            color='black', linewidth=2.5, solid_capstyle='butt')
    ax.plot([scale_x0, scale_x0],
            [scale_y - 1.5, scale_y + 1.5],
            color='black', linewidth=2)
    ax.plot([scale_x0 + scale_length, scale_x0 + scale_length],
            [scale_y - 1.5, scale_y + 1.5],
            color='black', linewidth=2)
    ax.text(scale_x0 + scale_length / 2, scale_y + 3.5, "5 cm",
            ha='center', va='bottom', fontsize=40, color='black')

    ax.set_xlim(-65, 185)
    ax.set_ylim(50, 150)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(1)
    plt.tight_layout(pad=0)
    plt.show()
    return fig, ax

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

        intersection, reflected_end, inside, _ = reflect_laser_ordered(
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
    metrics = _simulation_metrics(
        m1cx, m1cy, m2cx, m2cy,
        m3cx, m3cy, m4cx, m4cy,
        m1a, m2a, m3a, m4a
    )

    print(f"Exit slope: {metrics[0]}")
    print(f"Total length: {metrics[1]}")
    print(f"y191 error: {metrics[2]}")
    print(f"y300 error: {metrics[3]}")
    print(f"y595 error: {metrics[4]}")

    return metrics

def _simulation_metrics(m1cx, m1cy, m2cx, m2cy, m3cx, m3cy, m4cx, m4cy, m1a, m2a, m3a, m4a):
    mirrors = []

    mirror_centers = [(m1cx, m1cy), (m2cx, m2cy), (m3cx, m3cy), (m4cx, m4cy)]
    mirror_angles = [m1a, m2a, m3a, m4a] #in degrees

    for center, length, angle in zip(mirror_centers, mirror_lengths, mirror_angles):
        mirrors.append(calculate_mirror_endpoints(center, length, angle))

    laser_path, total_length, n_reflections = simulate_laser_with_length(laser_start, laser_angle, mirrors)

    if len(laser_path) < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    a = np.array(laser_path[-2])
    b = np.array(laser_path[-1])

    dx = b[0] - a[0]
    dy = b[1] - a[1]

    if abs(dx) < 1e-9:
        exit_slope = np.inf
    else:
        exit_slope = dy / dx

    y_int = a[1] - exit_slope * a[0]

    def y_at(x):
        return exit_slope * x + y_int

    y191 = y_at(-191)
    y300 = y_at(-300)
    y595 = y_at(-595)

    return (
        exit_slope,
        total_length,
        y191 - 158.24,
        y300 - 185.75,
        y595 - 260.20
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

def reflec_pts_cam_num_reflec(gray_img, eps=EPS, min_samples=35, show=False):

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

    # ---- RAW counts before enforcing pattern ----
    raw_counts = {k: len(v) for k, v in grouped.items()}

    return grouped, raw_counts

# Inverse Problem

def sim_to_pt(loc_x, loc_y):
    # Calibration constants from your original function
    calib_irl = [-2.65720102, -0.922]
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

def sim_to_px(x, y, a):  # For ArUcos
    sim_M_IRL = sim_to_pt(x, y)

    Xw = sim_M_IRL[0]
    Yw = sim_M_IRL[1]

    sim_M_corners = get_mount_corners(
        Xw, Yw, lsr_height, a,
        s_half=1.3 / 2,
        shift_dist=0.045
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

    M1_px = sim_to_px(M1x, M1y, M1a)
    M2_px = sim_to_px(M2x, M2y, M2a)
    M3_px = sim_to_px(M3x, M3y, M3a)
    M4_px = sim_to_px(M4x, M4y, M4a)

    M_all_px = np.array([M1_px, M2_px, M3_px, M4_px]).reshape(-1, 2)

    residuals = (M_all_px - camera_aruco_coords).reshape(-1)
    return residuals

# The residuals (differences) between the measured and simulated components (ArUcos, Reflection points, ...)
def residuals(theta, img_path_light, reflec_cam, expected_total):

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
        M1a, M2a, M3a, M4a, expected_reflections=expected_total)

    mirror_centers_world = [(M1x, M1y), (M2x, M2y), (M3x, M3y), (M4x, M4y),]
    r_refl = []

    for mirror_index, name in enumerate(["M1", "M2", "M3", "M4"]):

        meas_pts = reflec_cam[name]
        sim_for_mirror = [rec for rec in refl_sim if rec["mirror"] == mirror_index]

        half_length = mirror_lengths[mirror_index] / 2
        cx, cy = mirror_centers_world[mirror_index]
        residuals_mirror = []

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
        if meas_pts.shape[0] > 1:
            dists = np.linalg.norm(meas_pts[:, None, :] - sim_pts_px[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(dists)
        else:
            row_ind = np.array([0])
            col_ind = np.array([0])

        for r_idx, s_idx in zip(row_ind, col_ind):
            if inside_flags[s_idx]:
                du = meas_pts[r_idx, 0] - sim_pts_px[s_idx, 0]
                dv = meas_pts[r_idx, 1] - sim_pts_px[s_idx, 1]
                residuals_mirror.extend([du / SIGMA_REFL, dv / SIGMA_REFL])
            else:
                # Smooth world-space miss penalty
                xw, yw = sim_for_mirror[s_idx]["pt"]
                dx = xw - cx
                dy = yw - cy
                r = np.sqrt(dx*dx + dy*dy)
                overshoot = max(0.0, r - half_length)
                residuals_mirror.extend([overshoot / SIGMA_REFL, overshoot / SIGMA_REFL])

        r_refl.append(np.asarray(residuals_mirror, float))

    r_refl_pts = np.concatenate(r_refl) if r_refl else np.array([])

    # penalize one extra inside reflection
    r_extra_count = np.array([], dtype=float)
    refl_sim_plus = simulation_reflec(
        M1x, M1y, M2x, M2y, M3x, M3y, M4x, M4y,
        M1a, M2a, M3a, M4a,
        expected_reflections=expected_total + 1
    )
    if len(refl_sim_plus) > expected_total:
        extra_rec = refl_sim_plus[expected_total]
        if extra_rec["pt"] is not None and extra_rec["inside"]:
            r_extra_count = np.array([DEFAULT_PEN *10 / SIGMA_REFL], dtype=float)

    return np.concatenate([r_aruco, r_refl_pts, r_extra_count]) # r_exit_angle, r_exit_height

def align_sim_residuals(angles, M1, M2, M3, M4):
    g = simulation_identifier(
        M1[0], M1[1], M2[0], M2[1], M3[0], M3[1], M4[0], M4[1],
        angles[0], angles[1], angles[2], angles[3]
    )

    g = np.array(g, dtype=float)

    return np.array([
        g[2] / SIGMA_QC,   # QC1
        g[4] / SIGMA_QC    # QC2
    ], dtype=float)

def center_quadcells_residuals(angles, M1, M2, M3, M4, initial_reflections, u_min=0.2, u_max=0.8, sigma_edge=0.1):
    M1_new = np.array([M1[0], M1[1], angles[0]], dtype=float)
    M2_new = np.array([M2[0], M2[1], angles[1]], dtype=float)
    M3_new = np.array([M3[0], M3[1], angles[2]], dtype=float)
    M4_new = np.array([M4[0], M4[1], angles[3]], dtype=float)

    mirrors = build_mirrors(M1_new, M2_new, M3_new, M4_new)
    reflection_data = trace_reflections(laser_start, laser_angle, mirrors)

    n_reflections = len(reflection_data)

    if n_reflections != initial_reflections:
        return np.full(2 + max(initial_reflections - 2, 0), 1e6, dtype=float)

    g = simulation_identifier(
        M1_new[0], M1_new[1],
        M2_new[0], M2_new[1],
        M3_new[0], M3_new[1],
        M4_new[0], M4_new[1],
        M1_new[2], M2_new[2], M3_new[2], M4_new[2]
    )
    g = np.array(g, dtype=float)

    residuals = [
        g[2] / SIGMA_QC,
        g[4] / SIGMA_QC
    ]

    # Penalize only interior hits: skip first and last
    if n_reflections >= 3:
        for hit in reflection_data[1:-1]:
            p = edge_penalty(hit["u"], u_min=u_min, u_max=u_max)
            residuals.append(p / sigma_edge)

    return np.array(residuals, dtype=float)

def pack_mirrors(M1, M2, M3, M4):
    return np.array([
        M1[0], M1[1], M1[2],
        M2[0], M2[1], M2[2],
        M3[0], M3[1], M3[2],
        M4[0], M4[1], M4[2]
    ], dtype=float)

def unpack_mirrors(x):
    M1 = np.array(x[0:3], dtype=float)
    M2 = np.array(x[3:6], dtype=float)
    M3 = np.array(x[6:9], dtype=float)
    M4 = np.array(x[9:12], dtype=float)
    return M1, M2, M3, M4

def pack_variables(M1, M2, M3, M4): # Excluding y-values
    return np.array([
        M1[0], M1[2],
        M2[0], M2[2],
        M3[0], M3[2],
        M4[0], M4[2]
    ], dtype=float)

def unpack_variables(x, M1, M2, M3, M4): # Excluding y-values
    M1_new = np.array([x[0], M1[1], x[1]], dtype=float)
    M2_new = np.array([x[2], M2[1], x[3]], dtype=float)
    M3_new = np.array([x[4], M3[1], x[5]], dtype=float)
    M4_new = np.array([x[6], M4[1], x[7]], dtype=float)
    return M1_new, M2_new, M3_new, M4_new

def metrics_from_variables(x, M1, M2, M3, M4):
    M1_new, M2_new, M3_new, M4_new = unpack_variables(x, M1, M2, M3, M4)
    return np.array(_simulation_metrics(
        M1_new[0], M1_new[1],
        M2_new[0], M2_new[1],
        M3_new[0], M3_new[1],
        M4_new[0], M4_new[1],
        M1_new[2], M2_new[2], M3_new[2], M4_new[2]
    ), dtype=float)

def quadcell_errors_from_variables(x, M1, M2, M3, M4):
    g = metrics_from_variables(x, M1, M2, M3, M4)
    return g[2], g[4]

def reflection_data_from_variables(x, M1, M2, M3, M4):
    mirrors = build_mirrors(*unpack_variables(x, M1, M2, M3, M4))
    return trace_reflections(laser_start, laser_angle, mirrors)

def reflection_us_from_variables(x, M1, M2, M3, M4, include_ends=False):
    reflection_data = reflection_data_from_variables(x, M1, M2, M3, M4)

    if not include_ends and len(reflection_data) >= 3:
        reflection_data = reflection_data[1:-1]

    return np.array([hit["u"] for hit in reflection_data], dtype=float)

def reflection_edge_summary(x, M1, M2, M3, M4, include_ends=False):
    us = reflection_us_from_variables(x, M1, M2, M3, M4, include_ends=include_ends)

    if len(us) == 0:
        return {
            "min_u": np.nan,
            "max_u": np.nan,
            "closest_edge_margin": np.nan,
            "u_values": us
        }

    return {
        "min_u": float(np.min(us)),
        "max_u": float(np.max(us)),
        "closest_edge_margin": float(np.min(np.minimum(us, 1.0 - us))),
        "u_values": us
    }

def reflection_edge_penalties_from_variables(x, M1, M2, M3, M4,
                                             u_min=0.1,
                                             u_max=0.9,
                                             include_ends=False):
    us = reflection_us_from_variables(x, M1, M2, M3, M4, include_ends=include_ends)
    return np.array([edge_penalty(u, u_min=u_min, u_max=u_max) for u in us], dtype=float)

def selected_OPD_variable_indices(moving_linear_stages=("M1",)):
    if moving_linear_stages is None:
        return np.arange(8, dtype=int)

    linear_indices = {
        "M1": 0,
        "M2": 2,
        "M3": 4,
        "M4": 6
    }

    selected = []
    for mirror_name in moving_linear_stages:
        if mirror_name not in linear_indices:
            raise ValueError(f"Unknown moving linear stage: {mirror_name}")
        selected.append(linear_indices[mirror_name])

    selected.extend([1, 3, 5, 7])
    return np.array(sorted(set(selected)), dtype=int)

def expand_selected_variables(x_selected, x_base, variable_indices):
    x_full = np.array(x_base, dtype=float).copy()
    x_full[np.array(variable_indices, dtype=int)] = np.array(x_selected, dtype=float)
    return x_full

def quadcell_constraints_ok(qc1_error, qc2_error,
                            max_qc_error=2.0,
                            max_qc_difference=2.0,
                            tolerance=0.0):
    return (
        abs(qc1_error) <= max_qc_error + tolerance and
        abs(qc2_error) <= max_qc_error + tolerance and
        abs(qc1_error - qc2_error) <= max_qc_difference + tolerance
    )

def actuation_constraint_diagnostics(x, M1, M2, M3, M4,
                                     max_qc_error=2.0,
                                     max_qc_difference=2.0,
                                     expected_reflections=None,
                                     u_min=0.1,
                                     u_max=0.9,
                                     enforce_edge_bounds=True,
                                     include_edge_ends=False,
                                     constraint_tolerance=0.0):
    qc1_error, qc2_error = quadcell_errors_from_variables(x, M1, M2, M3, M4)
    mirrors = unpack_variables(x, M1, M2, M3, M4)
    reflection_count = get_reflection_count(*mirrors)
    edge_summary = reflection_edge_summary(x, M1, M2, M3, M4, include_ends=include_edge_ends)
    edge_penalties = reflection_edge_penalties_from_variables(
        x, M1, M2, M3, M4,
        u_min=u_min - constraint_tolerance,
        u_max=u_max + constraint_tolerance,
        include_ends=include_edge_ends
    )

    failures = []
    if not np.isfinite(qc1_error) or not np.isfinite(qc2_error):
        failures.append("quadcell metric is not finite")
    if abs(qc1_error) > max_qc_error + constraint_tolerance:
        failures.append(f"QC1 offset {qc1_error:.4g} exceeds {max_qc_error}")
    if abs(qc2_error) > max_qc_error + constraint_tolerance:
        failures.append(f"QC2 offset {qc2_error:.4g} exceeds {max_qc_error}")
    if abs(qc1_error - qc2_error) > max_qc_difference + constraint_tolerance:
        failures.append(f"QC difference {qc1_error - qc2_error:.4g} exceeds {max_qc_difference}")
    if expected_reflections is not None and reflection_count != expected_reflections:
        failures.append(f"reflection count {reflection_count} != expected {expected_reflections}")
    if enforce_edge_bounds and np.any(edge_penalties > 0):
        failures.append(
            f"reflection u range [{edge_summary['min_u']:.4g}, {edge_summary['max_u']:.4g}] "
            f"outside [{u_min}, {u_max}]"
        )

    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "qc1_error": qc1_error,
        "qc2_error": qc2_error,
        "qc_difference": qc1_error - qc2_error,
        "reflection_count": reflection_count,
        "min_u": edge_summary["min_u"],
        "max_u": edge_summary["max_u"],
        "closest_edge_margin": edge_summary["closest_edge_margin"]
    }

ACTUATOR_AXES = [
    ("M1", "dx", 0),
    ("M1", "dangle", 1),
    ("M2", "dx", 2),
    ("M2", "dangle", 3),
    ("M3", "dx", 4),
    ("M3", "dangle", 5),
    ("M4", "dx", 6),
    ("M4", "dangle", 7)
]

def actuator_label(axis_index):
    for mirror_name, command_name, idx in ACTUATOR_AXES:
        if idx == axis_index:
            return f"{mirror_name}.{command_name}"
    return None

def variables_with_axis_move(x, axis_index, amount):
    x_next = np.array(x, dtype=float).copy()
    x_next[axis_index] += amount
    return x_next

def state_satisfies_actuation_constraints(x, M1, M2, M3, M4,
                                          max_qc_error=2.0,
                                          max_qc_difference=2.0,
                                          expected_reflections=None,
                                          u_min=0.1,
                                          u_max=0.9,
                                          enforce_edge_bounds=True,
                                          include_edge_ends=False,
                                          constraint_tolerance=0.0):
    diagnostics = actuation_constraint_diagnostics(
        x, M1, M2, M3, M4,
        max_qc_error=max_qc_error,
        max_qc_difference=max_qc_difference,
        expected_reflections=expected_reflections,
        u_min=u_min,
        u_max=u_max,
        enforce_edge_bounds=enforce_edge_bounds,
        include_edge_ends=include_edge_ends,
        constraint_tolerance=constraint_tolerance
    )
    return diagnostics["ok"]

def one_actuator_motion_is_valid(x_previous, x_current, M1, M2, M3, M4,
                                 max_qc_error=2.0,
                                 max_qc_difference=2.0,
                                 expected_reflections=None,
                                 motion_samples_per_step=25,
                                 u_min=0.1,
                                 u_max=0.9,
                                 enforce_edge_bounds=True,
                                 include_edge_ends=False,
                                 constraint_tolerance=0.0):
    delta = np.array(x_current, dtype=float) - np.array(x_previous, dtype=float)
    if np.count_nonzero(np.abs(delta) > 1e-12) > 1:
        return False

    for fraction in np.linspace(0.0, 1.0, motion_samples_per_step + 1)[1:]:
        x_sample = x_previous + fraction * delta
        if not state_satisfies_actuation_constraints(
            x_sample, M1, M2, M3, M4,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            expected_reflections=expected_reflections,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=constraint_tolerance
        ):
            return False

    return True

def actuation_path_residuals(x, x_nominal, M1, M2, M3, M4,
                             variable_scale,
                             max_qc_error=2.0,
                             max_qc_difference=2.0,
                             qc_slack=0.05,
                             expected_reflections=None,
                             u_min=0.1,
                             u_max=0.9,
                             sigma_edge=0.02,
                             enforce_edge_bounds=True,
                             include_edge_ends=False):
    qc1_error, qc2_error = quadcell_errors_from_variables(x, M1, M2, M3, M4)

    residuals = list((x - x_nominal) / variable_scale)

    residuals.extend([
        max(0.0, abs(qc1_error) - max_qc_error) / qc_slack,
        max(0.0, abs(qc2_error) - max_qc_error) / qc_slack,
        max(0.0, abs(qc1_error - qc2_error) - max_qc_difference) / qc_slack
    ])

    if expected_reflections is not None:
        M1_new, M2_new, M3_new, M4_new = unpack_variables(x, M1, M2, M3, M4)
        n_reflections = get_reflection_count(M1_new, M2_new, M3_new, M4_new)
        if n_reflections != expected_reflections:
            residuals.append(1e4 * (n_reflections - expected_reflections))

    if enforce_edge_bounds:
        edge_penalties = reflection_edge_penalties_from_variables(
            x, M1, M2, M3, M4,
            u_min=u_min,
            u_max=u_max,
            include_ends=include_edge_ends
        )
        residuals.extend(edge_penalties / sigma_edge)

    return np.array(residuals, dtype=float)

def make_actuation_step(step_index, fraction, x_previous, x_current, M1, M2, M3, M4,
                        max_qc_error=2.0,
                        max_qc_difference=2.0,
                        motion_samples_per_step=None,
                        u_min=0.1,
                        u_max=0.9,
                        include_edge_ends=False,
                        enforce_edge_bounds=True,
                        constraint_tolerance=0.0):
    M1_new, M2_new, M3_new, M4_new = unpack_variables(x_current, M1, M2, M3, M4)
    g = metrics_from_variables(x_current, M1, M2, M3, M4)
    reflection_count = get_reflection_count(M1_new, M2_new, M3_new, M4_new)
    delta = np.array(x_current, dtype=float) - np.array(x_previous, dtype=float)
    active_axes = np.flatnonzero(np.abs(delta) > 1e-12)
    active_axis = int(active_axes[0]) if len(active_axes) == 1 else None

    commands = {
        "M1": {"dx": x_current[0] - x_previous[0], "dangle": x_current[1] - x_previous[1]},
        "M2": {"dx": x_current[2] - x_previous[2], "dangle": x_current[3] - x_previous[3]},
        "M3": {"dx": x_current[4] - x_previous[4], "dangle": x_current[5] - x_previous[5]},
        "M4": {"dx": x_current[6] - x_previous[6], "dangle": x_current[7] - x_previous[7]}
    }

    cumulative = {
        "M1": {"x": x_current[0], "angle": x_current[1]},
        "M2": {"x": x_current[2], "angle": x_current[3]},
        "M3": {"x": x_current[4], "angle": x_current[5]},
        "M4": {"x": x_current[6], "angle": x_current[7]}
    }

    qc1_error = g[2]
    qc2_error = g[4]
    edge_summary = reflection_edge_summary(
        x_current, M1, M2, M3, M4,
        include_ends=include_edge_ends
    )
    edge_penalties = reflection_edge_penalties_from_variables(
        x_current, M1, M2, M3, M4,
        u_min=u_min,
        u_max=u_max,
        include_ends=include_edge_ends
    )

    return {
        "step": step_index,
        "fraction": fraction,
        "mirrors": (M1_new, M2_new, M3_new, M4_new),
        "commands": commands,
        "positions": cumulative,
        "OPD": g[1],
        "qc1_error": qc1_error,
        "qc2_error": qc2_error,
        "qc_difference": qc1_error - qc2_error,
        "reflection_count": reflection_count,
        "actuator": actuator_label(active_axis) if active_axis is not None else None,
        "axis_index": active_axis,
        "command_value": delta[active_axis] if active_axis is not None else None,
        "single_actuator_step": len(active_axes) <= 1,
        "motion_samples_checked": motion_samples_per_step,
        "min_reflection_u": edge_summary["min_u"],
        "max_reflection_u": edge_summary["max_u"],
        "closest_edge_margin": edge_summary["closest_edge_margin"],
        "reflection_u_values": edge_summary["u_values"],
        "within_edge_bounds": (not enforce_edge_bounds) or not np.any(edge_penalties > 0),
        "within_constraints": quadcell_constraints_ok(
            qc1_error, qc2_error,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            tolerance=constraint_tolerance
        ) and ((not enforce_edge_bounds) or not np.any(edge_penalties > 0))
    }

def build_actuation_plan_summary(steps, x_start, x_target, M1, M2, M3, M4,
                                 start_reflections, target_reflections,
                                 start_within_constraints,
                                 expected_reflections,
                                 max_qc_error=2.0,
                                 max_qc_difference=2.0,
                                 motion_samples_per_step=25,
                                 u_min=0.1,
                                 u_max=0.9,
                                 include_edge_ends=False,
                                 search_mode=None,
                                 split_count=None,
                                 failure_reason=None):
    start_qc1_error, start_qc2_error = quadcell_errors_from_variables(x_start, M1, M2, M3, M4)

    if len(steps) > 0:
        max_abs_qc1_error = max(abs(step["qc1_error"]) for step in steps)
        max_abs_qc2_error = max(abs(step["qc2_error"]) for step in steps)
        max_abs_qc_difference = max(abs(step["qc_difference"]) for step in steps)
        min_reflection_u = min(step["min_reflection_u"] for step in steps)
        max_reflection_u = max(step["max_reflection_u"] for step in steps)
        min_closest_edge_margin = min(step["closest_edge_margin"] for step in steps)
    else:
        max_abs_qc1_error = abs(start_qc1_error)
        max_abs_qc2_error = abs(start_qc2_error)
        max_abs_qc_difference = abs(start_qc1_error - start_qc2_error)
        start_edge_summary = reflection_edge_summary(
            x_start, M1, M2, M3, M4,
            include_ends=include_edge_ends
        )
        min_reflection_u = start_edge_summary["min_u"]
        max_reflection_u = start_edge_summary["max_u"]
        min_closest_edge_margin = start_edge_summary["closest_edge_margin"]

    final_error = np.array(x_target, dtype=float) - np.array(x_start, dtype=float)
    if len(steps) > 0:
        last_positions = pack_variables(*steps[-1]["mirrors"])
        final_error = np.array(x_target, dtype=float) - last_positions

    return {
        "steps": steps,
        "n_steps": len(steps),
        "start_within_constraints": start_within_constraints,
        "all_within_constraints": failure_reason is None and start_within_constraints and all(step["within_constraints"] for step in steps),
        "waypoints_within_constraints": failure_reason is None and all(step["within_constraints"] for step in steps),
        "single_actuator_steps": all(step["single_actuator_step"] for step in steps),
        "start_qc1_error": start_qc1_error,
        "start_qc2_error": start_qc2_error,
        "start_qc_difference": start_qc1_error - start_qc2_error,
        "max_abs_qc1_error": max_abs_qc1_error,
        "max_abs_qc2_error": max_abs_qc2_error,
        "max_abs_qc_difference": max_abs_qc_difference,
        "start_reflections": start_reflections,
        "target_reflections": target_reflections,
        "preserved_reflection_count": expected_reflections is not None,
        "max_qc_error": max_qc_error,
        "max_qc_difference": max_qc_difference,
        "u_min": u_min,
        "u_max": u_max,
        "include_edge_ends": include_edge_ends,
        "min_reflection_u": min_reflection_u,
        "max_reflection_u": max_reflection_u,
        "min_closest_edge_margin": min_closest_edge_margin,
        "motion_samples_per_step": motion_samples_per_step,
        "search_mode": search_mode,
        "split_count": split_count,
        "final_variable_error": final_error,
        "failure_reason": failure_reason
    }

def plot_actuation_quadcell_offsets(actuation_plan, show_difference=True):
    steps = actuation_plan.get("steps", [])
    step_numbers = [0] + [step["step"] for step in steps]
    qc1_errors = [actuation_plan["start_qc1_error"]] + [step["qc1_error"] for step in steps]
    qc2_errors = [actuation_plan["start_qc2_error"]] + [step["qc2_error"] for step in steps]
    qc_differences = [actuation_plan["start_qc_difference"]] + [step["qc_difference"] for step in steps]

    max_qc_error = actuation_plan.get("max_qc_error", 2.0)
    max_qc_difference = actuation_plan.get("max_qc_difference", 2.0)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(step_numbers, qc1_errors, marker="o", linewidth=1.5, label="QC1 offset")
    ax.plot(step_numbers, qc2_errors, marker="o", linewidth=1.5, label="QC2 offset")

    if show_difference:
        ax.plot(
            step_numbers,
            qc_differences,
            marker=".",
            linewidth=1.0,
            linestyle="--",
            label="QC1 - QC2"
        )

    ax.axhline(max_qc_error, color="black", linestyle=":", linewidth=1, label="+/- QC limit")
    ax.axhline(-max_qc_error, color="black", linestyle=":", linewidth=1)

    if show_difference and max_qc_difference != max_qc_error:
        ax.axhline(max_qc_difference, color="gray", linestyle="--", linewidth=1, label="+/- difference limit")
        ax.axhline(-max_qc_difference, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Actuator step")
    ax.set_ylabel("Beam offset (mm)")
    ax.set_title("Quadcell Beam Offset During Actuation")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax

def plot_actuation_reflection_u(actuation_plan):
    steps = actuation_plan.get("steps", [])
    step_numbers = [0] + [step["step"] for step in steps]
    min_us = [np.nan] + [step["min_reflection_u"] for step in steps]
    max_us = [np.nan] + [step["max_reflection_u"] for step in steps]
    margins = [np.nan] + [step["closest_edge_margin"] for step in steps]

    u_min = actuation_plan.get("u_min", 0.1)
    u_max = actuation_plan.get("u_max", 0.9)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(step_numbers, min_us, marker="o", linewidth=1.5, label="minimum reflection u")
    ax.plot(step_numbers, max_us, marker="o", linewidth=1.5, label="maximum reflection u")
    ax.plot(step_numbers, margins, marker=".", linewidth=1.0, linestyle="--", label="closest edge margin")
    ax.axhline(u_min, color="black", linestyle=":", linewidth=1, label="u bounds")
    ax.axhline(u_max, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Actuator step")
    ax.set_ylabel("Reflection position u")
    ax.set_title("Reflection Positions During Actuation")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax

def try_one_actuator_sequence(x_start, x_target, axis_sequence, M1, M2, M3, M4,
                              max_qc_error=2.0,
                              max_qc_difference=2.0,
                              expected_reflections=None,
                              motion_samples_per_step=25,
                              u_min=0.1,
                              u_max=0.9,
                              enforce_edge_bounds=True,
                              include_edge_ends=False,
                              constraint_tolerance=0.0):
    x_current = np.array(x_start, dtype=float).copy()
    x_target = np.array(x_target, dtype=float)
    steps = []

    for axis_index in axis_sequence:
        amount = x_target[axis_index] - x_current[axis_index]
        if abs(amount) <= 1e-12:
            continue

        x_next = variables_with_axis_move(x_current, axis_index, amount)
        if not one_actuator_motion_is_valid(
            x_current, x_next, M1, M2, M3, M4,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            expected_reflections=expected_reflections,
            motion_samples_per_step=motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=constraint_tolerance
        ):
            return None

        steps.append(make_actuation_step(
            len(steps) + 1,
            np.linalg.norm(x_next - x_start) / max(np.linalg.norm(x_target - x_start), 1e-12),
            x_current, x_next, M1, M2, M3, M4,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            motion_samples_per_step=motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            include_edge_ends=include_edge_ends,
            enforce_edge_bounds=enforce_edge_bounds,
            constraint_tolerance=constraint_tolerance
        ))
        x_current = x_next

    if not np.allclose(x_current, x_target, atol=1e-9, rtol=0):
        return None

    return steps

def candidate_axis_orders(active_axes, delta, include_all_permutations=True):
    active_axes = list(active_axes)
    if len(active_axes) <= 1:
        return [tuple(active_axes)]

    ranked = tuple(sorted(active_axes, key=lambda idx: abs(delta[idx])))
    reverse_ranked = tuple(reversed(ranked))

    orders = [ranked, reverse_ranked]
    if include_all_permutations and len(active_axes) <= 8:
        orders.extend(itertools.permutations(active_axes))

    seen = set()
    unique_orders = []
    for order in orders:
        if order in seen:
            continue
        seen.add(order)
        unique_orders.append(order)

    return unique_orders

def plan_actuation_path(x_start, x_target, M1, M2, M3, M4,
                        max_axis_splits=64,
                        max_qc_error=2.0,
                        max_qc_difference=2.0,
                        preserve_reflection_count=True,
                        motion_samples_per_step=25,
                        u_min=0.1,
                        u_max=0.9,
                        enforce_edge_bounds=True,
                        include_edge_ends=False,
                        constraint_tolerance=0.05,
                        zero_tol=1e-9,
                        verbose=False):
    if max_axis_splits < 1:
        raise ValueError("max_axis_splits must be at least 1.")

    x_start = np.array(x_start, dtype=float)
    x_target = np.array(x_target, dtype=float)

    M_start = unpack_variables(x_start, M1, M2, M3, M4)
    M_target = unpack_variables(x_target, M1, M2, M3, M4)
    start_reflections = get_reflection_count(*M_start)
    target_reflections = get_reflection_count(*M_target)
    start_qc1_error, start_qc2_error = quadcell_errors_from_variables(x_start, M1, M2, M3, M4)
    start_diagnostics = actuation_constraint_diagnostics(
        x_start, M1, M2, M3, M4,
        max_qc_error=max_qc_error,
        max_qc_difference=max_qc_difference,
        expected_reflections=None,
        u_min=u_min,
        u_max=u_max,
        enforce_edge_bounds=enforce_edge_bounds,
        include_edge_ends=include_edge_ends,
        constraint_tolerance=constraint_tolerance
    )
    start_within_constraints = start_diagnostics["ok"]

    expected_reflections = None
    if preserve_reflection_count and start_reflections == target_reflections:
        expected_reflections = start_reflections

    delta = x_target - x_start
    active_axes = np.flatnonzero(np.abs(delta) > zero_tol)

    if not start_within_constraints:
        return build_actuation_plan_summary(
            [], x_start, x_target, M1, M2, M3, M4,
            start_reflections, target_reflections, start_within_constraints,
            expected_reflections,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            motion_samples_per_step=motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            include_edge_ends=include_edge_ends,
            search_mode="failed",
            split_count=0,
            failure_reason="Starting state is outside constraints: " + "; ".join(start_diagnostics["failures"])
        )

    target_diagnostics = actuation_constraint_diagnostics(
        x_target, M1, M2, M3, M4,
        max_qc_error=max_qc_error,
        max_qc_difference=max_qc_difference,
        expected_reflections=expected_reflections,
        u_min=u_min,
        u_max=u_max,
        enforce_edge_bounds=enforce_edge_bounds,
        include_edge_ends=include_edge_ends,
        constraint_tolerance=constraint_tolerance
    )
    if not target_diagnostics["ok"]:
        return build_actuation_plan_summary(
            [], x_start, x_target, M1, M2, M3, M4,
            start_reflections, target_reflections, start_within_constraints,
            expected_reflections,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            motion_samples_per_step=motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            include_edge_ends=include_edge_ends,
            search_mode="failed",
            split_count=0,
            failure_reason="Target state is outside constraints: " + "; ".join(target_diagnostics["failures"])
        )

    if len(active_axes) == 0:
        return build_actuation_plan_summary(
            [], x_start, x_target, M1, M2, M3, M4,
            start_reflections, target_reflections, start_within_constraints,
            expected_reflections,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            motion_samples_per_step=motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            include_edge_ends=include_edge_ends,
            search_mode="already_at_target",
            split_count=0
        )

    full_axis_orders = candidate_axis_orders(active_axes, delta, include_all_permutations=True)
    fast_axis_orders = candidate_axis_orders(active_axes, delta, include_all_permutations=False)

    for axis_order in full_axis_orders:
        steps = try_one_actuator_sequence(
            x_start, x_target, axis_order, M1, M2, M3, M4,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            expected_reflections=expected_reflections,
            motion_samples_per_step=motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=constraint_tolerance
        )
        if steps is not None:
            return build_actuation_plan_summary(
                steps, x_start, x_target, M1, M2, M3, M4,
                start_reflections, target_reflections, start_within_constraints,
                expected_reflections,
                max_qc_error=max_qc_error,
                max_qc_difference=max_qc_difference,
                motion_samples_per_step=motion_samples_per_step,
                u_min=u_min,
                u_max=u_max,
                include_edge_ends=include_edge_ends,
                search_mode="one_full_move_per_actuator",
                split_count=1
            )

    for split_count in range(2, max_axis_splits + 1):
        x_current = x_start.copy()
        steps = []
        failed = False

        for split_index in range(split_count):
            remaining_splits = split_count - split_index
            remaining_delta = x_target - x_current
            split_delta = remaining_delta / remaining_splits

            best_order = None
            best_steps = None
            best_max_qc = np.inf

            for axis_order in fast_axis_orders:
                x_trial = x_current.copy()
                trial_steps = []
                order_failed = False

                for axis_index in axis_order:
                    amount = split_delta[axis_index]
                    if abs(amount) <= zero_tol:
                        continue

                    x_next = variables_with_axis_move(x_trial, axis_index, amount)
                    if not one_actuator_motion_is_valid(
                        x_trial, x_next, M1, M2, M3, M4,
                        max_qc_error=max_qc_error,
                        max_qc_difference=max_qc_difference,
                        expected_reflections=expected_reflections,
                        motion_samples_per_step=motion_samples_per_step,
                        u_min=u_min,
                        u_max=u_max,
                        enforce_edge_bounds=enforce_edge_bounds,
                        include_edge_ends=include_edge_ends,
                        constraint_tolerance=constraint_tolerance
                    ):
                        order_failed = True
                        break

                    trial_steps.append((x_trial, x_next))
                    x_trial = x_next

                if order_failed:
                    continue

                qc1_error, qc2_error = quadcell_errors_from_variables(x_trial, M1, M2, M3, M4)
                order_max_qc = max(abs(qc1_error), abs(qc2_error), abs(qc1_error - qc2_error))
                if order_max_qc < best_max_qc:
                    best_order = axis_order
                    best_steps = trial_steps
                    best_max_qc = order_max_qc

            if best_order is None:
                failed = True
                break

            for x_previous, x_next in best_steps:
                steps.append(make_actuation_step(
                    len(steps) + 1,
                    np.linalg.norm(x_next - x_start) / max(np.linalg.norm(x_target - x_start), 1e-12),
                    x_previous, x_next, M1, M2, M3, M4,
                    max_qc_error=max_qc_error,
                    max_qc_difference=max_qc_difference,
                    motion_samples_per_step=motion_samples_per_step,
                    u_min=u_min,
                    u_max=u_max,
                    include_edge_ends=include_edge_ends,
                    enforce_edge_bounds=enforce_edge_bounds,
                    constraint_tolerance=constraint_tolerance
                ))
                x_current = x_next.copy()

        if not failed and np.allclose(x_current, x_target, atol=1e-8, rtol=0):
            return build_actuation_plan_summary(
                steps, x_start, x_target, M1, M2, M3, M4,
                start_reflections, target_reflections, start_within_constraints,
                expected_reflections,
                max_qc_error=max_qc_error,
                max_qc_difference=max_qc_difference,
                motion_samples_per_step=motion_samples_per_step,
                u_min=u_min,
                u_max=u_max,
                include_edge_ends=include_edge_ends,
                search_mode="split_single_actuator_moves",
                split_count=split_count
            )

        if verbose:
            print(f"No valid single-actuator path found with split_count={split_count}.")

    return build_actuation_plan_summary(
        [], x_start, x_target, M1, M2, M3, M4,
        start_reflections, target_reflections, start_within_constraints,
        expected_reflections,
        max_qc_error=max_qc_error,
        max_qc_difference=max_qc_difference,
        motion_samples_per_step=motion_samples_per_step,
        u_min=u_min,
        u_max=u_max,
        include_edge_ends=include_edge_ends,
        search_mode="failed",
        split_count=max_axis_splits,
        failure_reason="No valid single-actuator path found up to max_axis_splits."
    )

def combine_actuation_plans(segment_plans):
    if len(segment_plans) == 0:
        return None

    combined_steps = []
    for segment_index, plan in enumerate(segment_plans, start=1):
        for step in plan["steps"]:
            step_new = dict(step)
            step_new["segment"] = segment_index
            step_new["segment_step"] = step["step"]
            step_new["step"] = len(combined_steps) + 1
            combined_steps.append(step_new)

    first = segment_plans[0]

    return {
        "steps": combined_steps,
        "segments": segment_plans,
        "n_segments": len(segment_plans),
        "n_steps": len(combined_steps),
        "start_within_constraints": first["start_within_constraints"],
        "all_within_constraints": all(plan["all_within_constraints"] for plan in segment_plans),
        "waypoints_within_constraints": all(plan["waypoints_within_constraints"] for plan in segment_plans),
        "single_actuator_steps": all(plan["single_actuator_steps"] for plan in segment_plans),
        "start_qc1_error": first["start_qc1_error"],
        "start_qc2_error": first["start_qc2_error"],
        "start_qc_difference": first["start_qc_difference"],
        "max_abs_qc1_error": max(plan["max_abs_qc1_error"] for plan in segment_plans),
        "max_abs_qc2_error": max(plan["max_abs_qc2_error"] for plan in segment_plans),
        "max_abs_qc_difference": max(plan["max_abs_qc_difference"] for plan in segment_plans),
        "start_reflections": first["start_reflections"],
        "target_reflections": segment_plans[-1]["target_reflections"],
        "preserved_reflection_count": all(plan["preserved_reflection_count"] for plan in segment_plans),
        "max_qc_error": first["max_qc_error"],
        "max_qc_difference": first["max_qc_difference"],
        "u_min": first["u_min"],
        "u_max": first["u_max"],
        "include_edge_ends": first["include_edge_ends"],
        "min_reflection_u": min(plan["min_reflection_u"] for plan in segment_plans),
        "max_reflection_u": max(plan["max_reflection_u"] for plan in segment_plans),
        "min_closest_edge_margin": min(plan["min_closest_edge_margin"] for plan in segment_plans),
        "motion_samples_per_step": first["motion_samples_per_step"],
        "search_mode": "staged_OPD",
        "split_count": max(plan["split_count"] for plan in segment_plans),
        "final_variable_error": segment_plans[-1]["final_variable_error"],
        "failure_reason": next((plan["failure_reason"] for plan in segment_plans if plan["failure_reason"] is not None), None)
    }

def OPD_residuals(x, target_OPD, M1, M2, M3, M4,
                  u_min=0.1,
                  u_max=0.9,
                  sigma_edge=0.02,
                  enforce_edge_bounds=True,
                  include_edge_ends=False):
    g = metrics_from_variables(x, M1, M2, M3, M4)

    r_OPD = (g[1] - target_OPD) / SIGMA_OPD
    r_qc1 = g[2] / SIGMA_QC
    r_qc2 = g[4] / SIGMA_QC

    residuals = [r_OPD, r_qc1, r_qc2]

    if enforce_edge_bounds:
        edge_penalties = reflection_edge_penalties_from_variables(
            x, M1, M2, M3, M4,
            u_min=u_min,
            u_max=u_max,
            include_ends=include_edge_ends
        )
        residuals.extend(edge_penalties / sigma_edge)

    return np.array(residuals, dtype=float)

def OPD_residuals_selected(x_selected, x_base, variable_indices, target_OPD, M1, M2, M3, M4,
                           u_min=0.1,
                           u_max=0.9,
                           sigma_edge=0.02,
                           enforce_edge_bounds=True,
                           include_edge_ends=False):
    x_full = expand_selected_variables(x_selected, x_base, variable_indices)
    return OPD_residuals(
        x_full, target_OPD, M1, M2, M3, M4,
        u_min=u_min,
        u_max=u_max,
        sigma_edge=sigma_edge,
        enforce_edge_bounds=enforce_edge_bounds,
        include_edge_ends=include_edge_ends
    )

def solve_OPD_configuration(target_OPD, M1, M2, M3, M4,
                            moving_linear_stages=("M1",),
                            u_min=0.1,
                            u_max=0.9,
                            sigma_edge=0.02,
                            enforce_edge_bounds=True,
                            include_edge_ends=False,
                            verbose=0):
    x0 = pack_variables(M1, M2, M3, M4)
    variable_indices = selected_OPD_variable_indices(moving_linear_stages)
    x0_selected = x0[variable_indices]

    res = least_squares(
        fun=lambda x: OPD_residuals_selected(
            x, x0, variable_indices,
            target_OPD, M1, M2, M3, M4,
            u_min=u_min,
            u_max=u_max,
            sigma_edge=sigma_edge,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends
        ),
        x0=x0_selected,
        loss="linear",
        f_scale=1.0,
        verbose=verbose,
        x_scale='jac',
        max_nfev=4000,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10
    )

    x_opt = expand_selected_variables(res.x, x0, variable_indices)
    return x_opt, res

# OPTIMIZING

def optimize_inverse(M1, M2, M3, M4, img_path_light, img_path_dark=None):

    theta0 = np.array(
        [M1[0], M2[0], M3[0], M4[0],
         M1[1], M2[1], M3[1], M4[1],
         M1[2], M2[2], M3[2], M4[2]],
        dtype=float
    )

    if img_path_dark is None:
        residual_fun = lambda th: aruco_pixel_residuals(th, img_path_light) / SIGMA_PX
    else:
        img_dark = cv.imread(img_path_dark)
        if img_dark is None:
            raise ValueError(f"Could not read dark image: {img_path_dark}")

        img_gray = cv.cvtColor(img_dark, cv.COLOR_BGR2GRAY)

        reflec_cam = reflec_pts_cam(img_gray, show=False)
        expected_total = sum(len(v) for v in reflec_cam.values())

        residual_fun = lambda th: residuals(
            th,
            img_path_light=img_path_light,
            reflec_cam=reflec_cam,
            expected_total=expected_total
        )

    res = least_squares(
        fun=residual_fun,
        x0=theta0,
        loss="linear",
        f_scale=1.0,
        verbose=2,  # IMPORTANT for profiling
        x_scale = np.array([20,20,20,20,  20,20,20,20,  0.5,0.5,0.5,0.5], dtype=float),
        max_nfev=4000,
        ftol=1e-10, 
        xtol=1e-10, 
        gtol=1e-10
    )

    return res

def solve_center_once(theta0, M1, M2, M3, M4, initial_reflections,
                      u_min=0.2, u_max=0.8, sigma_edge=0.1):
    res = least_squares(
        fun=lambda th: center_quadcells_residuals(
            th, M1, M2, M3, M4,
            initial_reflections=initial_reflections,
            u_min=u_min, u_max=u_max,
            sigma_edge=sigma_edge
        ),
        x0=np.array(theta0, dtype=float),
        loss="linear",
        f_scale=1.0,
        verbose=0,
        x_scale='jac',
        max_nfev=4000,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10
    )
    return res

def center_quadcells(M1, M2, M3, M4,
                     n_tries=20,
                     angle_perturb=0.2,
                     seed=0,
                     u_min=0.2,
                     u_max=0.8,
                     sigma_edge=0.1):

    theta_init = np.array([M1[2], M2[2], M3[2], M4[2]], dtype=float)

    mirrors0 = build_mirrors(M1, M2, M3, M4)
    reflection_data0 = trace_reflections(laser_start, laser_angle, mirrors0)
    initial_reflections = len(reflection_data0)

    rng = np.random.default_rng(seed)

    starts = [theta_init]
    for _ in range(n_tries - 1):
        starts.append(theta_init + rng.uniform(-angle_perturb, angle_perturb, size=4))

    best_res = None
    best_norm = np.inf
    best_angles = None

    for th0 in starts:
        res = solve_center_once(
            th0, M1, M2, M3, M4,
            initial_reflections=initial_reflections,
            u_min=u_min, u_max=u_max,
            sigma_edge=sigma_edge
        )

        M1_new = np.array([M1[0], M1[1], res.x[0]], dtype=float)
        M2_new = np.array([M2[0], M2[1], res.x[1]], dtype=float)
        M3_new = np.array([M3[0], M3[1], res.x[2]], dtype=float)
        M4_new = np.array([M4[0], M4[1], res.x[3]], dtype=float)

        mirrors_new = build_mirrors(M1_new, M2_new, M3_new, M4_new)
        reflection_data_new = trace_reflections(laser_start, laser_angle, mirrors_new)

        if len(reflection_data_new) != initial_reflections:
            continue

        g_final = simulation_identifier(
            M1_new[0], M1_new[1],
            M2_new[0], M2_new[1],
            M3_new[0], M3_new[1],
            M4_new[0], M4_new[1],
            M1_new[2], M2_new[2], M3_new[2], M4_new[2]
        )
        g_final = np.array(g_final, dtype=float)

        qc_norm = np.linalg.norm([g_final[2], g_final[4]])

        if qc_norm < best_norm:
            best_norm = qc_norm
            best_res = res
            best_angles = res.x.copy()

    if best_res is None:
        raise RuntimeError("No valid centered solution found with the same reflection count.")

    M1_opt = np.array([M1[0], M1[1], best_angles[0]], dtype=float)
    M2_opt = np.array([M2[0], M2[1], best_angles[1]], dtype=float)
    M3_opt = np.array([M3[0], M3[1], best_angles[2]], dtype=float)
    M4_opt = np.array([M4[0], M4[1], best_angles[3]], dtype=float)

    return (M1_opt, M2_opt, M3_opt, M4_opt), best_res

#simulation(M1_opt[0], M1_opt[1],
#                      M2_opt[0], M2_opt[1], 
#                      M3_opt[0], M3_opt[1], 
#                      M4_opt[0], M4_opt[1],
#                      M1_opt[2], M2_opt[2], M3_opt[2], M4_opt[2]), best_res

def choose_OPD(target_OPD, M1, M2, M3, M4,
               return_actuation_plan=True,
               n_actuation_steps=None,
               max_axis_splits=20,
               max_qc_error=2.0,
               max_qc_difference=2.0,
               preserve_reflection_count=True,
               motion_samples_per_step=25,
               moving_linear_stages=("M1",),
               max_OPD_step=20.0,
               u_min=0.1,
               u_max=0.9,
               sigma_edge=0.02,
               enforce_edge_bounds=True,
               include_edge_ends=False,
               constraint_tolerance=0.05,
               auto_recenter_start=True,
               recenter_constraint_tolerance=0.25,
               optimizer_verbose=0):
    x_start = pack_variables(M1, M2, M3, M4)
    start_OPD = metrics_from_variables(x_start, M1, M2, M3, M4)[1]

    if max_OPD_step is None or abs(target_OPD - start_OPD) <= max_OPD_step:
        segment_targets = [target_OPD]
    else:
        n_segments = int(np.ceil(abs(target_OPD - start_OPD) / max_OPD_step))
        segment_targets = list(np.linspace(start_OPD, target_OPD, n_segments + 1)[1:])

    current_M1 = np.array(M1, dtype=float)
    current_M2 = np.array(M2, dtype=float)
    current_M3 = np.array(M3, dtype=float)
    current_M4 = np.array(M4, dtype=float)

    segment_plans = []
    final_res = None
    x_opt = None

    if return_actuation_plan and auto_recenter_start:
        start_diagnostics = actuation_constraint_diagnostics(
            x_start, current_M1, current_M2, current_M3, current_M4,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            expected_reflections=None,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=constraint_tolerance
        )

        if not start_diagnostics["ok"]:
            x_recentered, final_res = solve_OPD_configuration(
                start_OPD,
                current_M1, current_M2, current_M3, current_M4,
                moving_linear_stages=moving_linear_stages,
                u_min=u_min,
                u_max=u_max,
                sigma_edge=sigma_edge,
                enforce_edge_bounds=enforce_edge_bounds,
                include_edge_ends=include_edge_ends,
                verbose=optimizer_verbose
            )

            recenter_plan = plan_actuation_path(
                x_start, x_recentered,
                current_M1, current_M2, current_M3, current_M4,
                max_axis_splits=max_axis_splits,
                max_qc_error=max_qc_error,
                max_qc_difference=max_qc_difference,
                preserve_reflection_count=preserve_reflection_count,
                motion_samples_per_step=motion_samples_per_step,
                u_min=u_min,
                u_max=u_max,
                enforce_edge_bounds=False,
                include_edge_ends=include_edge_ends,
                constraint_tolerance=recenter_constraint_tolerance
            )
            recenter_plan["target_OPD"] = start_OPD
            recenter_plan["recenter_segment"] = True
            recenter_plan["recenter_reason"] = "; ".join(start_diagnostics["failures"])
            segment_plans.append(recenter_plan)

            if recenter_plan["failure_reason"] is not None:
                x_opt = x_recentered
                actuation_plan = combine_actuation_plans(segment_plans)
                M1_opt, M2_opt, M3_opt, M4_opt = unpack_variables(
                    x_opt, current_M1, current_M2, current_M3, current_M4
                )
                return (M1_opt, M2_opt, M3_opt, M4_opt), final_res, actuation_plan

            current_M1, current_M2, current_M3, current_M4 = unpack_variables(
                x_recentered,
                current_M1, current_M2, current_M3, current_M4
            )

    for segment_target in segment_targets:
        x_segment_start = pack_variables(current_M1, current_M2, current_M3, current_M4)
        x_segment_target, final_res = solve_OPD_configuration(
            segment_target,
            current_M1, current_M2, current_M3, current_M4,
            moving_linear_stages=moving_linear_stages,
            u_min=u_min,
            u_max=u_max,
            sigma_edge=sigma_edge,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            verbose=optimizer_verbose
        )

        if return_actuation_plan:
            if n_actuation_steps is not None:
                max_axis_splits = n_actuation_steps

            segment_plan = plan_actuation_path(
                x_segment_start, x_segment_target,
                current_M1, current_M2, current_M3, current_M4,
                max_axis_splits=max_axis_splits,
                max_qc_error=max_qc_error,
                max_qc_difference=max_qc_difference,
                preserve_reflection_count=preserve_reflection_count,
                motion_samples_per_step=motion_samples_per_step,
                u_min=u_min,
                u_max=u_max,
                enforce_edge_bounds=enforce_edge_bounds,
                include_edge_ends=include_edge_ends,
                constraint_tolerance=constraint_tolerance
            )
            segment_plan["target_OPD"] = segment_target
            segment_plans.append(segment_plan)

            if segment_plan["failure_reason"] is not None:
                x_opt = x_segment_target
                break

        current_M1, current_M2, current_M3, current_M4 = unpack_variables(
            x_segment_target,
            current_M1, current_M2, current_M3, current_M4
        )
        x_opt = x_segment_target
    
    M1_opt, M2_opt, M3_opt, M4_opt = unpack_variables(x_opt, current_M1, current_M2, current_M3, current_M4)

    if not return_actuation_plan:
        return (M1_opt, M2_opt, M3_opt, M4_opt), final_res

    actuation_plan = combine_actuation_plans(segment_plans)

    return (M1_opt, M2_opt, M3_opt, M4_opt), final_res, actuation_plan

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
        "M1": list(sim_to_px(M1x, M1y, M1a)),
        "M2": list(sim_to_px(M2x, M2y, M2a)),
        "M3": list(sim_to_px(M3x, M3y, M3a)),
        "M4": list(sim_to_px(M4x, M4y, M4a)),
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

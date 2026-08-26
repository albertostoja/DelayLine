import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
import json
import itertools
import math
import time
from pathlib import Path
from types import SimpleNamespace
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
qc_1 = np.array([-191, 159.46]) # Optimized for.    Initial calib was: ([-191, 158.24])
qc_2 = np.array([-300, 187.52]) # Optimized for.    Initial calib was: ([-300, 185.75])    

# Calculating OPD
OPD_x_start = 102.1 # This is the x-coordinate of where the mirror would be in the delay line arm of the M-Z if there was no delay line
exit_angle_mean = -0.25748389 # Mean exit angle from ArUco + Refl pts optimizations of 12 images. Used to be: -0.2523840245705327 from initial calib
OPD_cutoff_slope = -1/exit_angle_mean # Slope for 90/10 BS
OPD_end_point = np.array([-233.95478804,  169.4891394]) # Simulated point where the OPD path would end
OPD_cutoff_second_pt = np.array([OPD_end_point[0] + 100, OPD_end_point[1] + 100*OPD_cutoff_slope]) # Another point that lies on the line of OPD_end_point w/ slope: OPD_cutoff_slope
OPD_cutoff_points = np.array([[-233.95478804,  169.4891394],[OPD_cutoff_second_pt[0], OPD_cutoff_second_pt[0]]]) # Line where the OPD calculation would end

THRESHOLD = 220     # Pixel intensity threshold for reflection point detection
REFLECTION_ROI_THRESHOLDS = {
    "M1": THRESHOLD,
    "M2": THRESHOLD,
    "M3": THRESHOLD,
    "M4": 200,
}
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
    print(f"Total Laser Length: {total_length:.12f} mm")
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
    print(f"Total Laser Length: {total_length:.12f} mm")
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

    print(f"Exit slope: {metrics[0]:.12f}")
    print(f"Total length: {metrics[1]:.12f}")
    print(f"y191 error: {metrics[2]:.12f}")
    print(f"y300 error: {metrics[3]:.12f}")
    print(f"y595 error: {metrics[4]:.12f}")

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
        y191 - 159.46,
        y300 - 187.52,
        y595 - 263.48
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

def _threshold_for_roi(name, threshold):
    if isinstance(threshold, dict):
        return threshold.get(name, THRESHOLD)
    return threshold

def process_all_rois(gray_img, rois, threshold, eps=EPS, min_samples=35, show=False):
    results = {}
    for name, roi in rois.items():
        roi_threshold = _threshold_for_roi(name, threshold)
        clusters = clusters_in_roi(
            gray_img, roi,
            threshold=roi_threshold,
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
        threshold=REFLECTION_ROI_THRESHOLDS,
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
        threshold=REFLECTION_ROI_THRESHOLDS,
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

        sim_pts_px = np.asarray(sim_pts_px, float).reshape(-1, 2)
        meas_pts = np.asarray(meas_pts, float).reshape(-1, 2)

        if meas_pts.shape[0] == 0:
            r_refl.append(np.asarray(residuals_mirror, float))
            continue

        if sim_pts_px.shape[0] == 0:
            residuals_mirror.extend([DEFAULT_PEN / SIGMA_REFL] * (2 * meas_pts.shape[0]))
            r_refl.append(np.asarray(residuals_mirror, float))
            continue

        # Hungarian matching
        dists = np.linalg.norm(meas_pts[:, None, :] - sim_pts_px[None, :, :], axis=2)
        dists[~np.isfinite(dists)] = DEFAULT_PEN
        row_ind, col_ind = linear_sum_assignment(dists)

        for r_idx, s_idx in zip(row_ind, col_ind):
            if inside_flags[s_idx]:
                du = meas_pts[r_idx, 0] - sim_pts_px[s_idx, 0]
                dv = meas_pts[r_idx, 1] - sim_pts_px[s_idx, 1]
                residuals_mirror.extend([du / SIGMA_REFL, dv / SIGMA_REFL])
            else:
                if sim_for_mirror[s_idx]["pt"] is None:
                    residuals_mirror.extend([DEFAULT_PEN / SIGMA_REFL, DEFAULT_PEN / SIGMA_REFL])
                    continue

                # Smooth world-space miss penalty
                xw, yw = sim_for_mirror[s_idx]["pt"]
                dx = xw - cx
                dy = yw - cy
                r = np.sqrt(dx*dx + dy*dy)
                overshoot = max(0.0, r - half_length)
                residuals_mirror.extend([overshoot / SIGMA_REFL, overshoot / SIGMA_REFL])

        matched_rows = set(row_ind.tolist())
        for r_idx in range(meas_pts.shape[0]):
            if r_idx not in matched_rows:
                residuals_mirror.extend([DEFAULT_PEN / SIGMA_REFL, DEFAULT_PEN / SIGMA_REFL])

        r_refl.append(np.asarray(residuals_mirror, float))

    r_refl_pts = np.concatenate(r_refl) if r_refl else np.array([])

    # penalize one extra inside reflection
    r_extra_count = np.array([0.0], dtype=float)
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

def center_quadcells_residuals(angles, M1, M2, M3, M4, target_reflections, u_min=0.1, u_max=0.9, sigma_edge=0.1):
    M1_new = np.array([M1[0], M1[1], angles[0]], dtype=float)
    M2_new = np.array([M2[0], M2[1], angles[1]], dtype=float)
    M3_new = np.array([M3[0], M3[1], angles[2]], dtype=float)
    M4_new = np.array([M4[0], M4[1], angles[3]], dtype=float)

    mirrors = build_mirrors(M1_new, M2_new, M3_new, M4_new)
    reflection_data = trace_reflections(laser_start, laser_angle, mirrors)

    n_reflections = len(reflection_data)

    if n_reflections != target_reflections:
        return np.full(2 + max(target_reflections - 2, 0), 1e6, dtype=float)

    g = _simulation_metrics(
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

def fixed_reflection_edge_penalties_from_variables(x, M1, M2, M3, M4,
                                                   expected_count,
                                                   u_min=0.1,
                                                   u_max=0.9,
                                                   include_ends=False,
                                                   missing_penalty=1.0):
    penalties = reflection_edge_penalties_from_variables(
        x, M1, M2, M3, M4,
        u_min=u_min,
        u_max=u_max,
        include_ends=include_ends
    )

    if len(penalties) >= expected_count:
        return penalties[:expected_count]

    return np.pad(
        penalties,
        (0, expected_count - len(penalties)),
        mode="constant",
        constant_values=missing_penalty
    )

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
                            max_qc_difference=None,
                            tolerance=0.0):
    difference_ok = (
        True if max_qc_difference is None
        else abs(qc1_error - qc2_error) <= max_qc_difference + tolerance
    )
    return (
        abs(qc1_error) <= max_qc_error + tolerance and
        abs(qc2_error) <= max_qc_error + tolerance and
        difference_ok
    )

def actuation_constraint_diagnostics(x, M1, M2, M3, M4,
                                     max_qc_error=2.0,
                                     max_qc_difference=None,
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
        u_min=u_min,
        u_max=u_max,
        include_ends=include_edge_ends
    )

    failures = []
    if not np.isfinite(qc1_error) or not np.isfinite(qc2_error):
        failures.append("quadcell metric is not finite")
    if abs(qc1_error) > max_qc_error + constraint_tolerance:
        failures.append(f"QC1 offset {qc1_error:.4g} exceeds {max_qc_error}")
    if abs(qc2_error) > max_qc_error + constraint_tolerance:
        failures.append(f"QC2 offset {qc2_error:.4g} exceeds {max_qc_error}")
    if max_qc_difference is not None and abs(qc1_error - qc2_error) > max_qc_difference + constraint_tolerance:
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
                                          max_qc_difference=None,
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
                                 max_qc_difference=None,
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

def motion_stays_within_constraints(x_previous, x_current, M1, M2, M3, M4,
                                    max_qc_error=2.0,
                                    max_qc_difference=None,
                                    expected_reflections=None,
                                    motion_samples_per_step=25,
                                    u_min=0.1,
                                    u_max=0.9,
                                    enforce_edge_bounds=True,
                                    include_edge_ends=False,
                                    constraint_tolerance=0.0):
    x_previous = np.array(x_previous, dtype=float)
    x_current = np.array(x_current, dtype=float)
    delta = x_current - x_previous

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
                             max_qc_difference=None,
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
        max(0.0, abs(qc2_error) - max_qc_error) / qc_slack
    ])
    if max_qc_difference is not None:
        residuals.append(max(0.0, abs(qc1_error - qc2_error) - max_qc_difference) / qc_slack)

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
                        max_qc_difference=None,
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
                                 max_qc_difference=None,
                                 motion_samples_per_step=25,
                                 u_min=0.1,
                                 u_max=0.9,
                                 include_edge_ends=False,
                                 search_mode=None,
                                 split_count=None,
                                 failure_reason=None):
    start_metrics = metrics_from_variables(x_start, M1, M2, M3, M4)
    target_metrics = metrics_from_variables(x_target, M1, M2, M3, M4)
    start_qc1_error, start_qc2_error = start_metrics[2], start_metrics[4]
    start_mirrors = unpack_variables(x_start, M1, M2, M3, M4)
    target_mirrors = unpack_variables(x_target, M1, M2, M3, M4)

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
        "start_mirrors": start_mirrors,
        "target_mirrors": target_mirrors,
        "start_OPD": start_metrics[1],
        "target_OPD": target_metrics[1],
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
    start_qc1_error = actuation_plan.get(
        "start_qc1_error",
        actuation_plan.get("initial_qc1_error", np.nan)
    )
    start_qc2_error = actuation_plan.get(
        "start_qc2_error",
        actuation_plan.get("initial_qc2_error", np.nan)
    )
    start_qc_difference = actuation_plan.get(
        "start_qc_difference",
        start_qc1_error - start_qc2_error
        if np.isfinite(start_qc1_error) and np.isfinite(start_qc2_error)
        else np.nan
    )
    qc1_errors = [start_qc1_error] + [step.get("qc1_error", np.nan) for step in steps]
    qc2_errors = [start_qc2_error] + [step.get("qc2_error", np.nan) for step in steps]
    qc_differences = [start_qc_difference] + [
        step.get(
            "qc_difference",
            step.get("qc1_error", np.nan) - step.get("qc2_error", np.nan)
            if np.isfinite(step.get("qc1_error", np.nan)) and np.isfinite(step.get("qc2_error", np.nan))
            else np.nan
        )
        for step in steps
    ]

    max_qc_error = actuation_plan.get(
        "max_qc_error",
        actuation_plan.get("qc_reacquire_limit", actuation_plan.get("stage_qc_limit", 2.0))
    )
    max_qc_difference = actuation_plan.get("max_qc_difference", None)

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

    if show_difference and max_qc_difference is not None and max_qc_difference != max_qc_error:
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
    linear_u_min = actuation_plan.get("linear_u_min", u_min)
    linear_u_max = actuation_plan.get("linear_u_max", u_max)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(step_numbers, min_us, marker="o", linewidth=1.5, label="minimum reflection u")
    ax.plot(step_numbers, max_us, marker="o", linewidth=1.5, label="maximum reflection u")
    ax.plot(step_numbers, margins, marker=".", linewidth=1.0, linestyle="--", label="closest edge margin")
    ax.axhline(linear_u_min, color="gray", linestyle="--", linewidth=1, label="linear u bounds")
    ax.axhline(linear_u_max, color="gray", linestyle="--", linewidth=1)
    if linear_u_min != u_min or linear_u_max != u_max:
        ax.axhline(u_min, color="black", linestyle=":", linewidth=1, label="recenter u bounds")
        ax.axhline(u_max, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Actuator step")
    ax.set_ylabel("Reflection position u")
    ax.set_title("Reflection Positions During Actuation")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax

def plot_choose_OPD_quadcell_overlay(actuation_plan, show_difference=True):
    """Plot the planned quadcell offsets from a choose_OPD actuation plan."""
    return plot_actuation_quadcell_offsets(
        actuation_plan,
        show_difference=show_difference
    )

def plot_choose_OPD_reflection_u_overlay(actuation_plan):
    """Plot planned reflection-u positions from a choose_OPD actuation plan."""
    return plot_actuation_reflection_u(actuation_plan)

def _actuation_plan_frame_records(actuation_plan, include_start=True):
    records = []
    steps = actuation_plan.get("steps", [])

    if include_start and actuation_plan.get("start_mirrors") is not None:
        records.append({
            "step": 0,
            "mirrors": actuation_plan["start_mirrors"],
            "actuator": "start",
            "command_value": 0.0,
            "OPD": actuation_plan.get("start_OPD"),
            "qc1_error": actuation_plan.get("start_qc1_error"),
            "qc2_error": actuation_plan.get("start_qc2_error"),
            "reflection_count": actuation_plan.get("start_reflections"),
        })

    for step in steps:
        records.append(step)

    return records

def render_actuation_plan_frame(actuation_plan, frame_index=0, include_start=True,
                                figsize=(10, 5), dpi=120,
                                xlim=(-320, 230), ylim=(-15, 215),
                                qc_window=None, draw_mount_outline=True):
    """Render one choose_OPD actuation-plan waypoint as a matplotlib figure."""
    records = _actuation_plan_frame_records(actuation_plan, include_start=include_start)
    if len(records) == 0:
        raise ValueError("actuation_plan has no frames to render.")
    if frame_index < 0 or frame_index >= len(records):
        raise IndexError(f"frame_index {frame_index} outside 0..{len(records) - 1}.")

    record = records[frame_index]
    M1, M2, M3, M4 = record["mirrors"]
    mirrors = build_mirrors(M1, M2, M3, M4)
    laser_path, opd, reflection_count = simulate_laser_with_length(
        laser_start,
        laser_angle,
        mirrors
    )

    if qc_window is None:
        qc_window = actuation_plan.get("max_qc_error", 2.0)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if draw_mount_outline:
        doubled_lines, orthogonal_lines = process_mirrors(mirrors)
        for mount_line in doubled_lines + orthogonal_lines:
            ax.plot(
                [mount_line[0][0], mount_line[1][0]],
                [mount_line[0][1], mount_line[1][1]],
                linewidth=0.6,
                color="0.65",
                alpha=0.8
            )

    for mirror_index, mirror in enumerate(mirrors, start=1):
        ax.plot(
            [mirror[0][0], mirror[1][0]],
            [mirror[0][1], mirror[1][1]],
            color="black",
            linewidth=2.4
        )
        center = np.mean(np.array(mirror, dtype=float), axis=0)
        ax.text(center[0], center[1] + 5.0, f"M{mirror_index}",
                fontsize=8, ha="center", va="bottom")

    if len(laser_path) >= 2:
        path = np.array(laser_path, dtype=float)
        for i in range(len(path) - 1):
            is_exit_segment = i == len(path) - 2
            ax.plot(
                [path[i, 0], path[i + 1, 0]],
                [path[i, 1], path[i + 1, 1]],
                color="tab:red" if not is_exit_segment else "tab:orange",
                linewidth=1.6 if not is_exit_segment else 1.2,
                linestyle="-" if not is_exit_segment else "--",
                alpha=0.95
            )
        if len(path) > 2:
            ax.scatter(
                path[1:-1, 0],
                path[1:-1, 1],
                s=18,
                color="tab:red",
                zorder=4,
                label="reflection points"
            )

    ax.scatter([laser_start[0]], [laser_start[1]], color="tab:red", s=22, label="laser")
    ax.plot([qc_1[0], qc_1[0]], [qc_1[1] - qc_window, qc_1[1] + qc_window],
            linewidth=4, color="tab:blue", label="QC1 window")
    ax.plot([qc_2[0], qc_2[0]], [qc_2[1] - qc_window, qc_2[1] + qc_window],
            linewidth=4, color="tab:purple", label="QC2 window")
    ax.scatter([qc_1[0], qc_2[0]], [qc_1[1], qc_2[1]],
               color=["tab:blue", "tab:purple"], s=18, zorder=5)

    step_no = record.get("step", frame_index)
    actuator = record.get("actuator")
    command_value = record.get("command_value")
    displayed_opd = record.get("OPD", opd)
    displayed_reflections = record.get("reflection_count", reflection_count)
    qc1 = record.get("qc1_error")
    qc2 = record.get("qc2_error")

    title_parts = [f"step {step_no}"]
    if actuator is not None:
        title_parts.append(str(actuator))
    if command_value is not None:
        title_parts.append(f"cmd={float(command_value):.4g}")
    if displayed_opd is not None:
        title_parts.append(f"OPD={float(displayed_opd):.3f}")
    if displayed_reflections is not None:
        title_parts.append(f"N_R={int(displayed_reflections)}")
    if qc1 is not None and qc2 is not None:
        title_parts.append(f"qc=({float(qc1):.3f},{float(qc2):.3f})")
    ax.set_title(" | ".join(title_parts), fontsize=10)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig, ax

def _figure_to_rgb_array(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    return np.asarray(fig.canvas.buffer_rgba()).reshape(height, width, 4)[:, :, :3].copy()

def save_actuation_plan_simulation_gif(actuation_plan, output_path="choose_OPD_actuation.gif",
                                       fps=8, include_start=True,
                                       figsize=(10, 5), dpi=120,
                                       xlim=(-320, 230), ylim=(-15, 215),
                                       qc_window=None, draw_mount_outline=True):
    """Save a GIF showing the simulated geometry at each actuation-plan waypoint."""
    if fps <= 0:
        raise ValueError("fps must be positive.")

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Saving GIFs requires Pillow. Install it with `pip install pillow`.") from exc

    records = _actuation_plan_frame_records(actuation_plan, include_start=include_start)
    if len(records) == 0:
        raise ValueError("actuation_plan has no frames to save.")

    frames = []
    for frame_index in range(len(records)):
        fig, _ = render_actuation_plan_frame(
            actuation_plan,
            frame_index=frame_index,
            include_start=include_start,
            figsize=figsize,
            dpi=dpi,
            xlim=xlim,
            ylim=ylim,
            qc_window=qc_window,
            draw_mount_outline=draw_mount_outline
        )
        frames.append(Image.fromarray(_figure_to_rgb_array(fig)))
        plt.close(fig)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / fps))
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0
    )
    return str(output_path)

def save_choose_OPD_actuation_gif(actuation_plan, output_path="choose_OPD_actuation.gif",
                                  fps=8, **kwargs):
    """Alias for saving a choose_OPD actuation-plan simulation GIF."""
    return save_actuation_plan_simulation_gif(
        actuation_plan,
        output_path=output_path,
        fps=fps,
        **kwargs
    )

def try_one_actuator_sequence(x_start, x_target, axis_sequence, M1, M2, M3, M4,
                              max_qc_error=2.0,
                              max_qc_difference=None,
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

def max_valid_single_axis_fraction(x_current, x_target, axis_index, M1, M2, M3, M4,
                                   max_qc_error=2.0,
                                   max_qc_difference=None,
                                   expected_reflections=None,
                                   motion_samples_per_step=25,
                                   u_min=0.1,
                                   u_max=0.9,
                                   enforce_edge_bounds=True,
                                   include_edge_ends=False,
                                   constraint_tolerance=0.0,
                                   scan_samples=40,
                                   zero_tol=1e-10):
    amount = float(np.array(x_target, dtype=float)[axis_index] - np.array(x_current, dtype=float)[axis_index])
    if abs(amount) <= zero_tol:
        return 0.0

    def valid_at_fraction(fraction):
        x_next = variables_with_axis_move(x_current, axis_index, amount * fraction)
        return one_actuator_motion_is_valid(
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
        )

    if valid_at_fraction(1.0):
        return 1.0

    last_good = 0.0
    first_bad = None
    for fraction in np.linspace(0.0, 1.0, scan_samples + 1)[1:]:
        if valid_at_fraction(fraction):
            last_good = fraction
        else:
            first_bad = fraction
            break

    if first_bad is None:
        return last_good

    lo = last_good
    hi = first_bad
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if valid_at_fraction(mid):
            lo = mid
        else:
            hi = mid

    return lo

def try_greedy_one_actuator_path(x_start, x_target, M1, M2, M3, M4,
                                 max_steps=256,
                                 max_qc_error=2.0,
                                 max_qc_difference=None,
                                 expected_reflections=None,
                                 motion_samples_per_step=25,
                                 u_min=0.1,
                                 u_max=0.9,
                                 enforce_edge_bounds=True,
                                 include_edge_ends=False,
                                 constraint_tolerance=0.0,
                                 zero_tol=1e-9):
    x_current = np.array(x_start, dtype=float).copy()
    x_target = np.array(x_target, dtype=float)
    steps = []
    variable_scale = np.array([10.0, 0.1, 10.0, 0.1, 10.0, 0.1, 10.0, 0.1], dtype=float)

    for _ in range(max_steps):
        remaining = x_target - x_current
        active_axes = np.flatnonzero(np.abs(remaining) > zero_tol)
        if len(active_axes) == 0:
            return steps

        current_distance = np.linalg.norm(remaining / variable_scale)
        best_candidate = None

        for axis_index in active_axes:
            max_fraction = max_valid_single_axis_fraction(
                x_current, x_target, int(axis_index), M1, M2, M3, M4,
                max_qc_error=max_qc_error,
                max_qc_difference=max_qc_difference,
                expected_reflections=expected_reflections,
                motion_samples_per_step=motion_samples_per_step,
                u_min=u_min,
                u_max=u_max,
                enforce_edge_bounds=enforce_edge_bounds,
                include_edge_ends=include_edge_ends,
                constraint_tolerance=constraint_tolerance,
                scan_samples=max(motion_samples_per_step * 2, 20),
                zero_tol=zero_tol
            )
            if max_fraction <= 1e-6:
                continue

            x_next = variables_with_axis_move(
                x_current,
                int(axis_index),
                remaining[axis_index] * max_fraction
            )
            next_distance = np.linalg.norm((x_target - x_next) / variable_scale)
            progress = current_distance - next_distance
            if progress <= 1e-10:
                continue

            candidate = (progress, max_fraction, int(axis_index), x_next)
            if best_candidate is None or candidate[:2] > best_candidate[:2]:
                best_candidate = candidate

        if best_candidate is None:
            return None

        _, _, axis_index, x_next = best_candidate
        steps.append(make_actuation_step(
            len(steps) + 1,
            np.linalg.norm(x_next - x_start) / max(np.linalg.norm(x_target - x_start), 1e-12),
            x_current,
            x_next,
            M1, M2, M3, M4,
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

    if np.allclose(x_current, x_target, atol=1e-8, rtol=0):
        return steps

    return None

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

def validate_actuation_steps(steps, x_start, M1, M2, M3, M4,
                             max_qc_error=2.0,
                             max_qc_difference=None,
                             expected_reflections=None,
                             motion_samples_per_step=25,
                             u_min=0.1,
                             u_max=0.9,
                             enforce_edge_bounds=True,
                             include_edge_ends=False,
                             constraint_tolerance=0.0):
    x_previous = np.array(x_start, dtype=float).copy()

    for step in steps:
        x_current = pack_variables(*step["mirrors"])
        if not one_actuator_motion_is_valid(
            x_previous, x_current, M1, M2, M3, M4,
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
            return False, f"step {step['step']} failed dense validation"
        x_previous = x_current

    return True, None

def plan_actuation_path(x_start, x_target, M1, M2, M3, M4,
                        max_axis_splits=64,
                        max_qc_error=2.0,
                        max_qc_difference=None,
                        preserve_reflection_count=True,
                        motion_samples_per_step=25,
                        u_min=0.1,
                        u_max=0.9,
                        enforce_edge_bounds=True,
                        include_edge_ends=False,
                        constraint_tolerance=0.05,
                        zero_tol=1e-9,
                        verbose=False,
                        profile_callback=None):
    def profile_path(message):
        if profile_callback is not None:
            profile_callback(message)

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

    phase_t0 = time.perf_counter()
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
            profile_path(
                f"path search full_orders dt={time.perf_counter() - phase_t0:.3f}s "
                f"orders={len(full_axis_orders)} success=True steps={len(steps)}"
            )
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
    profile_path(
        f"path search full_orders dt={time.perf_counter() - phase_t0:.3f}s "
        f"orders={len(full_axis_orders)} success=False"
    )

    phase_t0 = time.perf_counter()
    greedy_steps = try_greedy_one_actuator_path(
        x_start, x_target, M1, M2, M3, M4,
        max_steps=max_axis_splits * max(len(active_axes), 1),
        max_qc_error=max_qc_error,
        max_qc_difference=max_qc_difference,
        expected_reflections=expected_reflections,
        motion_samples_per_step=motion_samples_per_step,
        u_min=u_min,
        u_max=u_max,
        enforce_edge_bounds=enforce_edge_bounds,
        include_edge_ends=include_edge_ends,
        constraint_tolerance=constraint_tolerance,
        zero_tol=zero_tol
    )
    if greedy_steps is not None:
        profile_path(
            f"path search greedy dt={time.perf_counter() - phase_t0:.3f}s "
            f"success=True steps={len(greedy_steps)}"
        )
        return build_actuation_plan_summary(
            greedy_steps, x_start, x_target, M1, M2, M3, M4,
            start_reflections, target_reflections, start_within_constraints,
            expected_reflections,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            motion_samples_per_step=motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            include_edge_ends=include_edge_ends,
            search_mode="greedy_single_actuator_moves",
            split_count=len(greedy_steps)
        )
    profile_path(
        f"path search greedy dt={time.perf_counter() - phase_t0:.3f}s success=False"
    )

    phase_t0 = time.perf_counter()
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
                qc_terms = [abs(qc1_error), abs(qc2_error)]
                if max_qc_difference is not None:
                    qc_terms.append(abs(qc1_error - qc2_error))
                order_max_qc = max(qc_terms)
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
            profile_path(
                f"path search split dt={time.perf_counter() - phase_t0:.3f}s "
                f"split_count={split_count} success=True steps={len(steps)}"
            )
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

    profile_path(
        f"path search split dt={time.perf_counter() - phase_t0:.3f}s "
        f"max_axis_splits={max_axis_splits} success=False"
    )
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
        "start_mirrors": first.get("start_mirrors"),
        "target_mirrors": segment_plans[-1].get("target_mirrors"),
        "start_OPD": first.get("start_OPD"),
        "target_OPD": segment_plans[-1].get("target_OPD"),
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
                  include_edge_ends=False,
                  expected_edge_count=None):
    g = metrics_from_variables(x, M1, M2, M3, M4)

    r_OPD = (g[1] - target_OPD) / SIGMA_OPD
    r_qc1 = g[2] / SIGMA_QC
    r_qc2 = g[4] / SIGMA_QC

    residuals = [r_OPD, r_qc1, r_qc2]

    if enforce_edge_bounds:
        if expected_edge_count is None:
            edge_penalties = reflection_edge_penalties_from_variables(
                x, M1, M2, M3, M4,
                u_min=u_min,
                u_max=u_max,
                include_ends=include_edge_ends
            )
        else:
            edge_penalties = fixed_reflection_edge_penalties_from_variables(
                x, M1, M2, M3, M4,
                expected_count=expected_edge_count,
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
                           include_edge_ends=False,
                           expected_edge_count=None):
    x_full = expand_selected_variables(x_selected, x_base, variable_indices)
    return OPD_residuals(
        x_full, target_OPD, M1, M2, M3, M4,
        u_min=u_min,
        u_max=u_max,
        sigma_edge=sigma_edge,
        enforce_edge_bounds=enforce_edge_bounds,
        include_edge_ends=include_edge_ends,
        expected_edge_count=expected_edge_count
    )

def solve_OPD_configuration(target_OPD, M1, M2, M3, M4,
                            moving_linear_stages=("M1",),
                            variable_bounds=None,
                            u_min=0.1,
                            u_max=0.9,
                            sigma_edge=0.02,
                            enforce_edge_bounds=True,
                            include_edge_ends=False,
                            verbose=0):
    x0 = pack_variables(M1, M2, M3, M4)
    variable_indices = selected_OPD_variable_indices(moving_linear_stages)
    x0_selected = x0[variable_indices]
    expected_edge_count = len(reflection_us_from_variables(
        x0, M1, M2, M3, M4,
        include_ends=include_edge_ends
    )) if enforce_edge_bounds else None

    if variable_bounds is None:
        bounds = (-np.inf, np.inf)
    else:
        lower_full, upper_full = variable_bounds
        lower_selected = np.asarray(lower_full, dtype=float)[variable_indices]
        upper_selected = np.asarray(upper_full, dtype=float)[variable_indices]
        x0_selected = np.clip(x0_selected, lower_selected, upper_selected)
        bounds = (lower_selected, upper_selected)

    res = least_squares(
        fun=lambda x: OPD_residuals_selected(
            x, x0, variable_indices,
            target_OPD, M1, M2, M3, M4,
            u_min=u_min,
            u_max=u_max,
            sigma_edge=sigma_edge,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            expected_edge_count=expected_edge_count
        ),
        x0=x0_selected,
        loss="linear",
        f_scale=1.0,
        bounds=bounds,
        verbose=verbose,
        x_scale='jac',
        max_nfev=4000,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10
    )

    x_opt = expand_selected_variables(res.x, x0, variable_indices)
    return x_opt, res

LINEAR_STAGE_TRAVEL_MM = 24.0

def linear_stage_x_bounds(M1, M2, M3, M4,
                          M1_linear_loc, M2_linear_loc, M3_linear_loc):
    for name, loc in [
        ("M1_linear_loc", M1_linear_loc),
        ("M2_linear_loc", M2_linear_loc),
        ("M3_linear_loc", M3_linear_loc)
    ]:
        if loc < 0 or loc > LINEAR_STAGE_TRAVEL_MM:
            raise ValueError(f"{name} must be between 0 and {LINEAR_STAGE_TRAVEL_MM} mm.")

    lower = np.full(8, -np.inf, dtype=float)
    upper = np.full(8, np.inf, dtype=float)

    # M1 and M3 stage motion increases simulation x as stage location increases.
    lower[0] = M1[0] - M1_linear_loc
    upper[0] = M1[0] + (LINEAR_STAGE_TRAVEL_MM - M1_linear_loc)

    lower[4] = M3[0] - M3_linear_loc
    upper[4] = M3[0] + (LINEAR_STAGE_TRAVEL_MM - M3_linear_loc)

    # M2 is mounted oppositely: increasing stage location moves in -x.
    lower[2] = M2[0] - (LINEAR_STAGE_TRAVEL_MM - M2_linear_loc)
    upper[2] = M2[0] + M2_linear_loc

    return lower, upper

def update_linear_stage_locs(previous_mirrors, current_mirrors,
                             M1_linear_loc, M2_linear_loc, M3_linear_loc):
    prev_M1, prev_M2, prev_M3, _ = previous_mirrors
    curr_M1, curr_M2, curr_M3, _ = current_mirrors

    M1_linear_loc += curr_M1[0] - prev_M1[0]
    M2_linear_loc -= curr_M2[0] - prev_M2[0]
    M3_linear_loc += curr_M3[0] - prev_M3[0]

    return (
        float(np.clip(M1_linear_loc, 0, LINEAR_STAGE_TRAVEL_MM)),
        float(np.clip(M2_linear_loc, 0, LINEAR_STAGE_TRAVEL_MM)),
        float(np.clip(M3_linear_loc, 0, LINEAR_STAGE_TRAVEL_MM))
    )

def set_OPD_result_full_x(res, M1, M2, M3, M4):
    if res is None:
        return res

    # Consistent choose_OPD result layout, independent of which stage moved last:
    # [M1x, M2x, M3x, M1 angle, M2 angle, M3 angle, M4 angle]
    res.x = np.array([
        M1[0], M2[0], M3[0],
        M1[2], M2[2], M3[2], M4[2]
    ], dtype=float)
    return res

def linear_stage_x_axis(stage_name):
    axis_indices = {"M1": 0, "M2": 2, "M3": 4}
    if stage_name not in axis_indices:
        raise ValueError(f"Unknown linear stage: {stage_name}")
    return axis_indices[stage_name]

def linear_stage_available_dx(stage_name, target_direction, M1_linear_loc, M2_linear_loc, M3_linear_loc):
    M1_linear_loc = float(M1_linear_loc)
    M2_linear_loc = float(M2_linear_loc)
    M3_linear_loc = float(M3_linear_loc)

    if stage_name == "M1":
        forward = max(0.0, LINEAR_STAGE_TRAVEL_MM - M1_linear_loc)
        backward = max(0.0, M1_linear_loc)
        return forward if target_direction > 0 else -backward
    if stage_name == "M2":
        forward = max(0.0, LINEAR_STAGE_TRAVEL_MM - M2_linear_loc)
        backward = max(0.0, M2_linear_loc)
        return -forward if target_direction > 0 else backward
    if stage_name == "M3":
        forward = max(0.0, LINEAR_STAGE_TRAVEL_MM - M3_linear_loc)
        backward = max(0.0, M3_linear_loc)
        return forward if target_direction > 0 else -backward
    raise ValueError(f"Unknown linear stage: {stage_name}")

def OPD_from_variables(x, M1, M2, M3, M4):
    return metrics_from_variables(x, M1, M2, M3, M4)[1]

def OPD_brackets_target(OPD_a, OPD_b, target_OPD):
    return (OPD_a - target_OPD) * (OPD_b - target_OPD) <= 0

def find_linear_fraction_to_target(x_start, axis_index, dx_limit, target_OPD, M1, M2, M3, M4):
    OPD_start = OPD_from_variables(x_start, M1, M2, M3, M4)
    if abs(OPD_start - target_OPD) <= 1e-12:
        return 0.0

    x_limit = variables_with_axis_move(x_start, axis_index, dx_limit)
    OPD_limit = OPD_from_variables(x_limit, M1, M2, M3, M4)

    if not OPD_brackets_target(OPD_start, OPD_limit, target_OPD):
        return None

    lo = 0.0
    hi = 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        x_mid = variables_with_axis_move(x_start, axis_index, dx_limit * mid)
        OPD_mid = OPD_from_variables(x_mid, M1, M2, M3, M4)
        if OPD_brackets_target(OPD_start, OPD_mid, target_OPD):
            hi = mid
        else:
            lo = mid
    return hi

def find_max_valid_linear_fraction(x_start, axis_index, dx_limit, M1, M2, M3, M4,
                                   max_qc_error=2.0,
                                   max_qc_difference=None,
                                   expected_reflections=None,
                                   u_min=0.1,
                                   u_max=0.9,
                                   enforce_edge_bounds=True,
                                   include_edge_ends=False,
                                   constraint_tolerance=0.0,
                                   scan_samples=80):
    if abs(dx_limit) <= 1e-12:
        return 0.0

    last_good = 0.0
    first_bad = None

    for fraction in np.linspace(0.0, 1.0, scan_samples + 1)[1:]:
        x_trial = variables_with_axis_move(x_start, axis_index, dx_limit * fraction)
        if state_satisfies_actuation_constraints(
            x_trial, M1, M2, M3, M4,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            expected_reflections=expected_reflections,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=constraint_tolerance
        ):
            last_good = fraction
        else:
            first_bad = fraction
            break

    if first_bad is None:
        return 1.0

    lo = last_good
    hi = first_bad
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        x_mid = variables_with_axis_move(x_start, axis_index, dx_limit * mid)
        if state_satisfies_actuation_constraints(
            x_mid, M1, M2, M3, M4,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            expected_reflections=expected_reflections,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=constraint_tolerance
        ):
            lo = mid
        else:
            hi = mid

    return lo

def append_axis_steps(steps, x_start, x_target, M1, M2, M3, M4,
                      max_qc_error=2.0,
                      max_qc_difference=None,
                      motion_samples_per_step=25,
                      u_min=0.1,
                      u_max=0.9,
                      enforce_edge_bounds=True,
                      include_edge_ends=False,
                      constraint_tolerance=0.0,
                      zero_tol=1e-10):
    x_current = np.array(x_start, dtype=float).copy()
    x_target = np.array(x_target, dtype=float)
    delta = x_target - x_current
    active_axes = np.flatnonzero(np.abs(delta) > zero_tol)

    for axis_index in active_axes:
        x_next = x_current.copy()
        x_next[axis_index] = x_target[axis_index]
        steps.append(make_actuation_step(
            len(steps) + 1,
            1.0,
            x_current,
            x_next,
            M1, M2, M3, M4,
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

    return x_current

def append_constrained_path_steps(steps, x_start, x_target, M1, M2, M3, M4,
                                  max_axis_splits=64,
                                  max_qc_error=2.0,
                                  max_qc_difference=None,
                                  preserve_reflection_count=True,
                                  motion_samples_per_step=25,
                                  u_min=0.1,
                                  u_max=0.9,
                                  enforce_edge_bounds=True,
                                  include_edge_ends=False,
                                  constraint_tolerance=0.0,
                                  profile_callback=None):
    x_current = np.array(x_start, dtype=float).copy()
    x_target = np.array(x_target, dtype=float)
    expected_reflections = None
    if preserve_reflection_count:
        expected_reflections = get_reflection_count(*unpack_variables(x_start, M1, M2, M3, M4))
    start_reflections = get_reflection_count(*unpack_variables(x_start, M1, M2, M3, M4))
    target_reflections = get_reflection_count(*unpack_variables(x_target, M1, M2, M3, M4))
    start_diagnostics = actuation_constraint_diagnostics(
        x_start, M1, M2, M3, M4,
        max_qc_error=max_qc_error,
        max_qc_difference=max_qc_difference,
        expected_reflections=expected_reflections,
        u_min=u_min,
        u_max=u_max,
        enforce_edge_bounds=enforce_edge_bounds,
        include_edge_ends=include_edge_ends,
        constraint_tolerance=constraint_tolerance
    )

    if np.allclose(x_current, x_target, atol=1e-10, rtol=0):
        return x_current, build_actuation_plan_summary(
            [], x_start, x_target, M1, M2, M3, M4,
            start_reflections,
            target_reflections,
            start_diagnostics["ok"],
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

    path_plan = plan_actuation_path(
        x_start,
        x_target,
        M1, M2, M3, M4,
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

    if path_plan["failure_reason"] is not None:
        return np.array(x_start, dtype=float).copy(), path_plan

    for step in path_plan["steps"]:
        step_new = dict(step)
        step_new["step"] = len(steps) + 1
        steps.append(step_new)

    if len(path_plan["steps"]) == 0:
        return np.array(x_target, dtype=float).copy(), path_plan

    return pack_variables(*path_plan["steps"][-1]["mirrors"]), path_plan

def append_constrained_path_steps_fast_then_dense(
        steps, x_start, x_target, M1, M2, M3, M4,
        max_axis_splits=64,
        max_qc_error=2.0,
        max_qc_difference=None,
        preserve_reflection_count=True,
        motion_samples_per_step=25,
        fast_motion_samples_per_step=5,
        u_min=0.1,
        u_max=0.9,
        enforce_edge_bounds=True,
        include_edge_ends=False,
        constraint_tolerance=0.0,
        profile_callback=None):
    def profile_path(message):
        if profile_callback is not None:
            profile_callback(message)

    dense_samples = motion_samples_per_step
    fast_samples = min(max(1, int(fast_motion_samples_per_step)), dense_samples)

    if fast_samples >= dense_samples:
        return append_constrained_path_steps(
            steps, x_start, x_target, M1, M2, M3, M4,
            max_axis_splits=max_axis_splits,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            preserve_reflection_count=preserve_reflection_count,
            motion_samples_per_step=dense_samples,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=constraint_tolerance,
            profile_callback=profile_callback
        )

    trial_steps = []
    phase_t0 = time.perf_counter()
    x_fast, fast_plan = append_constrained_path_steps(
        trial_steps, x_start, x_target, M1, M2, M3, M4,
        max_axis_splits=max_axis_splits,
        max_qc_error=max_qc_error,
        max_qc_difference=max_qc_difference,
        preserve_reflection_count=preserve_reflection_count,
        motion_samples_per_step=fast_samples,
        u_min=u_min,
        u_max=u_max,
        enforce_edge_bounds=enforce_edge_bounds,
        include_edge_ends=include_edge_ends,
        constraint_tolerance=constraint_tolerance,
        profile_callback=lambda msg: profile_path(f"fast {msg}")
    )
    profile_path(
        f"fast path proposal dt={time.perf_counter() - phase_t0:.3f}s "
        f"samples={fast_samples} steps={len(trial_steps)} "
        f"failure={fast_plan['failure_reason']}"
    )

    if fast_plan["failure_reason"] is None:
        expected_reflections = None
        if preserve_reflection_count:
            expected_reflections = get_reflection_count(*unpack_variables(x_start, M1, M2, M3, M4))

        phase_t0 = time.perf_counter()
        valid, validation_reason = validate_actuation_steps(
            trial_steps, x_start, M1, M2, M3, M4,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            expected_reflections=expected_reflections,
            motion_samples_per_step=dense_samples,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=constraint_tolerance
        )
        profile_path(
            f"fast path dense validation dt={time.perf_counter() - phase_t0:.3f}s "
            f"samples={dense_samples} ok={valid} reason={validation_reason}"
        )
        if valid:
            for step in trial_steps:
                step_new = dict(step)
                step_new["step"] = len(steps) + 1
                step_new["fast_path_proposal"] = True
                step_new["dense_validated"] = True
                steps.append(step_new)
            fast_plan["fast_path_used"] = True
            fast_plan["dense_validated"] = True
            return x_fast, fast_plan

    phase_t0 = time.perf_counter()
    x_dense, dense_plan = append_constrained_path_steps(
        steps, x_start, x_target, M1, M2, M3, M4,
        max_axis_splits=max_axis_splits,
        max_qc_error=max_qc_error,
        max_qc_difference=max_qc_difference,
        preserve_reflection_count=preserve_reflection_count,
        motion_samples_per_step=dense_samples,
        u_min=u_min,
        u_max=u_max,
        enforce_edge_bounds=enforce_edge_bounds,
        include_edge_ends=include_edge_ends,
        constraint_tolerance=constraint_tolerance,
        profile_callback=lambda msg: profile_path(f"dense fallback {msg}")
    )
    profile_path(
        f"dense fallback path dt={time.perf_counter() - phase_t0:.3f}s "
        f"samples={dense_samples} failure={dense_plan['failure_reason']}"
    )
    dense_plan["fast_path_used"] = False
    return x_dense, dense_plan

def append_waypoint_constrained_path_steps(
        steps, x_start, x_target, M1, M2, M3, M4,
        max_axis_splits=64,
        max_waypoint_depth=4,
        max_qc_error=2.0,
        max_qc_difference=None,
        preserve_reflection_count=True,
        motion_samples_per_step=25,
        fast_motion_samples_per_step=5,
        u_min=0.05,
        u_max=0.95,
        enforce_edge_bounds=True,
        include_edge_ends=False,
        constraint_tolerance=0.0,
        profile_callback=None):
    """Append a constrained path, recursively splitting through mid-waypoints."""
    def profile_path(message):
        if profile_callback is not None:
            profile_callback(message)

    x_start = np.array(x_start, dtype=float)
    x_target = np.array(x_target, dtype=float)
    segment_plans = []

    def route_segment(x_a, x_b, depth):
        trial_steps = []
        x_direct, direct_plan = append_constrained_path_steps_fast_then_dense(
            trial_steps, x_a, x_b, M1, M2, M3, M4,
            max_axis_splits=max_axis_splits,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            preserve_reflection_count=preserve_reflection_count,
            motion_samples_per_step=motion_samples_per_step,
            fast_motion_samples_per_step=fast_motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=constraint_tolerance,
            profile_callback=lambda msg: profile_path(f"depth={depth} {msg}")
        )
        if direct_plan["failure_reason"] is None:
            segment_plans.append(direct_plan)
            return x_direct, trial_steps, None

        if depth >= max_waypoint_depth:
            return x_a.copy(), [], direct_plan["failure_reason"]

        x_mid = 0.5 * (x_a + x_b)
        expected_reflections = None
        if preserve_reflection_count:
            expected_reflections = get_reflection_count(*unpack_variables(x_a, M1, M2, M3, M4))
        mid_diagnostics = actuation_constraint_diagnostics(
            x_mid, M1, M2, M3, M4,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            expected_reflections=expected_reflections,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=constraint_tolerance
        )
        if not mid_diagnostics["ok"]:
            return (
                x_a.copy(),
                [],
                "Waypoint midpoint is outside path constraints: " + "; ".join(mid_diagnostics["failures"])
            )

        profile_path(f"depth={depth} direct path failed; splitting through midpoint")
        x_first, first_steps, first_failure = route_segment(x_a, x_mid, depth + 1)
        if first_failure is not None:
            return x_a.copy(), [], first_failure

        x_second, second_steps, second_failure = route_segment(x_first, x_b, depth + 1)
        if second_failure is not None:
            return x_first.copy(), first_steps, second_failure

        return x_second, first_steps + second_steps, None

    x_final, routed_steps, failure_reason = route_segment(x_start, x_target, 0)
    if failure_reason is not None:
        start_reflections = get_reflection_count(*unpack_variables(x_start, M1, M2, M3, M4))
        target_reflections = get_reflection_count(*unpack_variables(x_target, M1, M2, M3, M4))
        expected_reflections = start_reflections if preserve_reflection_count and start_reflections == target_reflections else None
        start_diagnostics = actuation_constraint_diagnostics(
            x_start, M1, M2, M3, M4,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            expected_reflections=expected_reflections,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=constraint_tolerance
        )
        return x_start.copy(), build_actuation_plan_summary(
            [], x_start, x_target, M1, M2, M3, M4,
            start_reflections,
            target_reflections,
            start_diagnostics["ok"],
            expected_reflections,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            motion_samples_per_step=motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            include_edge_ends=include_edge_ends,
            search_mode="waypoint_failed",
            split_count=None,
            failure_reason=failure_reason
        )

    for step in routed_steps:
        step_new = dict(step)
        step_new["step"] = len(steps) + 1
        step_new["waypoint_path"] = True
        steps.append(step_new)

    plan = combine_actuation_plans(segment_plans)
    if plan is None:
        start_reflections = get_reflection_count(*unpack_variables(x_start, M1, M2, M3, M4))
        target_reflections = get_reflection_count(*unpack_variables(x_target, M1, M2, M3, M4))
        expected_reflections = start_reflections if preserve_reflection_count and start_reflections == target_reflections else None
        plan = build_actuation_plan_summary(
            [], x_start, x_target, M1, M2, M3, M4,
            start_reflections,
            target_reflections,
            True,
            expected_reflections,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            motion_samples_per_step=motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            include_edge_ends=include_edge_ends,
            search_mode="waypoint_already_at_target",
            split_count=0
        )
    plan["search_mode"] = "waypoint_" + str(plan.get("search_mode"))
    plan["waypoint_depth_limit"] = int(max_waypoint_depth)
    plan["waypoint_segments"] = len(segment_plans)
    return x_final, plan

def solve_recenter_angles(x_current, M1, M2, M3, M4,
                          target_reflections,
                          max_qc_error=3.9,
                          u_min=0.1,
                          u_max=0.9,
                          sigma_edge=0.02,
                          include_edge_ends=False,
                          verbose=0,
                          profile_callback=None):
    def profile_solve(message):
        if profile_callback is not None:
            profile_callback(message)

    M1_current, M2_current, M3_current, M4_current = unpack_variables(x_current, M1, M2, M3, M4)
    theta0 = np.array([M1_current[2], M2_current[2], M3_current[2], M4_current[2]], dtype=float)

    expected_u_count = target_reflections if include_edge_ends else max(target_reflections - 2, 0)

    def x_from_angles(angles):
        x_trial = np.array(x_current, dtype=float).copy()
        x_trial[[1, 3, 5, 7]] = angles
        return x_trial

    def recenter_objective(angles):
        x_trial = x_from_angles(angles)
        g = metrics_from_variables(x_trial, M1, M2, M3, M4)
        angle_penalty = 1e-4 * np.sum((np.array(angles, dtype=float) - theta0) ** 2)
        return float(g[2] ** 2 + g[4] ** 2 + angle_penalty)

    def constrained_us(angles):
        x_trial = x_from_angles(angles)
        mirrors_trial = unpack_variables(x_trial, M1, M2, M3, M4)
        if get_reflection_count(*mirrors_trial) != target_reflections:
            return np.full(expected_u_count, -np.inf, dtype=float)

        us = reflection_us_from_variables(
            x_trial, M1, M2, M3, M4,
            include_ends=include_edge_ends
        )
        if len(us) != expected_u_count:
            return np.full(expected_u_count, -np.inf, dtype=float)
        return us

    constraints = []
    for idx in range(expected_u_count):
        constraints.append({
            "type": "ineq",
            "fun": lambda angles, i=idx: constrained_us(angles)[i] - u_min
        })
        constraints.append({
            "type": "ineq",
            "fun": lambda angles, i=idx: u_max - constrained_us(angles)[i]
        })

    phase_t0 = time.perf_counter()
    res = least_squares(
        fun=lambda th: center_quadcells_residuals(
            th,
            M1_current, M2_current, M3_current, M4_current,
            target_reflections=target_reflections,
            u_min=u_min,
            u_max=u_max,
            sigma_edge=sigma_edge
        ),
        x0=theta0,
        loss="linear",
        f_scale=1.0,
        verbose=verbose,
        x_scale='jac',
        max_nfev=800,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10
    )
    profile_solve(
        f"least_squares dt={time.perf_counter() - phase_t0:.3f}s "
        f"success={res.success}"
    )

    x_recentered = np.array(x_current, dtype=float).copy()
    x_recentered[[1, 3, 5, 7]] = res.x
    diagnostics = actuation_constraint_diagnostics(
        x_recentered, M1, M2, M3, M4,
        max_qc_error=max_qc_error,
        expected_reflections=target_reflections,
        u_min=u_min,
        u_max=u_max,
        enforce_edge_bounds=True,
        include_edge_ends=include_edge_ends,
        constraint_tolerance=0.0
    )
    if diagnostics["ok"]:
        profile_solve("least_squares accepted")
        return x_recentered, res

    phase_t0 = time.perf_counter()
    minimize_res = minimize(
        recenter_objective,
        theta0,
        method="SLSQP",
        constraints=constraints,
        options={
            "maxiter": 300,
            "ftol": 1e-12,
            "disp": bool(verbose)
        }
    )
    profile_solve(
        f"SLSQP dt={time.perf_counter() - phase_t0:.3f}s "
        f"success={minimize_res.success}"
    )

    if minimize_res.success:
        x_recentered = x_from_angles(minimize_res.x)
        diagnostics = actuation_constraint_diagnostics(
            x_recentered, M1, M2, M3, M4,
            max_qc_error=max_qc_error,
            expected_reflections=target_reflections,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=True,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=0.0
        )
        if diagnostics["ok"]:
            profile_solve("SLSQP accepted")
            return x_recentered, minimize_res

    profile_solve("no recenter solution accepted")
    failed_res = SimpleNamespace(
        x=theta0.copy(),
        success=False,
        message="No recenter solution found that satisfies reflection u bounds."
    )
    return np.array(x_current, dtype=float).copy(), failed_res

def solve_final_centered_angles(x_current, target_OPD, M1, M2, M3, M4,
                                target_reflections,
                                qc_tolerance=0.5,
                                OPD_tolerance=0.05,
                                relaxed_OPD_tolerance=0.5,
                                qc_detector_limit=3.9,
                                qc_priority=True,
                                u_min=0.1,
                                u_max=0.9,
                                include_edge_ends=False,
                                verbose=0,
                                profile_callback=None):
    def profile_solve(message):
        if profile_callback is not None:
            profile_callback(message)

    M1_current, M2_current, M3_current, M4_current = unpack_variables(x_current, M1, M2, M3, M4)
    theta0 = np.array([M1_current[2], M2_current[2], M3_current[2], M4_current[2]], dtype=float)
    expected_u_count = target_reflections if include_edge_ends else max(target_reflections - 2, 0)

    def x_from_angles(angles):
        x_trial = np.array(x_current, dtype=float).copy()
        x_trial[[1, 3, 5, 7]] = angles
        return x_trial

    def constrained_us(angles):
        x_trial = x_from_angles(angles)
        mirrors_trial = unpack_variables(x_trial, M1, M2, M3, M4)
        if get_reflection_count(*mirrors_trial) != target_reflections:
            return np.full(expected_u_count, -np.inf, dtype=float)

        us = reflection_us_from_variables(
            x_trial, M1, M2, M3, M4,
            include_ends=include_edge_ends
        )
        if len(us) != expected_u_count:
            return np.full(expected_u_count, -np.inf, dtype=float)
        return us

    def qc_values(angles):
        g = metrics_from_variables(x_from_angles(angles), M1, M2, M3, M4)
        return np.array([g[2], g[4]], dtype=float)

    def OPD_error(angles):
        return OPD_from_variables(x_from_angles(angles), M1, M2, M3, M4) - target_OPD

    def strict_objective(angles):
        angle_penalty = 1e-4 * np.sum((np.array(angles, dtype=float) - theta0) ** 2)
        return float(OPD_error(angles) ** 2 + angle_penalty)

    def qc_priority_objective(angles):
        qc = qc_values(angles)
        angle_penalty = 1e-4 * np.sum((np.array(angles, dtype=float) - theta0) ** 2)
        OPD_penalty = 0.02 * OPD_error(angles) ** 2
        return float(np.sum(qc ** 2) + OPD_penalty + angle_penalty)

    base_constraints = []
    for idx in range(expected_u_count):
        base_constraints.append({
            "type": "ineq",
            "fun": lambda angles, i=idx: constrained_us(angles)[i] - u_min
        })
        base_constraints.append({
            "type": "ineq",
            "fun": lambda angles, i=idx: u_max - constrained_us(angles)[i]
        })

    detector_constraints = list(base_constraints)
    for idx in range(2):
        detector_constraints.append({
            "type": "ineq",
            "fun": lambda angles, i=idx: qc_detector_limit - qc_values(angles)[i]
        })
        detector_constraints.append({
            "type": "ineq",
            "fun": lambda angles, i=idx: qc_detector_limit + qc_values(angles)[i]
        })

    def constrained_attempt_constraints(opd_tolerance):
        constraints = list(base_constraints)
        for idx in range(2):
            constraints.append({
                "type": "ineq",
                "fun": lambda angles, i=idx: qc_tolerance - qc_values(angles)[i]
            })
            constraints.append({
                "type": "ineq",
                "fun": lambda angles, i=idx: qc_tolerance + qc_values(angles)[i]
            })

        constraints.append({
            "type": "ineq",
            "fun": lambda angles: opd_tolerance - OPD_error(angles)
        })
        constraints.append({
            "type": "ineq",
            "fun": lambda angles: opd_tolerance + OPD_error(angles)
        })
        return constraints

    def accept_result(res, opd_tolerance, relaxed):
        if not res.success:
            return None

        x_centered = x_from_angles(res.x)
        qc1_error, qc2_error = quadcell_errors_from_variables(x_centered, M1, M2, M3, M4)
        opd_error = OPD_error(res.x)
        diagnostics = actuation_constraint_diagnostics(
            x_centered, M1, M2, M3, M4,
            max_qc_error=qc_detector_limit,
            expected_reflections=target_reflections,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=True,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=0.0
        )
        if (
            diagnostics["ok"] and
            abs(qc1_error) <= qc_tolerance and
            abs(qc2_error) <= qc_tolerance and
            abs(opd_error) <= opd_tolerance
        ):
            res.final_center_mode = "qc_priority_relaxed" if relaxed else "strict"
            res.final_center_OPD_relaxed = bool(relaxed)
            res.final_center_OPD_error = float(opd_error)
            res.final_center_qc_error = [float(qc1_error), float(qc2_error)]
            res.final_center_OPD_tolerance_used = float(opd_tolerance)
            return x_centered

        return None

    phase_t0 = time.perf_counter()
    res = minimize(
        strict_objective,
        theta0,
        method="SLSQP",
        constraints=constrained_attempt_constraints(OPD_tolerance),
        options={
            "maxiter": 500,
            "ftol": 1e-12,
            "disp": bool(verbose)
        }
    )
    profile_solve(
        f"SLSQP strict dt={time.perf_counter() - phase_t0:.3f}s "
        f"success={res.success}"
    )

    x_accepted = accept_result(res, OPD_tolerance, relaxed=False)
    if x_accepted is not None:
        profile_solve("SLSQP strict accepted")
        return x_accepted, res

    relaxed_res = None
    if qc_priority and relaxed_OPD_tolerance is not None and relaxed_OPD_tolerance > OPD_tolerance:
        phase_t0 = time.perf_counter()
        relaxed_res = minimize(
            qc_priority_objective,
            theta0,
            method="SLSQP",
            constraints=constrained_attempt_constraints(relaxed_OPD_tolerance),
            options={
                "maxiter": 500,
                "ftol": 1e-12,
                "disp": bool(verbose)
            }
        )
        profile_solve(
            f"SLSQP qc_priority dt={time.perf_counter() - phase_t0:.3f}s "
            f"success={relaxed_res.success}"
        )

        x_accepted = accept_result(relaxed_res, relaxed_OPD_tolerance, relaxed=True)
        if x_accepted is not None:
            profile_solve("SLSQP qc_priority accepted")
            return x_accepted, relaxed_res

    profile_solve("no final center solution accepted")
    best_res = relaxed_res if relaxed_res is not None else res
    failed_res = SimpleNamespace(
        x=theta0.copy(),
        success=False,
        message=(
            f"No final centered solution found within +/-{qc_tolerance} mm QC "
            f"and OPD tolerance strict +/-{OPD_tolerance} mm "
            f"(relaxed +/-{relaxed_OPD_tolerance} mm)."
        ),
        strict_result=res,
        relaxed_result=relaxed_res,
        best_result=best_res
    )
    return np.array(x_current, dtype=float).copy(), failed_res

def solve_centered_OPD_endpoint(x_current, target_OPD, M1, M2, M3, M4,
                                target_reflections,
                                M1_linear_loc, M2_linear_loc, M3_linear_loc,
                                qc_tolerance=0.5,
                                OPD_tolerance=0.05,
                                relaxed_OPD_tolerance=0.5,
                                qc_detector_limit=3.9,
                                u_min=0.1,
                                u_max=0.9,
                                include_edge_ends=False,
                                verbose=0,
                                profile_callback=None):
    """Find a centered final OPD endpoint using linear x positions and angles."""
    def profile_solve(message):
        if profile_callback is not None:
            profile_callback(message)

    x_current = np.array(x_current, dtype=float)
    current_mirrors = unpack_variables(x_current, M1, M2, M3, M4)
    lower, upper = linear_stage_x_bounds(
        *current_mirrors,
        M1_linear_loc, M2_linear_loc, M3_linear_loc
    )
    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)
    lower[6] = x_current[6]
    upper[6] = x_current[6]
    bounds = list(zip(lower, upper))
    expected_u_count = target_reflections if include_edge_ends else max(target_reflections - 2, 0)

    motion_scale = np.array([2.0, 0.1, 2.0, 0.1, 2.0, 0.1, 1.0, 0.1], dtype=float)

    def qc_values(x):
        qc1_error, qc2_error = quadcell_errors_from_variables(x, M1, M2, M3, M4)
        return np.array([qc1_error, qc2_error], dtype=float)

    def OPD_error(x):
        return OPD_from_variables(x, M1, M2, M3, M4) - target_OPD

    def constrained_us(x):
        mirrors_trial = unpack_variables(x, M1, M2, M3, M4)
        if get_reflection_count(*mirrors_trial) != target_reflections:
            return np.full(expected_u_count, -np.inf, dtype=float)

        us = reflection_us_from_variables(
            x, M1, M2, M3, M4,
            include_ends=include_edge_ends
        )
        if len(us) != expected_u_count:
            return np.full(expected_u_count, -np.inf, dtype=float)
        return us

    base_constraints = []
    for idx in range(expected_u_count):
        base_constraints.append({
            "type": "ineq",
            "fun": lambda x, i=idx: constrained_us(x)[i] - u_min
        })
        base_constraints.append({
            "type": "ineq",
            "fun": lambda x, i=idx: u_max - constrained_us(x)[i]
        })

    for idx in range(2):
        base_constraints.append({
            "type": "ineq",
            "fun": lambda x, i=idx: qc_detector_limit - qc_values(x)[i]
        })
        base_constraints.append({
            "type": "ineq",
            "fun": lambda x, i=idx: qc_detector_limit + qc_values(x)[i]
        })

    def endpoint_constraints(opd_tolerance):
        constraints = list(base_constraints)
        for idx in range(2):
            constraints.append({
                "type": "ineq",
                "fun": lambda x, i=idx: qc_tolerance - qc_values(x)[i]
            })
            constraints.append({
                "type": "ineq",
                "fun": lambda x, i=idx: qc_tolerance + qc_values(x)[i]
            })

        constraints.append({
            "type": "ineq",
            "fun": lambda x: opd_tolerance - OPD_error(x)
        })
        constraints.append({
            "type": "ineq",
            "fun": lambda x: opd_tolerance + OPD_error(x)
        })
        return constraints

    def strict_objective(x):
        motion_penalty = 1e-3 * np.sum(((np.array(x, dtype=float) - x_current) / motion_scale) ** 2)
        qc = qc_values(x)
        return float(OPD_error(x) ** 2 + 1e-3 * np.sum(qc ** 2) + motion_penalty)

    def qc_priority_objective(x):
        qc = qc_values(x)
        motion_penalty = 1e-3 * np.sum(((np.array(x, dtype=float) - x_current) / motion_scale) ** 2)
        return float(np.sum(qc ** 2) + 0.02 * OPD_error(x) ** 2 + motion_penalty)

    def accept_result(res, opd_tolerance, relaxed):
        if not res.success:
            return None

        x_endpoint = np.array(res.x, dtype=float)
        qc1_error, qc2_error = quadcell_errors_from_variables(x_endpoint, M1, M2, M3, M4)
        opd_error = OPD_error(x_endpoint)
        diagnostics = actuation_constraint_diagnostics(
            x_endpoint, M1, M2, M3, M4,
            max_qc_error=qc_detector_limit,
            expected_reflections=target_reflections,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=True,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=0.0
        )
        if (
            diagnostics["ok"] and
            abs(qc1_error) <= qc_tolerance and
            abs(qc2_error) <= qc_tolerance and
            abs(opd_error) <= opd_tolerance
        ):
            res.final_center_mode = "flexible_qc_priority_relaxed" if relaxed else "flexible_strict"
            res.final_center_OPD_relaxed = bool(relaxed)
            res.final_center_OPD_error = float(opd_error)
            res.final_center_qc_error = [float(qc1_error), float(qc2_error)]
            res.final_center_OPD_tolerance_used = float(opd_tolerance)
            res.flexible_endpoint = True
            return x_endpoint

        return None

    phase_t0 = time.perf_counter()
    res = minimize(
        strict_objective,
        x_current,
        method="SLSQP",
        bounds=bounds,
        constraints=endpoint_constraints(OPD_tolerance),
        options={
            "maxiter": 600,
            "ftol": 1e-12,
            "disp": bool(verbose)
        }
    )
    profile_solve(
        f"flexible SLSQP strict dt={time.perf_counter() - phase_t0:.3f}s "
        f"success={res.success}"
    )

    x_accepted = accept_result(res, OPD_tolerance, relaxed=False)
    if x_accepted is not None:
        profile_solve("flexible SLSQP strict accepted")
        return x_accepted, res

    relaxed_res = None
    if relaxed_OPD_tolerance is not None and relaxed_OPD_tolerance > OPD_tolerance:
        phase_t0 = time.perf_counter()
        relaxed_res = minimize(
            qc_priority_objective,
            x_current,
            method="SLSQP",
            bounds=bounds,
            constraints=endpoint_constraints(relaxed_OPD_tolerance),
            options={
                "maxiter": 600,
                "ftol": 1e-12,
                "disp": bool(verbose)
            }
        )
        profile_solve(
            f"flexible SLSQP qc_priority dt={time.perf_counter() - phase_t0:.3f}s "
            f"success={relaxed_res.success}"
        )

        x_accepted = accept_result(relaxed_res, relaxed_OPD_tolerance, relaxed=True)
        if x_accepted is not None:
            profile_solve("flexible SLSQP qc_priority accepted")
            return x_accepted, relaxed_res

    profile_solve("no flexible centered endpoint accepted")
    best_res = relaxed_res if relaxed_res is not None else res
    failed_res = SimpleNamespace(
        x=x_current.copy(),
        success=False,
        message=(
            f"No flexible centered endpoint found within +/-{qc_tolerance} mm QC, "
            f"final u=[{u_min}, {u_max}], OPD strict +/-{OPD_tolerance} mm "
            f"(relaxed +/-{relaxed_OPD_tolerance} mm)."
        ),
        strict_result=res,
        relaxed_result=relaxed_res,
        best_result=best_res,
        flexible_endpoint=True
    )
    return x_current.copy(), failed_res

def plan_OPD_linear_then_recenter(target_OPD, M1, M2, M3, M4,
                                  M1_linear_loc, M2_linear_loc, M3_linear_loc,
                                  qc_plan_limit=1.5,
                                  qc_detector_limit=3.9,
                                  qc_hardware_stop=3.5,
                                  max_qc_difference=None,
                                  preserve_reflection_count=True,
                                  motion_samples_per_step=25,
                                  u_min=0.1,
                                  u_max=0.9,
                                  sigma_edge=0.02,
                                  enforce_edge_bounds=True,
                                  include_edge_ends=False,
                                  constraint_tolerance=0.0,
                                  optimizer_verbose=0,
                                  max_iterations=80,
                                  min_dx=1e-4,
                                  target_OPD_tolerance=0.05,
                                  final_qc_tolerance=0.5,
                                  final_center_qc_threshold=0.5,
                                  final_OPD_relaxed_tolerance=0.5,
                                  final_center_qc_priority=True,
                                  correction_max_axis_splits=64,
                                  fast_recenter_path=True,
                                  fast_recenter_motion_samples_per_step=5,
                                  linear_u_min=0.05,
                                  linear_u_max=0.95,
                                  final_endpoint_waypoint_depth=4,
                                  linear_stage_order=("M1", "M2", "M3"),
                                  profile=False,
                                  profile_sink=None):
    if linear_stage_order is None:
        linear_stage_order = ("M1", "M2", "M3")
    linear_stage_order = tuple(linear_stage_order)
    if len(linear_stage_order) == 0:
        raise ValueError("linear_stage_order must include at least one stage.")
    for stage_name in linear_stage_order:
        linear_stage_x_axis(stage_name)

    if qc_plan_limit is None:
        qc_plan_limit = 1.5
    if qc_detector_limit is None:
        qc_detector_limit = max(3.9, qc_plan_limit)
    if qc_hardware_stop is None:
        qc_hardware_stop = 3.5
    qc_plan_limit = float(qc_plan_limit)
    qc_detector_limit = float(qc_detector_limit)
    qc_hardware_stop = float(qc_hardware_stop)

    profile_events = []
    profile_t0 = time.perf_counter()
    if profile and profile_sink is None:
        profile_sink = print

    def profile_log(message):
        if not profile:
            return
        elapsed = time.perf_counter() - profile_t0
        line = f"[choose_OPD {elapsed:.3f}s] {message}"
        profile_events.append({"elapsed": elapsed, "message": message})
        profile_sink(line)

    def profile_qc_edge_text(x):
        qc1_error, qc2_error = quadcell_errors_from_variables(x, M1, M2, M3, M4)
        edge_summary = reflection_edge_summary(
            x, M1, M2, M3, M4,
            include_ends=include_edge_ends
        )
        return (
            f"qc=({qc1_error:.3f},{qc2_error:.3f}) "
            f"u=[{edge_summary['min_u']:.3f},{edge_summary['max_u']:.3f}]"
        )

    def append_profiled_recenter_path(label, step_list, x_from, x_to):
        callback = lambda msg: profile_log(f"{label} {msg}")
        from_qc = quadcell_errors_from_variables(x_from, M1, M2, M3, M4)
        to_qc = quadcell_errors_from_variables(x_to, M1, M2, M3, M4)
        from_edges = reflection_edge_summary(x_from, M1, M2, M3, M4, include_ends=include_edge_ends)
        to_edges = reflection_edge_summary(x_to, M1, M2, M3, M4, include_ends=include_edge_ends)
        endpoint_max_qc = max(
            abs(from_qc[0]), abs(from_qc[1]),
            abs(to_qc[0]), abs(to_qc[1])
        )
        endpoint_min_u = min(from_edges["min_u"], to_edges["min_u"])
        endpoint_max_u = max(from_edges["max_u"], to_edges["max_u"])
        path_qc_limit = qc_plan_limit
        path_u_min = linear_u_min
        path_u_max = linear_u_max
        if endpoint_min_u < linear_u_min:
            path_u_min = max(0.0, endpoint_min_u - 1e-3)
        if endpoint_max_u > linear_u_max:
            path_u_max = min(1.0, endpoint_max_u + 1e-3)
        if path_qc_limit > qc_plan_limit or path_u_min < linear_u_min or path_u_max > linear_u_max:
            profile_log(
                f"{label} using recovery_qc_limit={path_qc_limit:.3f} "
                f"endpoint_max_qc={endpoint_max_qc:.3f} "
                f"u=[{path_u_min:.4f},{path_u_max:.4f}]"
            )
        if fast_recenter_path:
            return append_constrained_path_steps_fast_then_dense(
                step_list, x_from, x_to, M1, M2, M3, M4,
                max_axis_splits=correction_max_axis_splits,
                max_qc_error=path_qc_limit,
                max_qc_difference=max_qc_difference,
                preserve_reflection_count=preserve_reflection_count,
                motion_samples_per_step=motion_samples_per_step,
                fast_motion_samples_per_step=fast_recenter_motion_samples_per_step,
                u_min=path_u_min,
                u_max=path_u_max,
                enforce_edge_bounds=enforce_edge_bounds,
                include_edge_ends=include_edge_ends,
                constraint_tolerance=0.0,
                profile_callback=callback
            )

        return append_constrained_path_steps(
            step_list, x_from, x_to, M1, M2, M3, M4,
            max_axis_splits=correction_max_axis_splits,
            max_qc_error=path_qc_limit,
            max_qc_difference=max_qc_difference,
            preserve_reflection_count=preserve_reflection_count,
            motion_samples_per_step=motion_samples_per_step,
            u_min=path_u_min,
            u_max=path_u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=0.0,
            profile_callback=callback
        )

    x_start = pack_variables(M1, M2, M3, M4)
    x_current = x_start.copy()
    start_OPD = OPD_from_variables(x_current, M1, M2, M3, M4)
    stage_indices_by_direction = {1: 0, -1: 0}
    expected_reflections = get_reflection_count(M1, M2, M3, M4) if preserve_reflection_count else None
    start_diagnostics = actuation_constraint_diagnostics(
        x_current, M1, M2, M3, M4,
        max_qc_error=qc_plan_limit,
        max_qc_difference=max_qc_difference,
        expected_reflections=expected_reflections,
        u_min=linear_u_min,
        u_max=linear_u_max,
        enforce_edge_bounds=enforce_edge_bounds,
        include_edge_ends=include_edge_ends,
        constraint_tolerance=0.0
    )
    profile_log(
        f"start OPD={start_OPD:.3f} target={target_OPD:.3f} "
        f"{profile_qc_edge_text(x_current)} ok={start_diagnostics['ok']}"
    )

    steps = []
    final_res = None
    failure_reason = None
    final_center_failure_reason = None
    final_center_endpoint_reason = None

    if not start_diagnostics["ok"]:
        phase_t0 = time.perf_counter()
        x_recentered, final_res = solve_recenter_angles(
            x_current, M1, M2, M3, M4,
            target_reflections=expected_reflections or get_reflection_count(*unpack_variables(x_current, M1, M2, M3, M4)),
            max_qc_error=qc_detector_limit,
            u_min=u_min,
            u_max=u_max,
            sigma_edge=sigma_edge,
            include_edge_ends=include_edge_ends,
            verbose=optimizer_verbose,
            profile_callback=lambda msg: profile_log(f"initial recenter solve {msg}")
        )
        profile_log(
            f"initial recenter solve dt={time.perf_counter() - phase_t0:.3f}s "
            f"{profile_qc_edge_text(x_recentered)}"
        )
        phase_t0 = time.perf_counter()
        step_count_before = len(steps)
        x_current, correction_plan = append_profiled_recenter_path(
            "initial recenter path", steps, x_current, x_recentered
        )
        profile_log(
            f"initial recenter path dt={time.perf_counter() - phase_t0:.3f}s "
            f"steps_added={len(steps) - step_count_before} "
            f"failure={correction_plan['failure_reason']}"
        )
        if correction_plan["failure_reason"] is not None:
            failure_reason = "Initial recenter path failed: " + correction_plan["failure_reason"]

    for iteration_index in range(1, max_iterations + 1):
        if failure_reason is not None:
            break

        current_OPD = OPD_from_variables(x_current, M1, M2, M3, M4)
        profile_log(
            f"iteration={iteration_index} OPD={current_OPD:.3f} "
            f"target={target_OPD:.3f} steps={len(steps)}"
        )
        if abs(current_OPD - target_OPD) <= target_OPD_tolerance:
            break

        target_direction = 1 if target_OPD >= current_OPD else -1
        stage_order = linear_stage_order if target_direction > 0 else tuple(reversed(linear_stage_order))
        stage_index = stage_indices_by_direction[target_direction]

        if stage_index >= len(stage_order):
            failure_reason = "Linear stages reached their travel limits before the target OPD."
            break

        stage_name = stage_order[stage_index]
        axis_index = linear_stage_x_axis(stage_name)
        dx_limit = linear_stage_available_dx(
            stage_name, target_direction,
            M1_linear_loc, M2_linear_loc, M3_linear_loc
        )

        if abs(dx_limit) <= min_dx:
            profile_log(f"stage={stage_name} skipped dx_limit={dx_limit:.6g}")
            stage_indices_by_direction[target_direction] += 1
            continue

        phase_t0 = time.perf_counter()
        fraction_to_constraint = find_max_valid_linear_fraction(
            x_current, axis_index, dx_limit, M1, M2, M3, M4,
            max_qc_error=qc_plan_limit,
            max_qc_difference=max_qc_difference,
            expected_reflections=expected_reflections,
            u_min=linear_u_min,
            u_max=linear_u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=0.0,
            scan_samples=max(motion_samples_per_step * 4, 40)
        )
        profile_log(
            f"linear search stage={stage_name} "
            f"valid_fraction={fraction_to_constraint:.6g} "
            f"dt={time.perf_counter() - phase_t0:.3f}s"
        )
        fraction_to_target = None
        phase_t0 = time.perf_counter()
        if fraction_to_constraint > 1e-12:
            fraction_to_target_in_valid_range = find_linear_fraction_to_target(
                x_current, axis_index, dx_limit * fraction_to_constraint,
                target_OPD, M1, M2, M3, M4
            )
            if fraction_to_target_in_valid_range is not None:
                fraction_to_target = fraction_to_constraint * fraction_to_target_in_valid_range

        reached_target = fraction_to_target is not None and fraction_to_target <= fraction_to_constraint + 1e-9
        move_fraction = fraction_to_target if reached_target else fraction_to_constraint
        profile_log(
            f"target search stage={stage_name} reached_target={reached_target} "
            f"move_fraction={move_fraction:.6g} "
            f"dt={time.perf_counter() - phase_t0:.3f}s"
        )

        if move_fraction <= 1e-8:
            phase_t0 = time.perf_counter()
            x_recentered, final_res = solve_recenter_angles(
                x_current, M1, M2, M3, M4,
                target_reflections=expected_reflections or get_reflection_count(*unpack_variables(x_current, M1, M2, M3, M4)),
                max_qc_error=qc_detector_limit,
                u_min=u_min,
                u_max=u_max,
                sigma_edge=sigma_edge,
                include_edge_ends=include_edge_ends,
                verbose=optimizer_verbose,
                profile_callback=lambda msg: profile_log(f"zero-move recenter solve {msg}")
            )
            profile_log(
                f"zero-move recenter solve dt={time.perf_counter() - phase_t0:.3f}s "
                f"{profile_qc_edge_text(x_recentered)}"
            )
            if np.allclose(x_recentered, x_current, atol=1e-8, rtol=0):
                profile_log(f"zero-move recenter unchanged; advancing past stage={stage_name}")
                stage_indices_by_direction[target_direction] += 1
                continue
            phase_t0 = time.perf_counter()
            step_count_before = len(steps)
            x_current, correction_plan = append_profiled_recenter_path(
                "zero-move recenter path", steps, x_current, x_recentered
            )
            profile_log(
                f"zero-move recenter path dt={time.perf_counter() - phase_t0:.3f}s "
                f"steps_added={len(steps) - step_count_before} "
                f"failure={correction_plan['failure_reason']}"
            )
            if correction_plan["failure_reason"] is not None:
                failure_reason = "Recenter path failed: " + correction_plan["failure_reason"]
                break
            continue

        x_next = variables_with_axis_move(x_current, axis_index, dx_limit * move_fraction)
        x_previous = x_current.copy()
        step_start_index = len(steps)
        phase_t0 = time.perf_counter()
        x_current = append_axis_steps(
            steps, x_current, x_next, M1, M2, M3, M4,
            max_qc_error=qc_plan_limit,
            max_qc_difference=max_qc_difference,
            motion_samples_per_step=motion_samples_per_step,
            u_min=linear_u_min,
            u_max=linear_u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=0.0
        )
        profile_log(
            f"linear move stage={stage_name} reached_target={reached_target} "
            f"hit_constraint={bool(not reached_target and move_fraction < 1.0 - 1e-8)} "
            f"steps_added={len(steps) - step_start_index} "
            f"OPD={OPD_from_variables(x_current, M1, M2, M3, M4):.3f} "
            f"{profile_qc_edge_text(x_current)} "
            f"dt={time.perf_counter() - phase_t0:.3f}s"
        )
        for step in steps[step_start_index:]:
            step["linear_OPD_move"] = True
            step["linear_stage"] = stage_name
            step["linear_move_fraction"] = move_fraction
            step["linear_move_reached_target"] = bool(reached_target)
            step["linear_move_hit_constraint"] = bool(not reached_target and move_fraction < 1.0 - 1e-8)

        previous_mirrors = unpack_variables(x_previous, M1, M2, M3, M4)
        current_mirrors = unpack_variables(x_current, M1, M2, M3, M4)
        M1_linear_loc, M2_linear_loc, M3_linear_loc = update_linear_stage_locs(
            previous_mirrors,
            current_mirrors,
            M1_linear_loc, M2_linear_loc, M3_linear_loc
        )

        if reached_target:
            break

        if move_fraction >= 1.0 - 1e-8:
            stage_indices_by_direction[target_direction] += 1
            continue

        phase_t0 = time.perf_counter()
        x_recentered, final_res = solve_recenter_angles(
            x_current, M1, M2, M3, M4,
            target_reflections=expected_reflections or get_reflection_count(*unpack_variables(x_current, M1, M2, M3, M4)),
            max_qc_error=qc_detector_limit,
            u_min=u_min,
            u_max=u_max,
            sigma_edge=sigma_edge,
            include_edge_ends=include_edge_ends,
            verbose=optimizer_verbose,
            profile_callback=lambda msg: profile_log(f"recenter solve {msg}")
        )
        profile_log(
            f"recenter solve dt={time.perf_counter() - phase_t0:.3f}s "
            f"{profile_qc_edge_text(x_recentered)}"
        )
        phase_t0 = time.perf_counter()
        step_count_before = len(steps)
        x_current, correction_plan = append_profiled_recenter_path(
            "recenter path", steps, x_current, x_recentered
        )
        profile_log(
            f"recenter path dt={time.perf_counter() - phase_t0:.3f}s "
            f"steps_added={len(steps) - step_count_before} "
            f"failure={correction_plan['failure_reason']}"
        )
        if correction_plan["failure_reason"] is not None:
            failure_reason = "Recenter path failed: " + correction_plan["failure_reason"]
            break
    else:
        failure_reason = "Reached max_iterations while planning OPD actuation."

    if failure_reason is None:
        pre_final_qc1_error, pre_final_qc2_error = quadcell_errors_from_variables(
            x_current, M1, M2, M3, M4
        )
        pre_final_OPD_error = OPD_from_variables(x_current, M1, M2, M3, M4) - target_OPD
        pre_final_diagnostics = actuation_constraint_diagnostics(
            x_current, M1, M2, M3, M4,
            max_qc_error=qc_plan_limit,
            max_qc_difference=max_qc_difference,
            expected_reflections=expected_reflections,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=0.0
        )
        pre_final_qc_ok = (
            final_center_qc_threshold is not None and
            max(abs(pre_final_qc1_error), abs(pre_final_qc2_error)) <= final_center_qc_threshold
        )
        pre_final_OPD_ok = abs(pre_final_OPD_error) <= target_OPD_tolerance
        skip_final_center = (
            pre_final_qc_ok and
            pre_final_OPD_ok and
            pre_final_diagnostics["ok"]
        )
        if skip_final_center:
            profile_log(
                f"final center skipped qc=({pre_final_qc1_error:.3f},{pre_final_qc2_error:.3f}) "
                f"threshold={final_center_qc_threshold} "
                f"final_constraints_ok=True"
            )
        else:
            if pre_final_qc_ok:
                reason_parts = []
                if not pre_final_OPD_ok:
                    reason_parts.append(f"OPD_error={pre_final_OPD_error:.6g}")
                if not pre_final_diagnostics["ok"]:
                    reason_parts.append("; ".join(pre_final_diagnostics["failures"]))
                profile_log(
                    "final center not skipped despite centered QC: " +
                    " | ".join(reason_parts)
                )
            final_center_succeeded = False
            skip_angle_only_final_center = pre_final_qc_ok and pre_final_OPD_ok and not pre_final_diagnostics["ok"]
            if skip_angle_only_final_center:
                final_center_failure_reason = "Skipped angle-only final centering because QC/OPD are already acceptable but final constraints are not."
                profile_log("final center angle-only skipped; trying flexible endpoint")
            else:
                phase_t0 = time.perf_counter()
                x_centered, final_center_res = solve_final_centered_angles(
                    x_current,
                    target_OPD,
                    M1, M2, M3, M4,
                    target_reflections=expected_reflections or get_reflection_count(*unpack_variables(x_current, M1, M2, M3, M4)),
                    qc_tolerance=final_qc_tolerance,
                    OPD_tolerance=target_OPD_tolerance,
                    relaxed_OPD_tolerance=final_OPD_relaxed_tolerance,
                    qc_detector_limit=qc_detector_limit,
                    qc_priority=final_center_qc_priority,
                    u_min=u_min,
                    u_max=u_max,
                    include_edge_ends=include_edge_ends,
                    verbose=optimizer_verbose,
                    profile_callback=lambda msg: profile_log(f"final center solve {msg}")
                )
                profile_log(
                    f"final center solve dt={time.perf_counter() - phase_t0:.3f}s "
                    f"changed={not np.allclose(x_centered, x_current, atol=1e-10, rtol=0)} "
                    f"success={getattr(final_center_res, 'success', None)}"
                )
                if not np.allclose(x_centered, x_current, atol=1e-10, rtol=0):
                    phase_t0 = time.perf_counter()
                    step_count_before = len(steps)
                    x_centered_path, correction_plan = append_profiled_recenter_path(
                        "final center path", steps, x_current, x_centered
                    )
                    profile_log(
                        f"final center path dt={time.perf_counter() - phase_t0:.3f}s "
                        f"steps_added={len(steps) - step_count_before} "
                        f"failure={correction_plan['failure_reason']}"
                    )
                    if correction_plan["failure_reason"] is None:
                        x_current = x_centered_path
                        final_res = final_center_res
                        final_center_succeeded = True
                    else:
                        final_center_failure_reason = "Final center path failed: " + correction_plan["failure_reason"]
                elif getattr(final_center_res, "success", False):
                    final_res = final_center_res
                    final_center_succeeded = True
                elif getattr(final_center_res, "success", False) is False:
                    final_center_failure_reason = final_center_res.message

            if not final_center_succeeded:
                phase_t0 = time.perf_counter()
                x_endpoint, endpoint_res = solve_centered_OPD_endpoint(
                    x_current,
                    target_OPD,
                    M1, M2, M3, M4,
                    target_reflections=expected_reflections or get_reflection_count(*unpack_variables(x_current, M1, M2, M3, M4)),
                    M1_linear_loc=M1_linear_loc,
                    M2_linear_loc=M2_linear_loc,
                    M3_linear_loc=M3_linear_loc,
                    qc_tolerance=final_qc_tolerance,
                    OPD_tolerance=target_OPD_tolerance,
                    relaxed_OPD_tolerance=final_OPD_relaxed_tolerance,
                    qc_detector_limit=qc_detector_limit,
                    u_min=u_min,
                    u_max=u_max,
                    include_edge_ends=include_edge_ends,
                    verbose=optimizer_verbose,
                    profile_callback=lambda msg: profile_log(f"final endpoint solve {msg}")
                )
                endpoint_qc = quadcell_errors_from_variables(x_endpoint, M1, M2, M3, M4)
                endpoint_edges = reflection_edge_summary(
                    x_endpoint, M1, M2, M3, M4,
                    include_ends=include_edge_ends
                )
                profile_log(
                    f"final endpoint solve dt={time.perf_counter() - phase_t0:.3f}s "
                    f"changed={not np.allclose(x_endpoint, x_current, atol=1e-10, rtol=0)} "
                    f"success={getattr(endpoint_res, 'success', None)} "
                    f"OPD={OPD_from_variables(x_endpoint, M1, M2, M3, M4):.3f} "
                    f"qc=({endpoint_qc[0]:.3f},{endpoint_qc[1]:.3f}) "
                    f"u=[{endpoint_edges['min_u']:.3f},{endpoint_edges['max_u']:.3f}]"
                )

                if getattr(endpoint_res, "success", False):
                    phase_t0 = time.perf_counter()
                    step_count_before = len(steps)
                    x_before_endpoint_path = x_current.copy()
                    x_endpoint_path, endpoint_plan = append_waypoint_constrained_path_steps(
                        steps, x_current, x_endpoint, M1, M2, M3, M4,
                        max_axis_splits=correction_max_axis_splits,
                        max_waypoint_depth=final_endpoint_waypoint_depth,
                        max_qc_error=qc_plan_limit,
                        max_qc_difference=max_qc_difference,
                        preserve_reflection_count=preserve_reflection_count,
                        motion_samples_per_step=motion_samples_per_step,
                        fast_motion_samples_per_step=fast_recenter_motion_samples_per_step,
                        u_min=linear_u_min,
                        u_max=linear_u_max,
                        enforce_edge_bounds=enforce_edge_bounds,
                        include_edge_ends=include_edge_ends,
                        constraint_tolerance=0.0,
                        profile_callback=lambda msg: profile_log(f"final endpoint path {msg}")
                    )
                    profile_log(
                        f"final endpoint path dt={time.perf_counter() - phase_t0:.3f}s "
                        f"steps_added={len(steps) - step_count_before} "
                        f"failure={endpoint_plan['failure_reason']}"
                    )
                    if endpoint_plan["failure_reason"] is None:
                        previous_mirrors = unpack_variables(x_before_endpoint_path, M1, M2, M3, M4)
                        current_mirrors = unpack_variables(x_endpoint_path, M1, M2, M3, M4)
                        M1_linear_loc, M2_linear_loc, M3_linear_loc = update_linear_stage_locs(
                            previous_mirrors,
                            current_mirrors,
                            M1_linear_loc, M2_linear_loc, M3_linear_loc
                        )
                        x_current = x_endpoint_path
                        final_res = endpoint_res
                        final_center_failure_reason = None
                    else:
                        final_center_endpoint_reason = (
                            "Centered endpoint found but no constrained path found: " +
                            endpoint_plan["failure_reason"]
                        )
                else:
                    final_center_endpoint_reason = endpoint_res.message

    if failure_reason is None:
        phase_t0 = time.perf_counter()
        final_OPD_error = OPD_from_variables(x_current, M1, M2, M3, M4) - target_OPD
        final_qc1_error, final_qc2_error = quadcell_errors_from_variables(x_current, M1, M2, M3, M4)
        final_OPD_tolerance_used = target_OPD_tolerance
        if getattr(final_res, "final_center_OPD_relaxed", False):
            final_OPD_tolerance_used = final_OPD_relaxed_tolerance
        final_diagnostics = actuation_constraint_diagnostics(
            x_current, M1, M2, M3, M4,
            max_qc_error=qc_plan_limit,
            max_qc_difference=max_qc_difference,
            expected_reflections=expected_reflections,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=0.0
        )
        if max(abs(final_qc1_error), abs(final_qc2_error)) > final_qc_tolerance:
            failure_reason = (
                f"Final QC offset ({final_qc1_error:.4g}, {final_qc2_error:.4g}) "
                f"exceeds final tolerance {final_qc_tolerance}."
            )
            if final_center_endpoint_reason is not None:
                failure_reason += " " + final_center_endpoint_reason
        elif abs(final_OPD_error) > final_OPD_tolerance_used:
            failure_reason = (
                f"Final OPD error {final_OPD_error:.4g} exceeds tolerance "
                f"{final_OPD_tolerance_used}."
            )
            if final_center_endpoint_reason is not None:
                failure_reason += " " + final_center_endpoint_reason
        elif not final_diagnostics["ok"]:
            failure_reason = "Final state is outside constraints: " + "; ".join(final_diagnostics["failures"])
            if final_center_endpoint_reason is not None:
                failure_reason += " " + final_center_endpoint_reason
        profile_log(
            f"final validation dt={time.perf_counter() - phase_t0:.3f}s "
            f"final_error={final_OPD_error:.6g} ok={failure_reason is None}"
        )

    M1_opt, M2_opt, M3_opt, M4_opt = unpack_variables(x_current, M1, M2, M3, M4)
    if final_res is None:
        final_res = SimpleNamespace()
    final_res = set_OPD_result_full_x(final_res, M1_opt, M2_opt, M3_opt, M4_opt)
    target_reflections = get_reflection_count(M1_opt, M2_opt, M3_opt, M4_opt)
    plan = build_actuation_plan_summary(
        steps, x_start, x_current, M1, M2, M3, M4,
        get_reflection_count(M1, M2, M3, M4),
        target_reflections,
        start_diagnostics["ok"],
        expected_reflections,
        max_qc_error=qc_plan_limit,
        max_qc_difference=max_qc_difference,
        motion_samples_per_step=motion_samples_per_step,
        u_min=linear_u_min,
        u_max=linear_u_max,
        include_edge_ends=include_edge_ends,
        search_mode="linear_then_recenter",
        split_count=None,
        failure_reason=failure_reason
    )
    plan["target_OPD"] = target_OPD
    plan["start_OPD"] = start_OPD
    plan["final_OPD"] = OPD_from_variables(x_current, M1, M2, M3, M4)
    plan["final_linear_stage_locs"] = {
        "M1": M1_linear_loc,
        "M2": M2_linear_loc,
        "M3": M3_linear_loc
    }
    plan["final_OPD_error"] = plan["final_OPD"] - target_OPD
    plan["final_qc1_error"], plan["final_qc2_error"] = quadcell_errors_from_variables(
        x_current, M1, M2, M3, M4
    )
    plan["final_qc_tolerance"] = final_qc_tolerance
    plan["final_center_qc_threshold"] = final_center_qc_threshold
    plan["qc_detector_limit"] = qc_detector_limit
    plan["qc_plan_limit"] = qc_plan_limit
    plan["qc_hardware_stop"] = qc_hardware_stop
    plan["final_OPD_relaxed_tolerance"] = final_OPD_relaxed_tolerance
    plan["final_center_qc_priority"] = bool(final_center_qc_priority)
    plan["final_center_OPD_relaxed"] = bool(getattr(final_res, "final_center_OPD_relaxed", False))
    plan["final_center_OPD_tolerance_used"] = float(getattr(
        final_res,
        "final_center_OPD_tolerance_used",
        target_OPD_tolerance
    ))
    plan["final_center_failure_reason"] = final_center_failure_reason
    plan["final_center_endpoint_reason"] = final_center_endpoint_reason
    plan["fast_recenter_path"] = fast_recenter_path
    plan["fast_recenter_motion_samples_per_step"] = fast_recenter_motion_samples_per_step
    plan["linear_u_min"] = linear_u_min
    plan["linear_u_max"] = linear_u_max
    plan["final_u_min"] = u_min
    plan["final_u_max"] = u_max
    plan["final_endpoint_waypoint_depth"] = final_endpoint_waypoint_depth
    plan["recenter_u_min"] = u_min
    plan["recenter_u_max"] = u_max
    if profile:
        plan["profile"] = profile_events

    linear_step_count = sum(1 for step in steps if step.get("linear_OPD_move"))
    profile_log(
        f"done steps={len(steps)} linear_steps={linear_step_count} "
        f"final_error={plan['final_OPD_error']:.6g} failure={failure_reason}"
    )

    return (M1_opt, M2_opt, M3_opt, M4_opt), final_res, plan

# OPTIMIZING

def qc_displacement_residuals(theta, qc1_disp=None, qc2_disp=None):
    if qc1_disp is None and qc2_disp is None:
        return np.array([], dtype=float)

    M1x, M2x, M3x, M4x, M1y, M2y, M3y, M4y, M1a, M2a, M3a, M4a = theta
    g = np.array(_simulation_metrics(
        M1x, M1y, M2x, M2y, M3x, M3y, M4x, M4y,
        M1a, M2a, M3a, M4a
    ), dtype=float)

    residuals_qc = []

    # QC readout sign is opposite the simulation y-error convention:
    # qc*_disp = +3 means the beam is -3 mm in simulation y relative to center.
    if qc1_disp is not None:
        residuals_qc.append((g[2] + qc1_disp) / SIGMA_QC)

    if qc2_disp is not None:
        residuals_qc.append((g[4] + qc2_disp) / SIGMA_QC)

    return np.array(residuals_qc, dtype=float)

def mirrors_from_inverse_theta(theta):
    M1x, M2x, M3x, M4x, M1y, M2y, M3y, M4y, M1a, M2a, M3a, M4a = np.array(theta, dtype=float)
    return (
        np.array([M1x, M1y, M1a], dtype=float),
        np.array([M2x, M2y, M2a], dtype=float),
        np.array([M3x, M3y, M3a], dtype=float),
        np.array([M4x, M4y, M4a], dtype=float),
    )

def reflection_count_residuals(theta, target_reflections=None, penalty_scale=None):
    if target_reflections is None:
        return np.array([], dtype=float)

    if penalty_scale is None:
        penalty_scale = DEFAULT_PEN * 10 / SIGMA_REFL

    mirrors = mirrors_from_inverse_theta(theta)
    n_reflections = get_reflection_count(*mirrors)
    if n_reflections == int(target_reflections):
        return np.array([0.0], dtype=float)

    return np.array([
        float(penalty_scale) * (n_reflections - int(target_reflections))
    ], dtype=float)

def optimize_inverse(M1, M2, M3, M4, img_path_light, img_path_dark=None,
                     qc1_disp=None, qc2_disp=None,
                     target_reflections=None,
                     N_R=None,
                     preserve_reflection_count=True,
                     reflection_count_penalty=None,
                     verbose=2):

    theta0 = np.array(
        [M1[0], M2[0], M3[0], M4[0],
         M1[1], M2[1], M3[1], M4[1],
         M1[2], M2[2], M3[2], M4[2]],
        dtype=float
    )

    initial_reflections = get_reflection_count(M1, M2, M3, M4)
    if N_R is not None:
        if target_reflections is not None and int(target_reflections) != int(N_R):
            raise ValueError("target_reflections and N_R were both provided with different values.")
        target_reflections = int(N_R)

    if target_reflections is None and preserve_reflection_count:
        target_reflections = int(initial_reflections)

    detected_reflection_count = None
    if img_path_dark is None:
        base_residual_fun = lambda th: aruco_pixel_residuals(th, img_path_light) / SIGMA_PX
    else:
        img_dark = cv.imread(img_path_dark)
        if img_dark is None:
            raise ValueError(f"Could not read dark image: {img_path_dark}")

        img_gray = cv.cvtColor(img_dark, cv.COLOR_BGR2GRAY)

        reflec_cam = reflec_pts_cam(img_gray, show=False)
        detected_total = sum(len(v) for v in reflec_cam.values())
        detected_reflection_count = int(detected_total)
        expected_total = int(target_reflections) if target_reflections is not None else detected_total

        base_residual_fun = lambda th: residuals(
            th,
            img_path_light=img_path_light,
            reflec_cam=reflec_cam,
            expected_total=expected_total
        )

    residual_fun = lambda th: np.concatenate([
        base_residual_fun(th),
        qc_displacement_residuals(th, qc1_disp=qc1_disp, qc2_disp=qc2_disp),
        reflection_count_residuals(
            th,
            target_reflections=target_reflections,
            penalty_scale=reflection_count_penalty
        )
    ])

    res = least_squares(
        fun=residual_fun,
        x0=theta0,
        loss="linear",
        f_scale=1.0,
        verbose=verbose,  # IMPORTANT for profiling
        x_scale = np.array([20,20,20,20,  20,20,20,20,  0.5,0.5,0.5,0.5], dtype=float),
        max_nfev=4000,
        ftol=1e-10, 
        xtol=1e-10, 
        gtol=1e-10
    )

    final_mirrors = mirrors_from_inverse_theta(res.x)
    final_reflections = get_reflection_count(*final_mirrors)
    res.initial_reflection_count = int(initial_reflections)
    res.target_reflection_count = None if target_reflections is None else int(target_reflections)
    res.detected_reflection_count = detected_reflection_count
    res.final_reflection_count = int(final_reflections)
    res.reflection_count_ok = (
        True if target_reflections is None
        else int(final_reflections) == int(target_reflections)
    )
    if target_reflections is not None and final_reflections != int(target_reflections):
        res.message = (
            str(res.message) +
            f" Final reflection count {final_reflections} != target {target_reflections}."
        )

    return res

def solve_center_once(theta0, M1, M2, M3, M4, target_reflections,
                      u_min=0.1, u_max=0.9, sigma_edge=0.1):
    res = least_squares(
        fun=lambda th: center_quadcells_residuals(
            th, M1, M2, M3, M4,
            target_reflections=target_reflections,
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
                     target_reflections=None,
                     N_R=None,
                     n_tries=20,
                     angle_perturb=0.2,
                     seed=0,
                     u_min=0.1,
                     u_max=0.9,
                     sigma_edge=0.1,
                     final_qc_tolerance=0.25):
    """Find a rotation-only solution with the requested reflection count.

    When final_qc_tolerance is set, endpoints must have both quadcell offsets
    inside +/- final_qc_tolerance. Among those endpoints, choose the one with
    the smallest total absolute angle change from the initial configuration.
    Set final_qc_tolerance=None to recover the legacy "smallest QC norm" choice.
    """

    theta_init = np.array([M1[2], M2[2], M3[2], M4[2]], dtype=float)

    mirrors0 = build_mirrors(M1, M2, M3, M4)
    reflection_data0 = trace_reflections(laser_start, laser_angle, mirrors0)
    initial_reflections = len(reflection_data0)
    if N_R is not None:
        if target_reflections is not None and target_reflections != N_R:
            raise ValueError("target_reflections and N_R were both provided with different values.")
        target_reflections = N_R

    if target_reflections is None:
        target_reflections = initial_reflections
    target_reflections = int(target_reflections)

    if final_qc_tolerance is not None:
        final_qc_tolerance = float(final_qc_tolerance)
        if final_qc_tolerance < 0:
            raise ValueError("final_qc_tolerance must be non-negative or None.")

    rng = np.random.default_rng(seed)

    starts = [theta_init]
    for _ in range(n_tries - 1):
        starts.append(theta_init + rng.uniform(-angle_perturb, angle_perturb, size=4))

    best_res = None
    best_score = (np.inf, np.inf)
    best_angles = None
    matching_start_count = 0
    valid_solution_count = 0
    centered_solution_count = 0

    for th0 in starts:
        M1_start = np.array([M1[0], M1[1], th0[0]], dtype=float)
        M2_start = np.array([M2[0], M2[1], th0[1]], dtype=float)
        M3_start = np.array([M3[0], M3[1], th0[2]], dtype=float)
        M4_start = np.array([M4[0], M4[1], th0[3]], dtype=float)
        start_reflections = get_reflection_count(M1_start, M2_start, M3_start, M4_start)

        if start_reflections != target_reflections:
            continue

        matching_start_count += 1

        res = solve_center_once(
            th0, M1, M2, M3, M4,
            target_reflections=target_reflections,
            u_min=u_min, u_max=u_max,
            sigma_edge=sigma_edge
        )

        M1_new = np.array([M1[0], M1[1], res.x[0]], dtype=float)
        M2_new = np.array([M2[0], M2[1], res.x[1]], dtype=float)
        M3_new = np.array([M3[0], M3[1], res.x[2]], dtype=float)
        M4_new = np.array([M4[0], M4[1], res.x[3]], dtype=float)

        mirrors_new = build_mirrors(M1_new, M2_new, M3_new, M4_new)
        reflection_data_new = trace_reflections(laser_start, laser_angle, mirrors_new)

        if len(reflection_data_new) != target_reflections:
            continue
        valid_solution_count += 1

        g_final = simulation_identifier(
            M1_new[0], M1_new[1],
            M2_new[0], M2_new[1],
            M3_new[0], M3_new[1],
            M4_new[0], M4_new[1],
            M1_new[2], M2_new[2], M3_new[2], M4_new[2]
        )
        g_final = np.array(g_final, dtype=float)

        qc = np.array([g_final[2], g_final[4]], dtype=float)
        qc_norm = float(np.linalg.norm(qc))
        qc_max_abs = float(np.max(np.abs(qc)))
        angle_change = float(np.sum(np.abs(np.array(res.x, dtype=float) - theta_init)))

        if final_qc_tolerance is None:
            score = (qc_norm, angle_change)
        else:
            if qc_max_abs > final_qc_tolerance:
                continue
            centered_solution_count += 1
            score = (angle_change, qc_norm)

        if score < best_score:
            best_score = score
            best_res = res
            best_angles = res.x.copy()
            best_res.final_qc_errors = qc.copy()
            best_res.final_qc_norm = qc_norm
            best_res.final_qc_max_abs = qc_max_abs
            best_res.final_angle_change_total_abs = angle_change

    if best_res is None:
        if matching_start_count == 0:
            raise RuntimeError(
                f"No starting angle set with N_R={target_reflections} was found. "
                "Try increasing n_tries and/or angle_perturb."
            )
        if valid_solution_count > 0 and final_qc_tolerance is not None:
            raise RuntimeError(
                f"No valid N_R={target_reflections} solution reached both quadcells "
                f"within +/-{final_qc_tolerance}. "
                f"Valid N_R solutions checked: {valid_solution_count}."
            )
        raise RuntimeError(f"No valid centered solution found with N_R={target_reflections}.")

    best_res.selection_mode = (
        "min_qc_norm" if final_qc_tolerance is None else "min_angle_change_within_qc_tolerance"
    )
    best_res.final_qc_tolerance = final_qc_tolerance
    best_res.matching_start_count = int(matching_start_count)
    best_res.valid_solution_count = int(valid_solution_count)
    best_res.centered_solution_count = int(centered_solution_count)

    M1_opt = np.array([M1[0], M1[1], best_angles[0]], dtype=float)
    M2_opt = np.array([M2[0], M2[1], best_angles[1]], dtype=float)
    M3_opt = np.array([M3[0], M3[1], best_angles[2]], dtype=float)
    M4_opt = np.array([M4[0], M4[1], best_angles[3]], dtype=float)

    return (M1_opt, M2_opt, M3_opt, M4_opt), best_res


def _active_angle_axes(active_actuators=None):
    angle_labels = ["M1.dangle", "M2.dangle", "M3.dangle", "M4.dangle"]
    angle_axes = np.array([1, 3, 5, 7], dtype=int)
    if active_actuators is None:
        return angle_labels, angle_axes
    if isinstance(active_actuators, str):
        active_actuators = [active_actuators]
    active_labels = list(active_actuators)
    active_indices = []
    for label in active_labels:
        if label not in angle_labels:
            raise ValueError(f"active actuator must be one of {angle_labels}, got {label!r}.")
        active_indices.append(angle_labels.index(label))
    if len(set(active_indices)) != len(active_indices):
        raise ValueError("active_actuators contains duplicates.")
    return [angle_labels[i] for i in active_indices], angle_axes[np.array(active_indices, dtype=int)]


def quadcell_angle_jacobian(M1, M2, M3, M4, angles=None, step_deg=1e-4, active_actuators=None):
    """Finite-difference Jacobian d(QC1, QC2) / d(M*.dangle)."""
    M_start = (
        np.array(M1, dtype=float),
        np.array(M2, dtype=float),
        np.array(M3, dtype=float),
        np.array(M4, dtype=float),
    )
    angle_labels, angle_axes = _active_angle_axes(active_actuators)
    n_angles = len(angle_axes)
    x_base = pack_variables(*M_start)
    if angles is None:
        angles = x_base[angle_axes]
    else:
        angles = np.array(angles, dtype=float)
        if angles.shape != (n_angles,):
            raise ValueError(
                f"angles must contain {n_angles} values for active_actuators={angle_labels}."
            )
    step_deg = float(abs(step_deg))
    if step_deg <= 0:
        raise ValueError("step_deg must be positive.")

    def qc_from_angles(theta):
        x = x_base.copy()
        x[angle_axes] = np.array(theta, dtype=float)
        return np.array(quadcell_errors_from_variables(x, *M_start), dtype=float)

    jac = np.zeros((2, n_angles), dtype=float)
    reflection_counts_minus = []
    reflection_counts_plus = []
    for idx in range(n_angles):
        delta = np.zeros(n_angles, dtype=float)
        delta[idx] = step_deg
        jac[:, idx] = (qc_from_angles(angles + delta) - qc_from_angles(angles - delta)) / (2.0 * step_deg)

        x_minus = x_base.copy()
        x_minus[angle_axes] = angles - delta
        x_plus = x_base.copy()
        x_plus[angle_axes] = angles + delta
        reflection_counts_minus.append(int(get_reflection_count(*unpack_variables(x_minus, *M_start))))
        reflection_counts_plus.append(int(get_reflection_count(*unpack_variables(x_plus, *M_start))))

    return {
        "jacobian": jac,
        "angles": angles,
        "angle_labels": angle_labels,
        "angle_axes": angle_axes,
        "step_deg": float(step_deg),
        "reflection_counts_minus": reflection_counts_minus,
        "reflection_counts_plus": reflection_counts_plus,
    }


def trace_centered_quadcell_angle_curve(M1, M2, M3, M4,
                                        target_reflections=None,
                                        start_angles=None,
                                        active_actuators=None,
                                        preferred_axis="M1.dangle",
                                        n_steps=80,
                                        step_deg=0.005,
                                        qc_tolerance=0.05,
                                        u_min=0.1,
                                        u_max=0.9,
                                        auto_center_start=True,
                                        jacobian_step_deg=1e-4,
                                        qc_scale=0.01,
                                        sigma_edge=0.02,
                                        correction_regularization=0.08,
                                        max_corrector_nfev=120,
                                        include_edge_ends=False):
    """Trace a 1D centered-QC curve in the selected mirror-angle space.

    QC centering gives two equations. With three active angle variables, the
    centered set is generally a true 1D curve. With four active variables, this
    traces one useful 1D curve on the larger centered surface.
    """
    M_start = (
        np.array(M1, dtype=float),
        np.array(M2, dtype=float),
        np.array(M3, dtype=float),
        np.array(M4, dtype=float),
    )
    angle_labels, angle_axes = _active_angle_axes(active_actuators)
    if len(angle_axes) < 3:
        raise ValueError("trace_centered_quadcell_angle_curve needs at least three active dangle actuators.")
    x_base = pack_variables(*M_start)
    if start_angles is None:
        start_angles = x_base[angle_axes]
    start_angles = np.array(start_angles, dtype=float)
    if start_angles.shape != (len(angle_axes),):
        raise ValueError(
            f"start_angles must contain {len(angle_axes)} values for active_actuators={angle_labels}."
        )
    if target_reflections is None:
        x_initial = x_base.copy()
        x_initial[angle_axes] = start_angles
        target_reflections = get_reflection_count(*unpack_variables(x_initial, *M_start))
    target_reflections = int(target_reflections)
    n_steps = max(0, int(n_steps))
    step_deg = float(abs(step_deg))
    qc_tolerance = float(abs(qc_tolerance))
    correction_regularization = float(abs(correction_regularization))
    max_corrector_nfev = max(20, int(max_corrector_nfev))
    expected_u_count = target_reflections if include_edge_ends else max(target_reflections - 2, 0)

    if isinstance(preferred_axis, str):
        if preferred_axis not in angle_labels:
            raise ValueError(f"preferred_axis must be one of {angle_labels}, got {preferred_axis!r}.")
        preferred = np.zeros(len(angle_axes), dtype=float)
        preferred[angle_labels.index(preferred_axis)] = 1.0
    else:
        preferred = np.array(preferred_axis, dtype=float)
        if preferred.shape != (len(angle_axes),):
            raise ValueError(
                f"preferred_axis must be an actuator label or a {len(angle_axes)}-vector."
            )
        if np.linalg.norm(preferred) <= 0:
            raise ValueError("preferred_axis vector must be nonzero.")
        preferred = preferred / np.linalg.norm(preferred)

    def x_from_angles(theta):
        x = x_base.copy()
        x[angle_axes] = np.array(theta, dtype=float)
        return x

    def state_payload(theta, coordinate):
        x = x_from_angles(theta)
        mirrors = unpack_variables(x, *M_start)
        qc = np.array(quadcell_errors_from_variables(x, *M_start), dtype=float)
        edge_summary = reflection_edge_summary(x, *M_start, include_ends=include_edge_ends)
        jac_info = quadcell_angle_jacobian(
            *M_start,
            angles=theta,
            step_deg=jacobian_step_deg,
            active_actuators=angle_labels,
        )
        return {
            "coordinate": float(coordinate),
            "angles": np.array(theta, dtype=float),
            "x": x,
            "qc": qc,
            "qc_norm": float(np.linalg.norm(qc)),
            "qc_max_abs": float(np.max(np.abs(qc))),
            "reflection_count": int(get_reflection_count(*mirrors)),
            "min_u": float(edge_summary["min_u"]),
            "max_u": float(edge_summary["max_u"]),
            "closest_edge_margin": float(edge_summary["closest_edge_margin"]),
            "reflection_u_values": np.array(edge_summary["u_values"], dtype=float),
            "jacobian": jac_info["jacobian"],
        }

    def validate_payload(payload):
        if payload["reflection_count"] != target_reflections:
            return False, f"reflection count {payload['reflection_count']} != {target_reflections}"
        if payload["qc_max_abs"] > qc_tolerance:
            return False, f"QC max abs {payload['qc_max_abs']:.4g} exceeds {qc_tolerance}"
        if np.isfinite(payload["min_u"]) and payload["min_u"] < float(u_min):
            return False, f"min u {payload['min_u']:.4g} < {u_min}"
        if np.isfinite(payload["max_u"]) and payload["max_u"] > float(u_max):
            return False, f"max u {payload['max_u']:.4g} > {u_max}"
        return True, None

    def fixed_length_edge_penalties(theta):
        x = x_from_angles(theta)
        mirrors = unpack_variables(x, *M_start)
        if get_reflection_count(*mirrors) != target_reflections:
            return np.full(expected_u_count, 100.0, dtype=float)
        penalties = reflection_edge_penalties_from_variables(
            x,
            *M_start,
            u_min=u_min,
            u_max=u_max,
            include_ends=include_edge_ends,
        )
        if len(penalties) != expected_u_count:
            return np.full(expected_u_count, 100.0, dtype=float)
        return np.array(penalties, dtype=float)

    def correction_residual(theta, theta_pred):
        x = x_from_angles(theta)
        mirrors = unpack_variables(x, *M_start)
        qc = np.array(quadcell_errors_from_variables(x, *M_start), dtype=float)
        if get_reflection_count(*mirrors) != target_reflections:
            qc = qc + 1e3
        residuals = [
            qc[0] / float(qc_scale),
            qc[1] / float(qc_scale),
        ]
        residuals.extend(fixed_length_edge_penalties(theta) / float(sigma_edge))
        if correction_regularization > 0:
            residuals.extend((np.array(theta, dtype=float) - np.array(theta_pred, dtype=float)) / correction_regularization)
        return np.array(residuals, dtype=float)

    def correct_to_center(theta_pred):
        res = least_squares(
            fun=lambda theta: correction_residual(theta, theta_pred),
            x0=np.array(theta_pred, dtype=float),
            loss="linear",
            f_scale=1.0,
            x_scale="jac",
            max_nfev=max_corrector_nfev,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        payload = state_payload(res.x, 0.0)
        ok, reason = validate_payload(payload)
        return ok, res.x, payload, reason, res

    def nullspace_direction(theta, desired):
        jac = quadcell_angle_jacobian(
            *M_start,
            angles=theta,
            step_deg=jacobian_step_deg,
            active_actuators=angle_labels,
        )["jacobian"]
        _, singular_values, vh = np.linalg.svd(jac, full_matrices=True)
        rank = int(np.sum(singular_values > 1e-10))
        nullspace = vh[rank:].T
        if nullspace.shape[1] == 0:
            return None, singular_values
        desired = np.array(desired, dtype=float)
        if np.linalg.norm(desired) <= 0:
            desired = nullspace[:, 0]
        projected = nullspace @ (nullspace.T @ desired)
        if np.linalg.norm(projected) <= 1e-12:
            projected = nullspace[:, 0]
        direction = projected / np.linalg.norm(projected)
        if np.dot(direction, desired) < 0:
            direction = -direction
        return direction, singular_values

    if auto_center_start:
        ok, centered_start, _, failure_reason, center_res = correct_to_center(start_angles)
        if not ok:
            return {
                "success": False,
                "failure_reason": "Could not center start: " + str(failure_reason),
                "center_start_result": center_res,
                "points": [],
                "angles": np.empty((0, len(angle_axes)), dtype=float),
                "angle_labels": angle_labels,
                "active_actuators": angle_labels,
                "angle_axes": angle_axes,
                "target_reflections": int(target_reflections),
            }
        start_angles = centered_start

    start_payload = state_payload(start_angles, 0.0)
    ok, failure_reason = validate_payload(start_payload)
    if not ok:
        return {
            "success": False,
            "failure_reason": "Start is not on centered curve: " + str(failure_reason),
            "points": [],
            "angles": np.empty((0, len(angle_axes)), dtype=float),
            "angle_labels": angle_labels,
            "active_actuators": angle_labels,
            "angle_axes": angle_axes,
            "target_reflections": int(target_reflections),
        }

    initial_direction, singular_values = nullspace_direction(start_angles, preferred)
    if initial_direction is None:
        return {
            "success": False,
            "failure_reason": "QC Jacobian has no nullspace direction at start.",
            "points": [start_payload],
            "angles": np.array([start_angles], dtype=float),
            "angle_labels": angle_labels,
            "active_actuators": angle_labels,
            "angle_axes": angle_axes,
            "target_reflections": int(target_reflections),
        }

    def trace_branch(sign):
        branch_points = []
        current = start_angles.copy()
        previous_direction = sign * initial_direction
        coordinate = 0.0
        stop_reason = None
        for _ in range(n_steps):
            direction, _ = nullspace_direction(current, previous_direction)
            if direction is None:
                stop_reason = "no nullspace direction"
                break
            if np.dot(direction, previous_direction) < 0:
                direction = -direction
            theta_pred = current + float(step_deg) * direction
            ok, corrected, payload, reason, _ = correct_to_center(theta_pred)
            if not ok:
                stop_reason = reason
                break
            delta = corrected - current
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm <= 1e-12:
                stop_reason = "corrector produced zero motion"
                break
            coordinate += delta_norm
            payload["coordinate"] = float(sign * coordinate)
            payload["predictor_direction"] = direction
            branch_points.append(payload)
            current = corrected
            previous_direction = delta / delta_norm
        else:
            stop_reason = f"step limit {n_steps} reached"
        return branch_points, stop_reason

    positive_points, positive_stop = trace_branch(1.0)
    negative_points, negative_stop = trace_branch(-1.0)
    points = list(reversed(negative_points)) + [start_payload] + positive_points
    angles = np.array([point["angles"] for point in points], dtype=float)
    coordinates = np.array([point["coordinate"] for point in points], dtype=float)

    return {
        "success": True,
        "failure_reason": None,
        "points": points,
        "angles": angles,
        "coordinates": coordinates,
        "angle_labels": angle_labels,
        "active_actuators": angle_labels,
        "angle_axes": angle_axes,
        "target_reflections": int(target_reflections),
        "preferred_axis": preferred_axis,
        "initial_direction": initial_direction,
        "initial_singular_values": singular_values,
        "positive_stop_reason": positive_stop,
        "negative_stop_reason": negative_stop,
        "qc_tolerance": float(qc_tolerance),
        "u_min": float(u_min),
        "u_max": float(u_max),
        "step_deg": float(step_deg),
    }


def trace_full_centered_quadcell_angle_curve(M1, M2, M3, M4,
                                             target_reflections=None,
                                             active_actuators=None,
                                             preferred_axis=None,
                                             max_steps_per_side=2000,
                                             step_deg=0.01,
                                             qc_tolerance=0.05,
                                             u_min=0.1,
                                             u_max=0.9,
                                             **kwargs):
    """Trace a long centered-QC curve until a boundary or step limit is reached.

    This is a convenience wrapper around trace_centered_quadcell_angle_curve
    with defaults intended for exploring the full visible branch of a
    three-actuator centered curve.
    """
    labels, _ = _active_angle_axes(active_actuators)
    if preferred_axis is None:
        preferred_axis = labels[0]

    curve = trace_centered_quadcell_angle_curve(
        M1,
        M2,
        M3,
        M4,
        target_reflections=target_reflections,
        active_actuators=labels,
        preferred_axis=preferred_axis,
        n_steps=max_steps_per_side,
        step_deg=step_deg,
        qc_tolerance=qc_tolerance,
        u_min=u_min,
        u_max=u_max,
        **kwargs,
    )
    curve["full_trace"] = True
    curve["max_steps_per_side"] = int(max_steps_per_side)
    return curve


def solve_and_trace_centered_quadcell_angle_curve(M1, M2, M3, M4,
                                                  target_reflections=None,
                                                  N_R=None,
                                                  center_n_tries=2000,
                                                  center_angle_perturb=0.3,
                                                  center_seed=0,
                                                  center_u_min=None,
                                                  center_u_max=None,
                                                  center_sigma_edge=0.1,
                                                  center_final_qc_tolerance=0.5,
                                                  active_actuators=None,
                                                  preferred_axis=None,
                                                  max_steps_per_side=2000,
                                                  step_deg=0.01,
                                                  qc_tolerance=0.05,
                                                  u_min=0.1,
                                                  u_max=0.9,
                                                  **trace_kwargs):
    """Find a centered fixed-N_R configuration, then trace its centered-QC curve."""
    if N_R is not None:
        if target_reflections is not None and int(target_reflections) != int(N_R):
            raise ValueError("target_reflections and N_R were both provided with different values.")
        target_reflections = int(N_R)
    if target_reflections is None:
        raise ValueError("target_reflections or N_R must be provided.")

    if center_u_min is None:
        center_u_min = u_min
    if center_u_max is None:
        center_u_max = u_max

    centered_mirrors, center_res = center_quadcells(
        M1,
        M2,
        M3,
        M4,
        N_R=int(target_reflections),
        n_tries=center_n_tries,
        angle_perturb=center_angle_perturb,
        seed=center_seed,
        u_min=center_u_min,
        u_max=center_u_max,
        sigma_edge=center_sigma_edge,
        final_qc_tolerance=center_final_qc_tolerance,
    )

    curve = trace_full_centered_quadcell_angle_curve(
        *centered_mirrors,
        target_reflections=int(target_reflections),
        active_actuators=active_actuators,
        preferred_axis=preferred_axis,
        max_steps_per_side=max_steps_per_side,
        step_deg=step_deg,
        qc_tolerance=qc_tolerance,
        u_min=u_min,
        u_max=u_max,
        **trace_kwargs,
    )
    curve["centered_mirrors"] = centered_mirrors
    curve["center_result"] = center_res
    curve["center_n_tries"] = int(center_n_tries)
    curve["center_angle_perturb"] = float(center_angle_perturb)
    curve["center_seed"] = None if center_seed is None else int(center_seed)
    return centered_mirrors, center_res, curve


def _dangle_value_for_label(mirrors, label):
    angle_labels = ["M1.dangle", "M2.dangle", "M3.dangle", "M4.dangle"]
    if label not in angle_labels:
        raise ValueError(f"angle label must be one of {angle_labels}, got {label!r}.")
    return float(np.array(mirrors[angle_labels.index(label)], dtype=float)[2])


def _mirrors_with_dangle_value(mirrors, label, value):
    angle_labels = ["M1.dangle", "M2.dangle", "M3.dangle", "M4.dangle"]
    if label not in angle_labels:
        raise ValueError(f"angle label must be one of {angle_labels}, got {label!r}.")
    updated = [np.array(mirror, dtype=float).copy() for mirror in mirrors]
    updated[angle_labels.index(label)][2] = float(value)
    return tuple(updated)


def trace_centered_quadcell_angle_surface(M1, M2, M3, M4,
                                          target_reflections=None,
                                          N_R=None,
                                          active_actuators=("M1.dangle", "M2.dangle", "M3.dangle"),
                                          sweep_actuator="M4.dangle",
                                          sweep_values=None,
                                          sweep_offsets=None,
                                          sweep_half_span_deg=0.3,
                                          sweep_samples=13,
                                          preferred_axis=None,
                                          max_steps_per_side=800,
                                          step_deg=0.01,
                                          qc_tolerance=0.05,
                                          u_min=0.1,
                                          u_max=0.9,
                                          **trace_kwargs):
    """Trace a centered-QC 2D surface as fixed-actuator 1D slices.

    For the common M1-M3 plot, keep M4 fixed at several values. Each fixed-M4
    slice is a centered-QC curve in M1/M2/M3; the collection is a surface in
    the full four-angle space.
    """
    if N_R is not None:
        if target_reflections is not None and int(target_reflections) != int(N_R):
            raise ValueError("target_reflections and N_R were both provided with different values.")
        target_reflections = int(N_R)
    if target_reflections is None:
        target_reflections = get_reflection_count(M1, M2, M3, M4)
    target_reflections = int(target_reflections)

    base_mirrors = (
        np.array(M1, dtype=float),
        np.array(M2, dtype=float),
        np.array(M3, dtype=float),
        np.array(M4, dtype=float),
    )
    active_labels, active_axes = _active_angle_axes(active_actuators)
    if sweep_actuator in active_labels:
        raise ValueError("sweep_actuator must not also be listed in active_actuators.")
    if len(active_labels) != 3:
        raise ValueError("trace_centered_quadcell_angle_surface currently expects exactly three active actuators.")
    if preferred_axis is None:
        preferred_axis = active_labels[0]

    base_sweep_value = _dangle_value_for_label(base_mirrors, sweep_actuator)
    if sweep_values is not None:
        sweep_values = np.array(sweep_values, dtype=float)
    elif sweep_offsets is not None:
        sweep_values = base_sweep_value + np.array(sweep_offsets, dtype=float)
    else:
        sweep_values = np.linspace(
            base_sweep_value - float(sweep_half_span_deg),
            base_sweep_value + float(sweep_half_span_deg),
            max(1, int(sweep_samples)),
            dtype=float,
        )
    if sweep_values.ndim != 1 or sweep_values.size == 0:
        raise ValueError("sweep_values/sweep_offsets must define at least one sweep value.")

    x_base = pack_variables(*base_mirrors)
    base_active_angles = x_base[active_axes]
    ordered_indices = sorted(
        range(len(sweep_values)),
        key=lambda idx: (abs(float(sweep_values[idx]) - base_sweep_value), float(sweep_values[idx])),
    )
    curves_by_index = {}
    failed_slices = []

    for idx in ordered_indices:
        sweep_value = float(sweep_values[idx])
        slice_mirrors = _mirrors_with_dangle_value(base_mirrors, sweep_actuator, sweep_value)
        successful_curves = list(curves_by_index.values())
        if successful_curves:
            nearest_curve = min(
                successful_curves,
                key=lambda curve: abs(float(curve["sweep_value"]) - sweep_value),
            )
            nearest_coordinates = np.array(nearest_curve.get("coordinates", []), dtype=float)
            nearest_start_index = (
                0 if nearest_coordinates.size == 0
                else int(np.argmin(np.abs(nearest_coordinates)))
            )
            start_angles = np.array(nearest_curve["points"][nearest_start_index]["angles"], dtype=float)
        else:
            start_angles = base_active_angles

        curve = trace_full_centered_quadcell_angle_curve(
            *slice_mirrors,
            target_reflections=target_reflections,
            start_angles=start_angles,
            active_actuators=active_labels,
            preferred_axis=preferred_axis,
            max_steps_per_side=max_steps_per_side,
            step_deg=step_deg,
            qc_tolerance=qc_tolerance,
            u_min=u_min,
            u_max=u_max,
            **trace_kwargs,
        )
        curve["sweep_actuator"] = sweep_actuator
        curve["sweep_value"] = sweep_value
        curve["slice_mirrors"] = slice_mirrors
        for point in curve.get("points", []):
            point["sweep_actuator"] = sweep_actuator
            point["sweep_value"] = sweep_value

        if curve.get("success") and len(curve.get("points", [])) > 0:
            curves_by_index[idx] = curve
        else:
            failed_slices.append({
                "slice_index": int(idx),
                "sweep_value": sweep_value,
                "failure_reason": curve.get("failure_reason", "unknown failure"),
            })

    curves = [curves_by_index[idx] for idx in sorted(curves_by_index)]
    successful_sweep_values = np.array([curve["sweep_value"] for curve in curves], dtype=float)
    include_edge_ends = bool(trace_kwargs.get("include_edge_ends", False))
    base_qc = np.array(quadcell_errors_from_variables(x_base, *base_mirrors), dtype=float)
    base_edge_summary = reflection_edge_summary(x_base, *base_mirrors, include_ends=include_edge_ends)
    base_point = {
        "angles": np.array(base_active_angles, dtype=float),
        "x": np.array(x_base, dtype=float),
        "sweep_value": float(base_sweep_value),
        "sweep_actuator": sweep_actuator,
        "qc": base_qc,
        "qc_norm": float(np.linalg.norm(base_qc)),
        "qc_max_abs": float(np.max(np.abs(base_qc))),
        "reflection_count": int(get_reflection_count(*base_mirrors)),
        "min_u": float(base_edge_summary["min_u"]),
        "max_u": float(base_edge_summary["max_u"]),
        "closest_edge_margin": float(base_edge_summary["closest_edge_margin"]),
        "reflection_u_values": np.array(base_edge_summary["u_values"], dtype=float),
    }
    return {
        "success": len(curves) > 0,
        "failure_reason": None if curves else "No surface slices could be traced.",
        "curves": curves,
        "failed_slices": failed_slices,
        "sweep_values": np.array(sweep_values, dtype=float),
        "successful_sweep_values": successful_sweep_values,
        "base_sweep_value": float(base_sweep_value),
        "base_active_angles": np.array(base_active_angles, dtype=float),
        "base_mirrors": base_mirrors,
        "base_x": np.array(x_base, dtype=float),
        "base_point": base_point,
        "sweep_actuator": sweep_actuator,
        "active_actuators": active_labels,
        "angle_labels": active_labels,
        "target_reflections": int(target_reflections),
        "preferred_axis": preferred_axis,
        "qc_tolerance": float(qc_tolerance),
        "u_min": float(u_min),
        "u_max": float(u_max),
        "step_deg": float(step_deg),
        "max_steps_per_side": int(max_steps_per_side),
    }


def solve_and_trace_centered_quadcell_angle_surface(M1, M2, M3, M4,
                                                    target_reflections=None,
                                                    N_R=None,
                                                    center_n_tries=2000,
                                                    center_angle_perturb=0.3,
                                                    center_seed=0,
                                                    center_u_min=None,
                                                    center_u_max=None,
                                                    center_sigma_edge=0.1,
                                                    center_final_qc_tolerance=0.5,
                                                    active_actuators=("M1.dangle", "M2.dangle", "M3.dangle"),
                                                    sweep_actuator="M4.dangle",
                                                    sweep_values=None,
                                                    sweep_offsets=None,
                                                    sweep_half_span_deg=0.3,
                                                    sweep_samples=13,
                                                    preferred_axis=None,
                                                    max_steps_per_side=800,
                                                    step_deg=0.01,
                                                    qc_tolerance=0.05,
                                                    u_min=0.1,
                                                    u_max=0.9,
                                                    **trace_kwargs):
    """Find a centered fixed-N_R config, then trace fixed-actuator slices."""
    if N_R is not None:
        if target_reflections is not None and int(target_reflections) != int(N_R):
            raise ValueError("target_reflections and N_R were both provided with different values.")
        target_reflections = int(N_R)
    if target_reflections is None:
        raise ValueError("target_reflections or N_R must be provided.")

    if center_u_min is None:
        center_u_min = u_min
    if center_u_max is None:
        center_u_max = u_max

    centered_mirrors, center_res = center_quadcells(
        M1,
        M2,
        M3,
        M4,
        N_R=int(target_reflections),
        n_tries=center_n_tries,
        angle_perturb=center_angle_perturb,
        seed=center_seed,
        u_min=center_u_min,
        u_max=center_u_max,
        sigma_edge=center_sigma_edge,
        final_qc_tolerance=center_final_qc_tolerance,
    )
    surface = trace_centered_quadcell_angle_surface(
        *centered_mirrors,
        target_reflections=int(target_reflections),
        active_actuators=active_actuators,
        sweep_actuator=sweep_actuator,
        sweep_values=sweep_values,
        sweep_offsets=sweep_offsets,
        sweep_half_span_deg=sweep_half_span_deg,
        sweep_samples=sweep_samples,
        preferred_axis=preferred_axis,
        max_steps_per_side=max_steps_per_side,
        step_deg=step_deg,
        qc_tolerance=qc_tolerance,
        u_min=u_min,
        u_max=u_max,
        **trace_kwargs,
    )
    surface["centered_mirrors"] = centered_mirrors
    surface["center_result"] = center_res
    surface["center_n_tries"] = int(center_n_tries)
    surface["center_angle_perturb"] = float(center_angle_perturb)
    surface["center_seed"] = None if center_seed is None else int(center_seed)
    return centered_mirrors, center_res, surface


def _axis_indices_for_labels(labels, axes):
    labels = list(labels)
    if axes is None:
        if len(labels) < 3:
            raise ValueError("At least three angle labels are required.")
        return labels[:3], [0, 1, 2]

    axis_specs = list(axes)
    if len(axis_specs) != 3:
        raise ValueError("axes must contain exactly three labels or indices.")

    axis_titles = []
    axis_indices = []
    for axis in axis_specs:
        if isinstance(axis, str):
            if axis not in labels:
                raise ValueError(f"axis {axis!r} is not in angle labels {labels}.")
            axis_titles.append(axis)
            axis_indices.append(labels.index(axis))
        else:
            index = int(axis)
            if index < 0 or index >= len(labels):
                raise ValueError("axes contains an out-of-range angle index.")
            axis_titles.append(labels[index])
            axis_indices.append(index)
    return axis_titles, axis_indices


def _surface_projected_records(surface, axes=None, point_stride=1):
    curves = list(surface.get("curves", []))
    if len(curves) == 0:
        return np.empty((0, 3), dtype=float), [], list(surface.get("angle_labels", []))

    point_stride = max(1, int(point_stride))
    base_labels = list(curves[0].get("angle_labels", surface.get("angle_labels", [])))
    axis_titles, _ = _axis_indices_for_labels(base_labels, axes)
    coords = []
    records = []
    for curve_idx, curve in enumerate(curves):
        angle_labels = list(curve.get("angle_labels", []))
        _, axis_indices = _axis_indices_for_labels(angle_labels, axis_titles)
        points = curve.get("points", [])
        for point_idx in range(0, len(points), point_stride):
            point = points[point_idx]
            angles = np.array(point["angles"], dtype=float)
            coord = angles[axis_indices].astype(float)
            coords.append(coord)
            records.append({
                "curve_index": int(curve_idx),
                "point_index": int(point_idx),
                "coordinates": coord,
                "angles": angles,
                "sweep_value": float(point.get("sweep_value", curve.get("sweep_value", np.nan))),
                "reflection_count": int(point.get("reflection_count", curve.get("target_reflections", -1))),
                "qc": np.array(point.get("qc", [np.nan, np.nan]), dtype=float),
                "qc_max_abs": float(point.get("qc_max_abs", np.nan)),
                "point": point,
                "curve": curve,
            })
    if len(coords) == 0:
        return np.empty((0, 3), dtype=float), records, axis_titles
    return np.array(coords, dtype=float), records, axis_titles


def _cloud_projected_records(cloud, axes=None, point_stride=1):
    points = list(cloud.get("points", []))
    point_stride = max(1, int(point_stride))
    labels = list(cloud.get("angle_labels", []))
    axis_titles, axis_indices = _axis_indices_for_labels(labels, axes)
    coords = []
    records = []
    for point_idx in range(0, len(points), point_stride):
        point = points[point_idx]
        angles = np.array(point["angles"], dtype=float)
        coord = angles[axis_indices].astype(float)
        coords.append(coord)
        records.append({
            "point_index": int(point_idx),
            "coordinates": coord,
            "angles": angles,
            "sweep_value": float(point.get("sweep_value", np.nan)),
            "reflection_count": int(point.get("reflection_count", cloud.get("target_reflections", -1))),
            "qc": np.array(point.get("qc", [np.nan, np.nan]), dtype=float),
            "qc_max_abs": float(point.get("qc_max_abs", np.nan)),
            "point": point,
        })
    if len(coords) == 0:
        return np.empty((0, 3), dtype=float), records, axis_titles
    return np.array(coords, dtype=float), records, axis_titles


def _nearest_record_pair(reference_coords, reference_records, target_coords, target_records):
    if len(reference_records) == 0 or len(target_records) == 0:
        raise ValueError("Both reference and target point sets must be non-empty.")
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(target_coords)
        distances, target_indices = tree.query(reference_coords, k=1)
        reference_index = int(np.argmin(distances))
        target_index = int(target_indices[reference_index])
        distance = float(distances[reference_index])
    except Exception:
        best = (np.inf, 0, 0)
        chunk_size = 4096
        for start in range(0, len(reference_coords), chunk_size):
            chunk = reference_coords[start:start + chunk_size]
            diffs = chunk[:, None, :] - target_coords[None, :, :]
            distances = np.linalg.norm(diffs, axis=2)
            flat_index = int(np.argmin(distances))
            local_ref_index, target_index = np.unravel_index(flat_index, distances.shape)
            distance = float(distances[local_ref_index, target_index])
            if distance < best[0]:
                best = (distance, start + int(local_ref_index), int(target_index))
        distance, reference_index, target_index = best
    return distance, reference_records[reference_index], target_records[target_index]


def find_nearest_surface_curve_by_projection(reference_surface, target_surface,
                                             axes=None,
                                             reference_point_stride=1,
                                             target_point_stride=1):
    """Find the target surface curve closest to a reference surface projection."""
    reference_coords, reference_records, axis_titles = _surface_projected_records(
        reference_surface,
        axes=axes,
        point_stride=reference_point_stride,
    )
    target_coords, target_records, _ = _surface_projected_records(
        target_surface,
        axes=axis_titles,
        point_stride=target_point_stride,
    )
    distance, reference_record, target_record = _nearest_record_pair(
        reference_coords,
        reference_records,
        target_coords,
        target_records,
    )
    return {
        "distance": float(distance),
        "axes": axis_titles,
        "reference_curve_index": int(reference_record["curve_index"]),
        "reference_point_index": int(reference_record["point_index"]),
        "reference_coordinates": np.array(reference_record["coordinates"], dtype=float),
        "reference_sweep_value": float(reference_record["sweep_value"]),
        "reference_qc": np.array(reference_record["qc"], dtype=float),
        "target_curve_index": int(target_record["curve_index"]),
        "target_point_index": int(target_record["point_index"]),
        "target_coordinates": np.array(target_record["coordinates"], dtype=float),
        "target_sweep_value": float(target_record["sweep_value"]),
        "target_qc": np.array(target_record["qc"], dtype=float),
        "target_curve": target_record["curve"],
        "target_point": target_record["point"],
        "reference_curve": reference_record["curve"],
        "reference_point": reference_record["point"],
    }


def find_nearest_surface_cloud_point_by_projection(reference_surface, target_cloud,
                                                   axes=None,
                                                   reference_point_stride=1,
                                                   cloud_point_stride=1):
    """Find the target cloud point closest to a reference surface projection."""
    reference_coords, reference_records, axis_titles = _surface_projected_records(
        reference_surface,
        axes=axes,
        point_stride=reference_point_stride,
    )
    cloud_coords, cloud_records, _ = _cloud_projected_records(
        target_cloud,
        axes=axis_titles,
        point_stride=cloud_point_stride,
    )
    distance, reference_record, cloud_record = _nearest_record_pair(
        reference_coords,
        reference_records,
        cloud_coords,
        cloud_records,
    )
    return {
        "distance": float(distance),
        "axes": axis_titles,
        "reference_curve_index": int(reference_record["curve_index"]),
        "reference_point_index": int(reference_record["point_index"]),
        "reference_coordinates": np.array(reference_record["coordinates"], dtype=float),
        "reference_sweep_value": float(reference_record["sweep_value"]),
        "reference_qc": np.array(reference_record["qc"], dtype=float),
        "cloud_point_index": int(cloud_record["point_index"]),
        "cloud_coordinates": np.array(cloud_record["coordinates"], dtype=float),
        "cloud_sweep_value": float(cloud_record["sweep_value"]),
        "cloud_qc": np.array(cloud_record["qc"], dtype=float),
        "cloud_qc_max_abs": float(cloud_record["qc_max_abs"]),
        "cloud_point": cloud_record["point"],
        "reference_point": reference_record["point"],
    }


def _normal_plane_basis(tangent):
    tangent = np.array(tangent, dtype=float)
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
        tangent = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        tangent = tangent / norm
    basis_vectors = np.eye(3)
    seed = basis_vectors[int(np.argmin(np.abs(basis_vectors @ tangent)))]
    n1 = np.cross(tangent, seed)
    n1_norm = float(np.linalg.norm(n1))
    if n1_norm <= 1e-12:
        seed = np.array([0.0, 1.0, 0.0], dtype=float)
        n1 = np.cross(tangent, seed)
        n1_norm = float(np.linalg.norm(n1))
    n1 = n1 / n1_norm
    n2 = np.cross(tangent, n1)
    n2 = n2 / max(float(np.linalg.norm(n2)), 1e-12)
    return n1, n2


def _linearized_qc_radius(base_qc, jacobian, direction, qc_limit, max_radius):
    base_qc = np.array(base_qc, dtype=float)
    jacobian = np.array(jacobian, dtype=float)
    direction = np.array(direction, dtype=float)
    qc_limit = float(abs(qc_limit))
    max_radius = float(abs(max_radius))

    def valid(radius):
        qc = base_qc + jacobian @ (float(radius) * direction)
        return bool(np.max(np.abs(qc)) <= qc_limit)

    if not valid(0.0):
        return 0.0
    if valid(max_radius):
        return max_radius

    lo = 0.0
    hi = max_radius
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if valid(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def sample_quadcell_tolerance_cloud_around_surface_curve(
        surface,
        curve_indices=None,
        qc_limit=3.0,
        max_angle_radius_deg=0.25,
        radial_samples=4,
        angular_samples=16,
        max_curve_points=200,
        point_stride=None,
        adaptive_radius=True,
        target_reflections=None,
        u_min=None,
        u_max=None,
        include_edge_ends=False,
        include_center_points=True):
    """Sample a loose-QC target cloud around selected fixed-sweep surface curves."""
    curves = list(surface.get("curves", []))
    if len(curves) == 0:
        raise ValueError("surface contains no curves.")

    if curve_indices is None:
        curve_indices = list(range(len(curves)))
    elif isinstance(curve_indices, (int, np.integer)):
        curve_indices = [int(curve_indices)]
    else:
        curve_indices = [int(index) for index in curve_indices]
    for index in curve_indices:
        if index < 0 or index >= len(curves):
            raise ValueError(f"curve index {index} is out of range for {len(curves)} curves.")

    qc_limit = float(abs(qc_limit))
    max_angle_radius_deg = float(abs(max_angle_radius_deg))
    radial_samples = max(0, int(radial_samples))
    angular_samples = max(1, int(angular_samples))
    if target_reflections is None:
        target_reflections = surface.get("target_reflections")
    if target_reflections is None:
        raise ValueError("target_reflections must be provided or stored in the surface.")
    target_reflections = int(target_reflections)
    if u_min is None:
        u_min = float(surface.get("u_min", 0.1))
    if u_max is None:
        u_max = float(surface.get("u_max", 0.9))

    cloud_points = []
    checked_count = 0
    rejected_counts = {
        "reflection_count": 0,
        "qc_limit": 0,
        "u_bounds": 0,
    }
    selected_point_count = 0

    for curve_index in curve_indices:
        curve = curves[curve_index]
        points = list(curve.get("points", []))
        if len(points) == 0:
            continue
        angle_labels = list(curve.get("angle_labels", surface.get("angle_labels", [])))
        if len(angle_labels) != 3:
            raise ValueError("Cloud sampling currently expects curves with exactly three active angles.")
        _, angle_axes = _active_angle_axes(angle_labels)
        slice_mirrors = curve.get("slice_mirrors")
        if slice_mirrors is None:
            slice_mirrors = surface.get("base_mirrors")
        if slice_mirrors is None:
            raise ValueError("surface curve is missing slice_mirrors/base_mirrors needed for simulation checks.")
        slice_mirrors = tuple(np.array(mirror, dtype=float) for mirror in slice_mirrors)

        if point_stride is None:
            stride = 1
            if max_curve_points is not None and max_curve_points > 0:
                stride = max(1, int(math.ceil(len(points) / float(max_curve_points))))
        else:
            stride = max(1, int(point_stride))
        selected_indices = list(range(0, len(points), stride))
        selected_point_count += len(selected_indices)

        curve_angles = np.array([point["angles"] for point in points], dtype=float)
        for point_index in selected_indices:
            center_point = points[point_index]
            center_angles = np.array(center_point["angles"], dtype=float)
            if len(points) == 1:
                tangent = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                prev_index = max(0, point_index - 1)
                next_index = min(len(points) - 1, point_index + 1)
                tangent = curve_angles[next_index] - curve_angles[prev_index]
            normal_1, normal_2 = _normal_plane_basis(tangent)

            base_qc = np.array(center_point.get("qc", [0.0, 0.0]), dtype=float)
            if adaptive_radius:
                jacobian = quadcell_angle_jacobian(
                    *slice_mirrors,
                    angles=center_angles,
                    active_actuators=angle_labels,
                )["jacobian"]
            else:
                jacobian = None

            candidates = []
            if include_center_points:
                candidates.append((center_angles, 0.0, np.nan))
            for angle_index in range(angular_samples):
                phi = 2.0 * math.pi * float(angle_index) / float(angular_samples)
                direction = math.cos(phi) * normal_1 + math.sin(phi) * normal_2
                if adaptive_radius:
                    radius = _linearized_qc_radius(
                        base_qc,
                        jacobian,
                        direction,
                        qc_limit=qc_limit,
                        max_radius=max_angle_radius_deg,
                    )
                else:
                    radius = max_angle_radius_deg
                if radius <= 1e-12 or radial_samples <= 0:
                    continue
                for frac in np.linspace(1.0 / radial_samples, 1.0, radial_samples):
                    candidates.append((center_angles + float(frac) * radius * direction,
                                       float(frac) * radius,
                                       float(phi)))

            for candidate_angles, candidate_radius, candidate_phi in candidates:
                checked_count += 1
                x_candidate = np.array(center_point["x"], dtype=float).copy()
                x_candidate[angle_axes] = candidate_angles
                mirrors = unpack_variables(x_candidate, *slice_mirrors)
                reflection_count = int(get_reflection_count(*mirrors))
                if reflection_count != target_reflections:
                    rejected_counts["reflection_count"] += 1
                    continue

                qc = np.array(quadcell_errors_from_variables(x_candidate, *slice_mirrors), dtype=float)
                qc_max_abs = float(np.max(np.abs(qc)))
                if qc_max_abs > qc_limit:
                    rejected_counts["qc_limit"] += 1
                    continue

                edge_summary = reflection_edge_summary(
                    x_candidate,
                    *slice_mirrors,
                    include_ends=include_edge_ends,
                )
                min_u = float(edge_summary["min_u"])
                max_u = float(edge_summary["max_u"])
                if np.isfinite(min_u) and min_u < float(u_min):
                    rejected_counts["u_bounds"] += 1
                    continue
                if np.isfinite(max_u) and max_u > float(u_max):
                    rejected_counts["u_bounds"] += 1
                    continue

                cloud_points.append({
                    "angles": np.array(candidate_angles, dtype=float),
                    "x": x_candidate,
                    "qc": qc,
                    "qc_norm": float(np.linalg.norm(qc)),
                    "qc_max_abs": qc_max_abs,
                    "reflection_count": reflection_count,
                    "min_u": min_u,
                    "max_u": max_u,
                    "closest_edge_margin": float(edge_summary["closest_edge_margin"]),
                    "reflection_u_values": np.array(edge_summary["u_values"], dtype=float),
                    "sweep_value": float(center_point.get("sweep_value", curve.get("sweep_value", np.nan))),
                    "sweep_actuator": curve.get("sweep_actuator", surface.get("sweep_actuator")),
                    "source_curve_index": int(curve_index),
                    "source_point_index": int(point_index),
                    "offset_radius_deg": float(candidate_radius),
                    "offset_phi_rad": float(candidate_phi),
                })

    angles = (
        np.array([point["angles"] for point in cloud_points], dtype=float)
        if cloud_points else np.empty((0, 3), dtype=float)
    )
    return {
        "success": len(cloud_points) > 0,
        "failure_reason": None if cloud_points else "No valid cloud points were sampled.",
        "points": cloud_points,
        "angles": angles,
        "angle_labels": list(surface.get("angle_labels", curves[curve_indices[0]].get("angle_labels", []))),
        "active_actuators": list(surface.get("active_actuators", curves[curve_indices[0]].get("active_actuators", []))),
        "target_reflections": target_reflections,
        "qc_limit": qc_limit,
        "u_min": float(u_min),
        "u_max": float(u_max),
        "curve_indices": [int(index) for index in curve_indices],
        "max_angle_radius_deg": max_angle_radius_deg,
        "radial_samples": radial_samples,
        "angular_samples": angular_samples,
        "adaptive_radius": bool(adaptive_radius),
        "checked_count": int(checked_count),
        "kept_count": int(len(cloud_points)),
        "selected_center_point_count": int(selected_point_count),
        "rejected_counts": rejected_counts,
    }


def sample_quadcell_tolerance_tube_around_surface_curve(
        surface,
        curve_index,
        qc_limit=3.0,
        max_angle_radius_deg=0.25,
        angular_samples=32,
        max_curve_points=250,
        point_stride=None,
        adaptive_radius=True,
        target_reflections=None,
        u_min=None,
        u_max=None,
        include_edge_ends=False,
        min_radius_deg=1e-6,
        binary_search_iterations=24):
    """Build a translucent tube around one centered-QC surface curve.

    The tube lives in the plotted active-angle coordinates. Its local
    cross-section is sampled in the normal plane to the centered curve, with
    the boundary chosen so that the full simulation still satisfies the
    reflection count, QC limit, and reflection-u bounds.
    """
    curves = list(surface.get("curves", []))
    if len(curves) == 0:
        raise ValueError("surface contains no curves.")
    curve_index = int(curve_index)
    if curve_index < 0 or curve_index >= len(curves):
        raise ValueError(f"curve_index {curve_index} is out of range for {len(curves)} curves.")

    curve = curves[curve_index]
    points = list(curve.get("points", []))
    if len(points) == 0:
        raise ValueError("selected surface curve contains no points.")

    angle_labels = list(curve.get("angle_labels", surface.get("angle_labels", [])))
    if len(angle_labels) != 3:
        raise ValueError("Tube sampling currently expects curves with exactly three active angles.")
    _, angle_axes = _active_angle_axes(angle_labels)

    slice_mirrors = curve.get("slice_mirrors")
    if slice_mirrors is None:
        slice_mirrors = surface.get("base_mirrors")
    if slice_mirrors is None:
        raise ValueError("surface curve is missing slice_mirrors/base_mirrors needed for simulation checks.")
    slice_mirrors = tuple(np.array(mirror, dtype=float) for mirror in slice_mirrors)

    qc_limit = float(abs(qc_limit))
    max_angle_radius_deg = float(abs(max_angle_radius_deg))
    angular_samples = max(3, int(angular_samples))
    min_radius_deg = float(abs(min_radius_deg))
    binary_search_iterations = max(1, int(binary_search_iterations))
    if target_reflections is None:
        target_reflections = surface.get("target_reflections", curve.get("target_reflections"))
    if target_reflections is None:
        raise ValueError("target_reflections must be provided or stored in the surface.")
    target_reflections = int(target_reflections)
    if u_min is None:
        u_min = float(surface.get("u_min", 0.1))
    if u_max is None:
        u_max = float(surface.get("u_max", 0.9))

    if point_stride is None:
        stride = 1
        if max_curve_points is not None and max_curve_points > 0:
            stride = max(1, int(math.ceil(len(points) / float(max_curve_points))))
    else:
        stride = max(1, int(point_stride))
    selected_indices = list(range(0, len(points), stride))
    if selected_indices[-1] != len(points) - 1:
        selected_indices.append(len(points) - 1)

    curve_angles = np.array([point["angles"] for point in points], dtype=float)
    rings = []
    rejected_counts = {
        "center_invalid": 0,
        "zero_radius": 0,
        "reflection_count": 0,
        "qc_limit": 0,
        "u_bounds": 0,
    }

    def evaluate_candidate(center_point, candidate_angles):
        x_candidate = np.array(center_point["x"], dtype=float).copy()
        x_candidate[angle_axes] = np.array(candidate_angles, dtype=float)
        mirrors = unpack_variables(x_candidate, *slice_mirrors)
        reflection_count = int(get_reflection_count(*mirrors))
        if reflection_count != target_reflections:
            return False, x_candidate, None, "reflection_count"
        qc = np.array(quadcell_errors_from_variables(x_candidate, *slice_mirrors), dtype=float)
        qc_max_abs = float(np.max(np.abs(qc)))
        if qc_max_abs > qc_limit:
            return False, x_candidate, {
                "qc": qc,
                "qc_max_abs": qc_max_abs,
                "reflection_count": reflection_count,
            }, "qc_limit"
        edge_summary = reflection_edge_summary(
            x_candidate,
            *slice_mirrors,
            include_ends=include_edge_ends,
        )
        min_u = float(edge_summary["min_u"])
        max_u = float(edge_summary["max_u"])
        if np.isfinite(min_u) and min_u < float(u_min):
            return False, x_candidate, {
                "qc": qc,
                "qc_max_abs": qc_max_abs,
                "reflection_count": reflection_count,
                "edge_summary": edge_summary,
            }, "u_bounds"
        if np.isfinite(max_u) and max_u > float(u_max):
            return False, x_candidate, {
                "qc": qc,
                "qc_max_abs": qc_max_abs,
                "reflection_count": reflection_count,
                "edge_summary": edge_summary,
            }, "u_bounds"
        return True, x_candidate, {
            "qc": qc,
            "qc_norm": float(np.linalg.norm(qc)),
            "qc_max_abs": qc_max_abs,
            "reflection_count": reflection_count,
            "min_u": min_u,
            "max_u": max_u,
            "closest_edge_margin": float(edge_summary["closest_edge_margin"]),
            "reflection_u_values": np.array(edge_summary["u_values"], dtype=float),
        }, None

    for point_index in selected_indices:
        center_point = points[point_index]
        center_angles = np.array(center_point["angles"], dtype=float)
        center_ok, _, _, center_failure = evaluate_candidate(center_point, center_angles)
        if not center_ok:
            rejected_counts["center_invalid"] += 1
            continue

        if len(points) == 1:
            tangent = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            prev_index = max(0, point_index - 1)
            next_index = min(len(points) - 1, point_index + 1)
            tangent = curve_angles[next_index] - curve_angles[prev_index]
        normal_1, normal_2 = _normal_plane_basis(tangent)

        base_qc = np.array(center_point.get("qc", [0.0, 0.0]), dtype=float)
        if adaptive_radius:
            jacobian = quadcell_angle_jacobian(
                *slice_mirrors,
                angles=center_angles,
                active_actuators=angle_labels,
            )["jacobian"]
        else:
            jacobian = None

        ring_vertices = []
        ring_payloads = []
        ring_radii = []
        for angle_index in range(angular_samples):
            phi = 2.0 * math.pi * float(angle_index) / float(angular_samples)
            direction = math.cos(phi) * normal_1 + math.sin(phi) * normal_2
            if adaptive_radius:
                proposed_radius = _linearized_qc_radius(
                    base_qc,
                    jacobian,
                    direction,
                    qc_limit=qc_limit,
                    max_radius=max_angle_radius_deg,
                )
            else:
                proposed_radius = max_angle_radius_deg

            if proposed_radius <= min_radius_deg:
                rejected_counts["zero_radius"] += 1
                ring_vertices.append(center_angles.copy())
                ring_payloads.append(None)
                ring_radii.append(0.0)
                continue

            ok, x_candidate, payload, failure = evaluate_candidate(
                center_point,
                center_angles + proposed_radius * direction,
            )
            if ok:
                radius = proposed_radius
                valid_payload = payload
                valid_x = x_candidate
            else:
                rejected_counts[failure] = rejected_counts.get(failure, 0) + 1
                lo = 0.0
                hi = proposed_radius
                valid_payload = None
                valid_x = None
                for _ in range(binary_search_iterations):
                    mid = 0.5 * (lo + hi)
                    ok_mid, x_mid, payload_mid, _ = evaluate_candidate(
                        center_point,
                        center_angles + mid * direction,
                    )
                    if ok_mid:
                        lo = mid
                        valid_payload = payload_mid
                        valid_x = x_mid
                    else:
                        hi = mid
                radius = lo

            if radius <= min_radius_deg or valid_payload is None:
                rejected_counts["zero_radius"] += 1
                ring_vertices.append(center_angles.copy())
                ring_payloads.append(None)
                ring_radii.append(0.0)
                continue

            candidate_angles = center_angles + radius * direction
            valid_payload = dict(valid_payload)
            valid_payload.update({
                "angles": np.array(candidate_angles, dtype=float),
                "x": valid_x,
                "sweep_value": float(center_point.get("sweep_value", curve.get("sweep_value", np.nan))),
                "sweep_actuator": curve.get("sweep_actuator", surface.get("sweep_actuator")),
                "source_curve_index": int(curve_index),
                "source_point_index": int(point_index),
                "offset_radius_deg": float(radius),
                "offset_phi_rad": float(phi),
            })
            ring_vertices.append(np.array(candidate_angles, dtype=float))
            ring_payloads.append(valid_payload)
            ring_radii.append(float(radius))

        rings.append({
            "source_point_index": int(point_index),
            "center_angles": np.array(center_angles, dtype=float),
            "center_point": center_point,
            "vertices": np.array(ring_vertices, dtype=float),
            "payloads": ring_payloads,
            "radii": np.array(ring_radii, dtype=float),
            "sweep_value": float(center_point.get("sweep_value", curve.get("sweep_value", np.nan))),
        })

    vertices = []
    customdata = []
    ring_vertex_indices = []
    boundary_points = []
    for ring_index, ring in enumerate(rings):
        indices = []
        for vertex_index, vertex in enumerate(ring["vertices"]):
            indices.append(len(vertices))
            vertices.append(np.array(vertex, dtype=float))
            payload = ring["payloads"][vertex_index]
            radius = float(ring["radii"][vertex_index])
            if payload is None:
                customdata.append([
                    float(ring_index),
                    float(vertex_index),
                    float(ring["source_point_index"]),
                    float(radius),
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ])
            else:
                boundary_points.append(payload)
                qc = np.array(payload["qc"], dtype=float)
                customdata.append([
                    float(ring_index),
                    float(vertex_index),
                    float(ring["source_point_index"]),
                    float(radius),
                    float(qc[0]),
                    float(qc[1]),
                    float(payload["qc_max_abs"]),
                    float(payload["min_u"]),
                    float(payload["max_u"]),
                    float(payload["closest_edge_margin"]),
                ])
        ring_vertex_indices.append(indices)

    faces_i = []
    faces_j = []
    faces_k = []
    for ring_index in range(len(ring_vertex_indices) - 1):
        ring_a = ring_vertex_indices[ring_index]
        ring_b = ring_vertex_indices[ring_index + 1]
        if len(ring_a) != angular_samples or len(ring_b) != angular_samples:
            continue
        for angle_index in range(angular_samples):
            a0 = ring_a[angle_index]
            a1 = ring_a[(angle_index + 1) % angular_samples]
            b0 = ring_b[angle_index]
            b1 = ring_b[(angle_index + 1) % angular_samples]
            faces_i.extend([a0, a1])
            faces_j.extend([b0, b0])
            faces_k.extend([a1, b1])

    vertices_array = (
        np.array(vertices, dtype=float)
        if vertices else np.empty((0, 3), dtype=float)
    )
    return {
        "success": len(rings) > 0 and len(faces_i) > 0,
        "failure_reason": None if len(rings) > 0 and len(faces_i) > 0 else "No tube mesh could be built.",
        "rings": rings,
        "vertices": vertices_array,
        "faces_i": np.array(faces_i, dtype=int),
        "faces_j": np.array(faces_j, dtype=int),
        "faces_k": np.array(faces_k, dtype=int),
        "customdata": np.array(customdata, dtype=float) if customdata else np.empty((0, 10), dtype=float),
        "boundary_points": boundary_points,
        "angle_labels": angle_labels,
        "active_actuators": angle_labels,
        "target_reflections": target_reflections,
        "qc_limit": qc_limit,
        "u_min": float(u_min),
        "u_max": float(u_max),
        "curve_index": int(curve_index),
        "sweep_value": float(curve.get("sweep_value", np.nan)),
        "sweep_actuator": curve.get("sweep_actuator", surface.get("sweep_actuator")),
        "max_angle_radius_deg": max_angle_radius_deg,
        "angular_samples": angular_samples,
        "selected_center_point_count": int(len(rings)),
        "mesh_vertex_count": int(len(vertices)),
        "mesh_face_count": int(len(faces_i)),
        "rejected_counts": rejected_counts,
    }


def scan_one_actuator_target_cloud_from_surface(
        surface,
        target_reflections,
        actuator=None,
        actuator_values=None,
        actuator_offsets=None,
        actuator_half_span_deg=0.8,
        actuator_samples=161,
        qc_limit=3.0,
        curve_indices=None,
        max_source_points_per_curve=300,
        point_stride=None,
        u_min=None,
        u_max=None,
        include_edge_ends=False,
        keep_best_per_source=False,
        min_abs_actuator_delta=1e-12):
    """Scan one actuator from a source surface and keep valid target-N_R landings.

    This is the direct test for a one-actuator reflection-count jump. For each
    sampled source point on the surface, all active plotted angles are held
    fixed while one actuator is scanned. Valid target endpoints are returned as
    a cloud in the same projected angle coordinates.
    """
    curves = list(surface.get("curves", []))
    if len(curves) == 0:
        raise ValueError("surface contains no curves.")

    if actuator is None:
        actuator = surface.get("sweep_actuator", "M4.dangle")
    if isinstance(actuator, str):
        actuator_label = actuator
    else:
        raise ValueError("actuator must be a dangle actuator label such as 'M4.dangle'.")
    _, actuator_axis_array = _active_angle_axes([actuator_label])
    actuator_axis = int(actuator_axis_array[0])

    active_labels = list(surface.get("angle_labels", curves[0].get("angle_labels", [])))
    if actuator_label in active_labels:
        raise ValueError(
            "actuator must be outside the plotted active_actuators for a hidden one-actuator jump. "
            f"Got actuator={actuator_label!r}, active_actuators={active_labels}."
        )
    _, active_axes = _active_angle_axes(active_labels)

    if curve_indices is None:
        curve_indices = list(range(len(curves)))
    elif isinstance(curve_indices, (int, np.integer)):
        curve_indices = [int(curve_indices)]
    else:
        curve_indices = [int(index) for index in curve_indices]
    for index in curve_indices:
        if index < 0 or index >= len(curves):
            raise ValueError(f"curve index {index} is out of range for {len(curves)} curves.")

    if actuator_values is not None:
        actuator_values = np.array(actuator_values, dtype=float)
        if actuator_values.ndim != 1 or actuator_values.size == 0:
            raise ValueError("actuator_values must contain at least one value.")
        offsets = None
    else:
        if actuator_offsets is None:
            offsets = np.linspace(
                -float(abs(actuator_half_span_deg)),
                float(abs(actuator_half_span_deg)),
                max(1, int(actuator_samples)),
                dtype=float,
            )
        else:
            offsets = np.array(actuator_offsets, dtype=float)
            if offsets.ndim != 1 or offsets.size == 0:
                raise ValueError("actuator_offsets must contain at least one value.")
        actuator_values = None

    target_reflections = int(target_reflections)
    qc_limit = float(abs(qc_limit))
    if u_min is None:
        u_min = float(surface.get("u_min", 0.1))
    if u_max is None:
        u_max = float(surface.get("u_max", 0.9))
    min_abs_actuator_delta = float(abs(min_abs_actuator_delta))

    cloud_points = []
    checked_count = 0
    rejected_counts = {
        "min_delta": 0,
        "reflection_count": 0,
        "qc_limit": 0,
        "u_bounds": 0,
    }
    selected_source_count = 0

    for curve_index in curve_indices:
        curve = curves[curve_index]
        points = list(curve.get("points", []))
        if len(points) == 0:
            continue
        slice_mirrors = curve.get("slice_mirrors")
        if slice_mirrors is None:
            slice_mirrors = surface.get("base_mirrors")
        if slice_mirrors is None:
            raise ValueError("surface curve is missing slice_mirrors/base_mirrors needed for simulation checks.")
        slice_mirrors = tuple(np.array(mirror, dtype=float) for mirror in slice_mirrors)

        if point_stride is None:
            stride = 1
            if max_source_points_per_curve is not None and max_source_points_per_curve > 0:
                stride = max(1, int(math.ceil(len(points) / float(max_source_points_per_curve))))
        else:
            stride = max(1, int(point_stride))
        selected_indices = list(range(0, len(points), stride))
        selected_source_count += len(selected_indices)

        for point_index in selected_indices:
            source_point = points[point_index]
            source_x = np.array(source_point["x"], dtype=float)
            source_actuator_value = float(source_x[actuator_axis])
            if actuator_values is None:
                scan_values = source_actuator_value + offsets
            else:
                scan_values = actuator_values

            source_candidates = []
            for target_actuator_value in scan_values:
                target_actuator_value = float(target_actuator_value)
                actuator_delta = target_actuator_value - source_actuator_value
                if abs(actuator_delta) < min_abs_actuator_delta:
                    rejected_counts["min_delta"] += 1
                    continue

                checked_count += 1
                x_candidate = source_x.copy()
                x_candidate[actuator_axis] = target_actuator_value
                mirrors = unpack_variables(x_candidate, *slice_mirrors)
                reflection_count = int(get_reflection_count(*mirrors))
                if reflection_count != target_reflections:
                    rejected_counts["reflection_count"] += 1
                    continue

                qc = np.array(quadcell_errors_from_variables(x_candidate, *slice_mirrors), dtype=float)
                qc_max_abs = float(np.max(np.abs(qc)))
                if qc_max_abs > qc_limit:
                    rejected_counts["qc_limit"] += 1
                    continue

                edge_summary = reflection_edge_summary(
                    x_candidate,
                    *slice_mirrors,
                    include_ends=include_edge_ends,
                )
                min_u = float(edge_summary["min_u"])
                max_u = float(edge_summary["max_u"])
                if np.isfinite(min_u) and min_u < float(u_min):
                    rejected_counts["u_bounds"] += 1
                    continue
                if np.isfinite(max_u) and max_u > float(u_max):
                    rejected_counts["u_bounds"] += 1
                    continue

                active_angles = x_candidate[active_axes].astype(float)
                source_qc = np.array(source_point.get("qc", [np.nan, np.nan]), dtype=float)
                closest_edge_margin = float(edge_summary["closest_edge_margin"])
                score = (
                    qc_max_abs,
                    abs(float(actuator_delta)),
                    -closest_edge_margin if np.isfinite(closest_edge_margin) else np.inf,
                )
                candidate = {
                    "angles": np.array(active_angles, dtype=float),
                    "source_angles": np.array(source_point["angles"], dtype=float),
                    "x": x_candidate,
                    "source_x": source_x,
                    "qc": qc,
                    "qc_norm": float(np.linalg.norm(qc)),
                    "qc_max_abs": qc_max_abs,
                    "source_qc": source_qc,
                    "source_qc_max_abs": float(np.max(np.abs(source_qc))),
                    "reflection_count": reflection_count,
                    "source_reflection_count": int(source_point.get(
                        "reflection_count",
                        surface.get("target_reflections", -1),
                    )),
                    "min_u": min_u,
                    "max_u": max_u,
                    "closest_edge_margin": closest_edge_margin,
                    "reflection_u_values": np.array(edge_summary["u_values"], dtype=float),
                    "sweep_value": target_actuator_value,
                    "target_sweep_value": target_actuator_value,
                    "source_sweep_value": source_actuator_value,
                    "sweep_delta": float(actuator_delta),
                    "abs_sweep_delta": abs(float(actuator_delta)),
                    "sweep_actuator": actuator_label,
                    "source_curve_index": int(curve_index),
                    "source_point_index": int(point_index),
                    "offset_radius_deg": abs(float(actuator_delta)),
                    "offset_phi_rad": np.nan,
                    "score": score,
                }
                source_candidates.append(candidate)

            if keep_best_per_source and source_candidates:
                cloud_points.append(min(source_candidates, key=lambda point: point["score"]))
            elif source_candidates:
                cloud_points.extend(source_candidates)

    cloud_points.sort(key=lambda point: point["score"])
    angles = (
        np.array([point["angles"] for point in cloud_points], dtype=float)
        if cloud_points else np.empty((0, len(active_labels)), dtype=float)
    )
    best_point = cloud_points[0] if cloud_points else None
    return {
        "success": len(cloud_points) > 0,
        "failure_reason": None if cloud_points else "No valid one-actuator target landings were found.",
        "points": cloud_points,
        "angles": angles,
        "angle_labels": active_labels,
        "active_actuators": active_labels,
        "source_reflections": int(surface.get("target_reflections", -1)),
        "target_reflections": target_reflections,
        "qc_limit": qc_limit,
        "u_min": float(u_min),
        "u_max": float(u_max),
        "sweep_actuator": actuator_label,
        "curve_indices": [int(index) for index in curve_indices],
        "actuator_half_span_deg": float(abs(actuator_half_span_deg)),
        "actuator_samples": int(actuator_samples),
        "actuator_offsets": None if offsets is None else np.array(offsets, dtype=float),
        "actuator_values": None if actuator_values is None else np.array(actuator_values, dtype=float),
        "checked_count": int(checked_count),
        "kept_count": int(len(cloud_points)),
        "selected_source_point_count": int(selected_source_count),
        "keep_best_per_source": bool(keep_best_per_source),
        "rejected_counts": rejected_counts,
        "best_point": best_point,
    }


def plot_centered_quadcell_angle_curve(curve, color_by="coordinate"):
    """Plot pairwise projections of a traced centered-QC angle curve."""
    points = curve.get("points", [])
    angles = np.array(curve.get("angles", []), dtype=float)
    labels = curve.get("angle_labels", ["M1.dangle", "M2.dangle", "M3.dangle", "M4.dangle"])
    if angles.size == 0 or len(points) == 0:
        raise ValueError("curve contains no points to plot.")

    color_values = None
    color_label = color_by
    if color_by == "coordinate":
        color_values = np.array(curve.get("coordinates", np.arange(len(points))), dtype=float)
        color_label = "curve coordinate (deg)"
    elif color_by in points[0]:
        color_values = np.array([point[color_by] for point in points], dtype=float)

    n_angles = angles.shape[1]
    pairs = list(itertools.combinations(range(n_angles), 2))
    n_plots = max(1, len(pairs))
    n_cols = min(3, n_plots)
    n_rows = int(math.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.6 * n_rows))
    axes = np.array(axes, dtype=object).reshape(-1)
    scatter = None
    coordinates = np.array(curve.get("coordinates", np.arange(len(points))), dtype=float)
    start_index = int(np.argmin(np.abs(coordinates)))
    for ax, (i, j) in zip(axes, pairs):
        ax.plot(angles[:, i], angles[:, j], color="0.75", linewidth=1.0)
        if color_values is None:
            ax.scatter(angles[:, i], angles[:, j], s=18)
        else:
            scatter = ax.scatter(angles[:, i], angles[:, j], c=color_values, s=20, cmap="viridis")
        ax.scatter(angles[start_index, i], angles[start_index, j], marker="*", s=100, color="black", label="start")
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.grid(True, linewidth=0.3)
    for ax in axes[len(pairs):]:
        ax.axis("off")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes.tolist(), shrink=0.85, label=color_label)
    axes[0].legend()
    fig.suptitle(f"Centered-QC angle curve, N_R={curve.get('target_reflections')}", y=0.995)
    fig.tight_layout()
    return fig, axes


def plot_centered_quadcell_angle_curve_3d(curve, axes=None, color_by="coordinate",
                                          marker_size=4, show=True, width="100%",
                                          height=650, renderer=None, axis_ranges=None):
    """Create an interactive 3D plot for a three-angle centered-QC curve."""
    angles = np.array(curve.get("angles", []), dtype=float)
    labels = list(curve.get("angle_labels", []))
    points = curve.get("points", [])
    if angles.size == 0 or len(points) == 0:
        raise ValueError("curve contains no points to plot.")
    if angles.ndim != 2 or angles.shape[1] < 3:
        raise ValueError("curve must contain at least three active angle columns.")

    if axes is None:
        axis_indices = [0, 1, 2]
    else:
        axis_indices = []
        for axis in axes:
            if isinstance(axis, str):
                if axis not in labels:
                    raise ValueError(f"axis {axis!r} is not in curve angle labels {labels}.")
                axis_indices.append(labels.index(axis))
            else:
                axis_indices.append(int(axis))
        if len(axis_indices) != 3:
            raise ValueError("axes must contain exactly three labels or indices.")
    if any(index < 0 or index >= angles.shape[1] for index in axis_indices):
        raise ValueError("axes contains an out-of-range angle index.")

    coordinates = np.array(curve.get("coordinates", np.arange(len(points))), dtype=float)
    if color_by == "coordinate":
        color_values = coordinates
        color_label = "curve coordinate (deg)"
    elif color_by == "index":
        color_values = np.arange(len(points), dtype=float)
        color_label = "point index"
    elif color_by in points[0]:
        color_values = np.array([point[color_by] for point in points], dtype=float)
        color_label = color_by
    else:
        raise ValueError(
            f"color_by must be 'coordinate', 'index', or a point key such as "
            f"{sorted(points[0].keys())}."
        )

    qc = np.array([point["qc"] for point in points], dtype=float)
    customdata = np.column_stack([
        coordinates,
        qc[:, 0],
        qc[:, 1],
        np.array([point["qc_max_abs"] for point in points], dtype=float),
        np.array([point["min_u"] for point in points], dtype=float),
        np.array([point["max_u"] for point in points], dtype=float),
        np.array([point["closest_edge_margin"] for point in points], dtype=float),
        np.array([point["reflection_count"] for point in points], dtype=float),
    ])
    hovertemplate = (
        f"{labels[axis_indices[0]]}: %{{x:.6f}} deg<br>"
        f"{labels[axis_indices[1]]}: %{{y:.6f}} deg<br>"
        f"{labels[axis_indices[2]]}: %{{z:.6f}} deg<br>"
        "coord: %{customdata[0]:.6f} deg<br>"
        "QC: (%{customdata[1]:.4g}, %{customdata[2]:.4g}) mm<br>"
        "QC max: %{customdata[3]:.4g} mm<br>"
        "u range: [%{customdata[4]:.4f}, %{customdata[5]:.4f}]<br>"
        "edge margin: %{customdata[6]:.4f}<br>"
        "N_R: %{customdata[7]:.0f}<extra></extra>"
    )
    x_values = angles[:, axis_indices[0]]
    y_values = angles[:, axis_indices[1]]
    z_values = angles[:, axis_indices[2]]
    start_index = int(np.argmin(np.abs(coordinates)))
    title = f"Centered-QC angle curve, N_R={curve.get('target_reflections')}"

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode="lines+markers",
            marker={
                "size": float(marker_size),
                "color": color_values,
                "colorscale": "Viridis",
                "colorbar": {"title": color_label},
            },
            line={"width": 4, "color": "rgba(90,90,90,0.45)"},
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="centered curve",
        ))
        fig.add_trace(go.Scatter3d(
            x=[x_values[start_index]],
            y=[y_values[start_index]],
            z=[z_values[start_index]],
            mode="markers",
            marker={"size": max(float(marker_size) + 3.0, 7.0), "color": "black", "symbol": "diamond"},
            name="start",
            hovertemplate="start<extra></extra>",
        ))
        fig.update_layout(
            title=title,
            width=None if width == "100%" else width,
            height=int(height),
            scene=_plotly_3d_scene(
                [labels[axis_indices[0]], labels[axis_indices[1]], labels[axis_indices[2]]],
                axis_ranges=axis_ranges,
            ),
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )
        if show:
            if renderer is not None:
                fig.show(renderer=renderer)
            else:
                try:
                    from IPython.display import display
                    display(fig)
                except Exception:
                    fig.show()
        return fig
    except ImportError:
        import json
        import uuid

        root_id = "centered-curve-3d-" + uuid.uuid4().hex
        payload = {
            "data": [
                {
                    "type": "scatter3d",
                    "mode": "lines+markers",
                    "x": x_values.tolist(),
                    "y": y_values.tolist(),
                    "z": z_values.tolist(),
                    "marker": {
                        "size": float(marker_size),
                        "color": color_values.tolist(),
                        "colorscale": "Viridis",
                        "colorbar": {"title": color_label},
                    },
                    "line": {"width": 4, "color": "rgba(90,90,90,0.45)"},
                    "customdata": customdata.tolist(),
                    "hovertemplate": hovertemplate,
                    "name": "centered curve",
                },
                {
                    "type": "scatter3d",
                    "mode": "markers",
                    "x": [float(x_values[start_index])],
                    "y": [float(y_values[start_index])],
                    "z": [float(z_values[start_index])],
                    "marker": {
                        "size": max(float(marker_size) + 3.0, 7.0),
                        "color": "black",
                        "symbol": "diamond",
                    },
                    "name": "start",
                    "hovertemplate": "start<extra></extra>",
                },
            ],
            "layout": {
                "title": title,
                "height": int(height),
                "scene": _plotly_3d_scene(
                    [labels[axis_indices[0]], labels[axis_indices[1]], labels[axis_indices[2]]],
                    axis_ranges=axis_ranges,
                ),
                "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
            },
            "config": {"responsive": True, "displaylogo": False},
        }
        html = f"""
<div id="{root_id}" style="width:{width};height:{int(height)}px;"></div>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>
(function() {{
  const payload = {json.dumps(payload)};
  const root = document.getElementById("{root_id}");
  if (root && window.Plotly) {{
    window.Plotly.newPlot(root, payload.data, payload.layout, payload.config);
  }}
}})();
</script>
"""
        try:
            from IPython.display import HTML, display
            html_obj = HTML(html)
            if show:
                display(html_obj)
            return html_obj
        except Exception:
            if show:
                print("Plotly Python is not installed; returning an HTML fragment instead.")
            return html


def plot_centered_quadcell_angle_curves_3d(curves, labels=None, axes=None,
                                           marker_size=4, show=True, width="100%",
                                           height=650, renderer=None, title=None,
                                           show_starts=True, axis_ranges=None):
    """Create an interactive 3D overlay for multiple centered-QC angle curves."""
    if isinstance(curves, dict):
        curves = [curves]
    curves = list(curves)
    if len(curves) == 0:
        raise ValueError("curves must contain at least one curve.")

    if labels is None:
        labels = [
            f"N_R={curve.get('target_reflections', idx)}"
            for idx, curve in enumerate(curves)
        ]
    labels = list(labels)
    if len(labels) != len(curves):
        raise ValueError("labels must have the same length as curves.")

    base_angle_labels = list(curves[0].get("angle_labels", []))
    if len(base_angle_labels) < 3:
        raise ValueError("curves must contain at least three active angle labels.")
    if axes is None:
        axis_specs = base_angle_labels[:3]
    else:
        axis_specs = list(axes)
        if len(axis_specs) != 3:
            raise ValueError("axes must contain exactly three labels or indices.")

    axis_titles = []
    for axis in axis_specs:
        if isinstance(axis, str):
            axis_titles.append(axis)
        else:
            index = int(axis)
            if index < 0 or index >= len(base_angle_labels):
                raise ValueError("axes contains an out-of-range angle index.")
            axis_titles.append(base_angle_labels[index])

    prepared = []
    for curve, curve_label in zip(curves, labels):
        angles = np.array(curve.get("angles", []), dtype=float)
        points = curve.get("points", [])
        angle_labels = list(curve.get("angle_labels", []))
        if angles.size == 0 or len(points) == 0:
            raise ValueError(f"curve {curve_label!r} contains no points to plot.")
        if angles.ndim != 2 or angles.shape[1] < 3:
            raise ValueError(f"curve {curve_label!r} must contain at least three active angle columns.")

        axis_indices = []
        for axis in axis_specs:
            if isinstance(axis, str):
                if axis not in angle_labels:
                    raise ValueError(
                        f"axis {axis!r} is not in curve {curve_label!r} angle labels {angle_labels}."
                    )
                axis_indices.append(angle_labels.index(axis))
            else:
                index = int(axis)
                if index < 0 or index >= angles.shape[1]:
                    raise ValueError(f"axes contains an out-of-range angle index for curve {curve_label!r}.")
                axis_indices.append(index)

        coordinates = np.array(curve.get("coordinates", np.arange(len(points))), dtype=float)
        qc = np.array([point["qc"] for point in points], dtype=float)
        customdata = np.column_stack([
            coordinates,
            qc[:, 0],
            qc[:, 1],
            np.array([point["qc_max_abs"] for point in points], dtype=float),
            np.array([point["min_u"] for point in points], dtype=float),
            np.array([point["max_u"] for point in points], dtype=float),
            np.array([point["closest_edge_margin"] for point in points], dtype=float),
            np.array([point["reflection_count"] for point in points], dtype=float),
            np.arange(len(points), dtype=float),
        ])
        prepared.append({
            "label": curve_label,
            "angles": angles,
            "axis_indices": axis_indices,
            "coordinates": coordinates,
            "customdata": customdata,
            "start_index": int(np.argmin(np.abs(coordinates))),
            "target_reflections": curve.get("target_reflections"),
        })

    hovertemplate = (
        "%{fullData.name}<br>"
        f"{axis_titles[0]}: %{{x:.6f}} deg<br>"
        f"{axis_titles[1]}: %{{y:.6f}} deg<br>"
        f"{axis_titles[2]}: %{{z:.6f}} deg<br>"
        "coord: %{customdata[0]:.6f} deg<br>"
        "QC: (%{customdata[1]:.4g}, %{customdata[2]:.4g}) mm<br>"
        "QC max: %{customdata[3]:.4g} mm<br>"
        "u range: [%{customdata[4]:.4f}, %{customdata[5]:.4f}]<br>"
        "edge margin: %{customdata[6]:.4f}<br>"
        "N_R: %{customdata[7]:.0f}<br>"
        "point: %{customdata[8]:.0f}<extra></extra>"
    )
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    plot_title = title if title is not None else "Centered-QC angle curves"

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for idx, item in enumerate(prepared):
            color = palette[idx % len(palette)]
            axis_indices = item["axis_indices"]
            angles = item["angles"]
            x_values = angles[:, axis_indices[0]]
            y_values = angles[:, axis_indices[1]]
            z_values = angles[:, axis_indices[2]]
            fig.add_trace(go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="lines+markers",
                marker={"size": float(marker_size), "color": color},
                line={"width": 5, "color": color},
                customdata=item["customdata"],
                hovertemplate=hovertemplate,
                name=item["label"],
            ))
            if show_starts:
                start_index = item["start_index"]
                fig.add_trace(go.Scatter3d(
                    x=[x_values[start_index]],
                    y=[y_values[start_index]],
                    z=[z_values[start_index]],
                    mode="markers",
                    marker={
                        "size": max(float(marker_size) + 3.0, 7.0),
                        "color": color,
                        "symbol": "diamond",
                        "line": {"color": "black", "width": 2},
                    },
                    name=f"{item['label']} start",
                    hovertemplate=f"{item['label']} start<extra></extra>",
                    showlegend=False,
                ))

        fig.update_layout(
            title=plot_title,
            width=None if width == "100%" else width,
            height=int(height),
            scene=_plotly_3d_scene(axis_titles, axis_ranges=axis_ranges),
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            legend={"itemsizing": "constant"},
        )
        if show:
            if renderer is not None:
                fig.show(renderer=renderer)
            else:
                try:
                    from IPython.display import display
                    display(fig)
                except Exception:
                    fig.show()
        return fig
    except ImportError:
        import json
        import uuid

        traces = []
        for idx, item in enumerate(prepared):
            color = palette[idx % len(palette)]
            axis_indices = item["axis_indices"]
            angles = item["angles"]
            x_values = angles[:, axis_indices[0]]
            y_values = angles[:, axis_indices[1]]
            z_values = angles[:, axis_indices[2]]
            traces.append({
                "type": "scatter3d",
                "mode": "lines+markers",
                "x": x_values.tolist(),
                "y": y_values.tolist(),
                "z": z_values.tolist(),
                "marker": {"size": float(marker_size), "color": color},
                "line": {"width": 5, "color": color},
                "customdata": item["customdata"].tolist(),
                "hovertemplate": hovertemplate,
                "name": item["label"],
            })
            if show_starts:
                start_index = item["start_index"]
                traces.append({
                    "type": "scatter3d",
                    "mode": "markers",
                    "x": [float(x_values[start_index])],
                    "y": [float(y_values[start_index])],
                    "z": [float(z_values[start_index])],
                    "marker": {
                        "size": max(float(marker_size) + 3.0, 7.0),
                        "color": color,
                        "symbol": "diamond",
                        "line": {"color": "black", "width": 2},
                    },
                    "name": f"{item['label']} start",
                    "hovertemplate": f"{item['label']} start<extra></extra>",
                    "showlegend": False,
                })

        root_id = "centered-curves-3d-" + uuid.uuid4().hex
        payload = {
            "data": traces,
            "layout": {
                "title": plot_title,
                "height": int(height),
                "scene": _plotly_3d_scene(axis_titles, axis_ranges=axis_ranges),
                "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
                "legend": {"itemsizing": "constant"},
            },
            "config": {"responsive": True, "displaylogo": False},
        }
        html = f"""
<div id="{root_id}" style="width:{width};height:{int(height)}px;"></div>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>
(function() {{
  const payload = {json.dumps(payload)};
  const root = document.getElementById("{root_id}");
  if (root && window.Plotly) {{
    window.Plotly.newPlot(root, payload.data, payload.layout, payload.config);
  }}
}})();
</script>
"""
        try:
            from IPython.display import HTML, display
            html_obj = HTML(html)
            if show:
                display(html_obj)
            return html_obj
        except Exception:
            if show:
                print("Plotly Python is not installed; returning an HTML fragment instead.")
            return html


def _rgba_from_colormap_value(value, vmin, vmax, cmap_name="viridis", alpha=1.0):
    if not np.isfinite(value):
        norm_value = 0.5
    elif vmax <= vmin:
        norm_value = 0.5
    else:
        norm_value = (float(value) - float(vmin)) / (float(vmax) - float(vmin))
        norm_value = min(1.0, max(0.0, norm_value))
    rgba = plt.get_cmap(cmap_name)(norm_value)
    return (
        f"rgba({int(round(255 * rgba[0]))},"
        f"{int(round(255 * rgba[1]))},"
        f"{int(round(255 * rgba[2]))},{float(alpha):.3f})"
    )


def _normalize_3d_axis_ranges(axis_ranges, axis_titles):
    if axis_ranges is None:
        return [None, None, None]

    def clean_range(value):
        if value is None:
            return None
        values = list(value)
        if len(values) != 2:
            raise ValueError("Each axis range must contain exactly [min, max].")
        return [float(values[0]), float(values[1])]

    if isinstance(axis_ranges, dict):
        axis_letters = ["x", "y", "z"]
        normalized = []
        for idx, title in enumerate(axis_titles):
            value = None
            for key in (title, axis_letters[idx], idx, str(idx)):
                if key in axis_ranges:
                    value = axis_ranges[key]
                    break
            normalized.append(clean_range(value))
        return normalized

    values = list(axis_ranges)
    if len(values) != 3:
        raise ValueError("axis_ranges must be a dict or a three-item sequence.")
    return [clean_range(value) for value in values]


def _plotly_3d_scene(axis_titles, axis_ranges=None):
    ranges = _normalize_3d_axis_ranges(axis_ranges, axis_titles)
    scene = {}
    for axis_key, axis_title, axis_range in zip(
            ["xaxis", "yaxis", "zaxis"], axis_titles, ranges):
        axis_config = {"title": axis_title}
        if axis_range is not None:
            axis_config["range"] = axis_range
        scene[axis_key] = axis_config
    return scene


def plot_centered_quadcell_angle_surface_3d(surface, label=None, axes=None,
                                            marker_size=3, show=True, width="100%",
                                            height=650, renderer=None, title=None,
                                            opacity=0.9, show_start_markers=False,
                                            show_reference_markers=True,
                                            reference_marker_size=9,
                                            axis_ranges=None,
                                            clouds=None,
                                            cloud_labels=None,
                                            cloud_marker_size=2,
                                            cloud_opacity=0.35,
                                            tubes=None,
                                            tube_labels=None,
                                            tube_opacity=0.18,
                                            tube_color="rgba(35, 170, 255, 0.55)"):
    """Interactive 3D plot of a fixed-actuator slice surface."""
    labels = None if label is None else [label]
    return plot_centered_quadcell_angle_surfaces_3d(
        [surface],
        labels=labels,
        axes=axes,
        marker_size=marker_size,
        show=show,
        width=width,
        height=height,
        renderer=renderer,
        title=title,
        opacity=opacity,
        show_start_markers=show_start_markers,
        show_reference_markers=show_reference_markers,
        reference_marker_size=reference_marker_size,
        axis_ranges=axis_ranges,
        clouds=clouds,
        cloud_labels=cloud_labels,
        cloud_marker_size=cloud_marker_size,
        cloud_opacity=cloud_opacity,
        tubes=tubes,
        tube_labels=tube_labels,
        tube_opacity=tube_opacity,
        tube_color=tube_color,
    )


def plot_centered_quadcell_angle_surfaces_3d(surfaces, labels=None, axes=None,
                                             marker_size=3, show=True, width="100%",
                                             height=650, renderer=None, title=None,
                                             opacity=0.9, show_start_markers=False,
                                             show_reference_markers=True,
                                             reference_marker_size=9,
                                             axis_ranges=None,
                                             clouds=None,
                                             cloud_labels=None,
                                             cloud_marker_size=2,
                                             cloud_opacity=0.35,
                                             tubes=None,
                                             tube_labels=None,
                                             tube_opacity=0.18,
                                             tube_color="rgba(35, 170, 255, 0.55)"):
    """Overlay centered-QC slice surfaces in 3D, using color for the swept angle."""
    if isinstance(surfaces, dict):
        surfaces = [surfaces]
    surfaces = list(surfaces)
    if len(surfaces) == 0:
        raise ValueError("surfaces must contain at least one surface.")

    if labels is None:
        labels = [
            f"N_R={surface.get('target_reflections', idx)}"
            for idx, surface in enumerate(surfaces)
        ]
    labels = list(labels)
    if len(labels) != len(surfaces):
        raise ValueError("labels must have the same length as surfaces.")

    if clouds is None:
        clouds = []
    elif isinstance(clouds, dict):
        clouds = [clouds]
    else:
        clouds = list(clouds)
    if cloud_labels is None:
        cloud_labels = [
            f"N_R={cloud.get('target_reflections', idx)} QC cloud"
            for idx, cloud in enumerate(clouds)
        ]
    cloud_labels = list(cloud_labels)
    if len(cloud_labels) != len(clouds):
        raise ValueError("cloud_labels must have the same length as clouds.")

    if tubes is None:
        tubes = []
    elif isinstance(tubes, dict):
        tubes = [tubes]
    else:
        tubes = list(tubes)
    if tube_labels is None:
        tube_labels = [
            f"N_R={tube.get('target_reflections', idx)} QC tube"
            for idx, tube in enumerate(tubes)
        ]
    tube_labels = list(tube_labels)
    if len(tube_labels) != len(tubes):
        raise ValueError("tube_labels must have the same length as tubes.")

    nonempty_surfaces = [
        surface for surface in surfaces
        if len(surface.get("curves", [])) > 0
    ]
    if len(nonempty_surfaces) == 0:
        raise ValueError("surfaces contain no successfully traced slices to plot.")

    first_curve = nonempty_surfaces[0]["curves"][0]
    base_angle_labels = list(first_curve.get("angle_labels", []))
    if len(base_angle_labels) < 3:
        raise ValueError("surface curves must contain at least three active angle labels.")

    if axes is None:
        axis_specs = base_angle_labels[:3]
    else:
        axis_specs = list(axes)
        if len(axis_specs) != 3:
            raise ValueError("axes must contain exactly three labels or indices.")

    axis_titles = []
    for axis in axis_specs:
        if isinstance(axis, str):
            axis_titles.append(axis)
        else:
            index = int(axis)
            if index < 0 or index >= len(base_angle_labels):
                raise ValueError("axes contains an out-of-range angle index.")
            axis_titles.append(base_angle_labels[index])

    all_sweep_values = []
    for surface in nonempty_surfaces:
        for curve in surface.get("curves", []):
            all_sweep_values.append(float(curve.get("sweep_value", np.nan)))
    for cloud in clouds:
        for point in cloud.get("points", []):
            all_sweep_values.append(float(point.get("sweep_value", np.nan)))
    for tube in tubes:
        all_sweep_values.append(float(tube.get("sweep_value", np.nan)))
    all_sweep_values = np.array(all_sweep_values, dtype=float)
    finite_sweep = all_sweep_values[np.isfinite(all_sweep_values)]
    if finite_sweep.size == 0:
        cmin, cmax = 0.0, 1.0
    else:
        cmin, cmax = float(np.min(finite_sweep)), float(np.max(finite_sweep))
        if cmax <= cmin:
            cmin -= 0.5
            cmax += 0.5

    sweep_actuators = {
        str(surface.get("sweep_actuator", "sweep value"))
        for surface in nonempty_surfaces
    }
    colorbar_title = (
        next(iter(sweep_actuators)) + " (deg)"
        if len(sweep_actuators) == 1
        else "swept angle (deg)"
    )
    symbols = ["circle", "diamond", "square", "cross", "x"]
    plot_title = title if title is not None else "Centered-QC angle surface slices"

    prepared = []
    for surface_idx, (surface, surface_label) in enumerate(zip(surfaces, labels)):
        for curve_idx, curve in enumerate(surface.get("curves", [])):
            angles = np.array(curve.get("angles", []), dtype=float)
            points = curve.get("points", [])
            angle_labels = list(curve.get("angle_labels", []))
            if angles.size == 0 or len(points) == 0:
                continue
            axis_indices = []
            for axis in axis_specs:
                if isinstance(axis, str):
                    if axis not in angle_labels:
                        raise ValueError(
                            f"axis {axis!r} is not in curve angle labels {angle_labels}."
                        )
                    axis_indices.append(angle_labels.index(axis))
                else:
                    index = int(axis)
                    if index < 0 or index >= angles.shape[1]:
                        raise ValueError("axes contains an out-of-range angle index for a curve.")
                    axis_indices.append(index)

            coordinates = np.array(curve.get("coordinates", np.arange(len(points))), dtype=float)
            sweep_value = float(curve.get("sweep_value", np.nan))
            point_customdata = []
            for point_idx, point in enumerate(points):
                qc = np.array(point["qc"], dtype=float)
                point_customdata.append([
                    float(coordinates[point_idx]),
                    float(qc[0]),
                    float(qc[1]),
                    float(point["qc_max_abs"]),
                    float(point["min_u"]),
                    float(point["max_u"]),
                    float(point["closest_edge_margin"]),
                    float(point["reflection_count"]),
                    float(point_idx),
                    sweep_value,
                    surface_label,
                ])
            prepared.append({
                "surface_index": surface_idx,
                "surface_label": surface_label,
                "curve_index": curve_idx,
                "curve": curve,
                "angles": angles,
                "axis_indices": axis_indices,
                "coordinates": coordinates,
                "sweep_value": sweep_value,
                "customdata": point_customdata,
                "symbol": symbols[surface_idx % len(symbols)],
                "start_index": int(np.argmin(np.abs(coordinates))) if coordinates.size else 0,
            })

    if len(prepared) == 0:
        raise ValueError("surfaces contain no successfully traced slices to plot.")

    prepared_clouds = []
    for cloud_idx, (cloud, cloud_label) in enumerate(zip(clouds, cloud_labels)):
        points = list(cloud.get("points", []))
        if len(points) == 0:
            continue
        angle_labels = list(cloud.get("angle_labels", []))
        _, axis_indices = _axis_indices_for_labels(angle_labels, axis_titles)
        angles = np.array([point["angles"] for point in points], dtype=float)
        x_values = angles[:, axis_indices[0]]
        y_values = angles[:, axis_indices[1]]
        z_values = angles[:, axis_indices[2]]
        customdata = []
        sweep_values = []
        for point_idx, point in enumerate(points):
            qc = np.array(point.get("qc", [np.nan, np.nan]), dtype=float)
            sweep_value = float(point.get("sweep_value", np.nan))
            sweep_values.append(sweep_value)
            customdata.append([
                cloud_label,
                sweep_value,
                float(qc[0]),
                float(qc[1]),
                float(point.get("qc_max_abs", np.nan)),
                float(point.get("min_u", np.nan)),
                float(point.get("max_u", np.nan)),
                float(point.get("closest_edge_margin", np.nan)),
                float(point.get("reflection_count", np.nan)),
                float(point_idx),
                float(point.get("offset_radius_deg", np.nan)),
                float(point.get("source_curve_index", np.nan)),
                float(point.get("source_point_index", np.nan)),
                float(point.get("source_sweep_value", np.nan)),
                float(point.get("target_sweep_value", sweep_value)),
                float(point.get("sweep_delta", np.nan)),
                float(point.get("abs_sweep_delta", np.nan)),
                float(point.get("source_qc_max_abs", np.nan)),
            ])
        prepared_clouds.append({
            "label": cloud_label,
            "x": x_values,
            "y": y_values,
            "z": z_values,
            "sweep_values": np.array(sweep_values, dtype=float),
            "customdata": customdata,
        })

    prepared_tubes = []
    for tube_idx, (tube, tube_label) in enumerate(zip(tubes, tube_labels)):
        vertices = np.array(tube.get("vertices", []), dtype=float)
        if vertices.size == 0:
            continue
        angle_labels = list(tube.get("angle_labels", []))
        _, axis_indices = _axis_indices_for_labels(angle_labels, axis_titles)
        projected_vertices = vertices[:, axis_indices]
        faces_i = np.array(tube.get("faces_i", []), dtype=int)
        faces_j = np.array(tube.get("faces_j", []), dtype=int)
        faces_k = np.array(tube.get("faces_k", []), dtype=int)
        if faces_i.size == 0:
            continue
        prepared_tubes.append({
            "label": tube_label,
            "x": projected_vertices[:, 0],
            "y": projected_vertices[:, 1],
            "z": projected_vertices[:, 2],
            "i": faces_i,
            "j": faces_j,
            "k": faces_k,
            "customdata": np.array(tube.get("customdata", []), dtype=float),
        })

    reference_items = []
    if show_reference_markers:
        for surface_idx, (surface, surface_label) in enumerate(zip(surfaces, labels)):
            base_point = surface.get("base_point")
            if base_point is None:
                continue
            angle_labels = list(surface.get("angle_labels", []))
            if len(angle_labels) < 3:
                continue
            axis_indices = []
            for axis in axis_specs:
                if isinstance(axis, str):
                    if axis not in angle_labels:
                        raise ValueError(
                            f"axis {axis!r} is not in surface angle labels {angle_labels}."
                        )
                    axis_indices.append(angle_labels.index(axis))
                else:
                    index = int(axis)
                    if index < 0 or index >= len(angle_labels):
                        raise ValueError("axes contains an out-of-range angle index for a surface base point.")
                    axis_indices.append(index)

            base_angles = np.array(base_point.get("angles", []), dtype=float)
            if base_angles.size < max(axis_indices) + 1:
                continue
            base_qc = np.array(base_point.get("qc", [np.nan, np.nan]), dtype=float)
            reference_items.append({
                "surface_index": int(surface_idx),
                "surface_label": surface_label,
                "x": float(base_angles[axis_indices[0]]),
                "y": float(base_angles[axis_indices[1]]),
                "z": float(base_angles[axis_indices[2]]),
                "customdata": [[
                    surface_label,
                    float(base_point.get("sweep_value", np.nan)),
                    float(base_qc[0]),
                    float(base_qc[1]),
                    float(base_point.get("qc_max_abs", np.nan)),
                    float(base_point.get("min_u", np.nan)),
                    float(base_point.get("max_u", np.nan)),
                    float(base_point.get("closest_edge_margin", np.nan)),
                    float(base_point.get("reflection_count", np.nan)),
                ]],
            })

    hovertemplate = (
        "%{customdata[10]}<br>"
        f"{axis_titles[0]}: %{{x:.6f}} deg<br>"
        f"{axis_titles[1]}: %{{y:.6f}} deg<br>"
        f"{axis_titles[2]}: %{{z:.6f}} deg<br>"
        "sweep: %{customdata[9]:.6f} deg<br>"
        "coord: %{customdata[0]:.6f} deg<br>"
        "QC: (%{customdata[1]:.4g}, %{customdata[2]:.4g}) mm<br>"
        "QC max: %{customdata[3]:.4g} mm<br>"
        "u range: [%{customdata[4]:.4f}, %{customdata[5]:.4f}]<br>"
        "edge margin: %{customdata[6]:.4f}<br>"
        "N_R: %{customdata[7]:.0f}<br>"
        "point: %{customdata[8]:.0f}<extra></extra>"
    )
    reference_hovertemplate = (
        "%{customdata[0]} base<br>"
        f"{axis_titles[0]}: %{{x:.6f}} deg<br>"
        f"{axis_titles[1]}: %{{y:.6f}} deg<br>"
        f"{axis_titles[2]}: %{{z:.6f}} deg<br>"
        "sweep: %{customdata[1]:.6f} deg<br>"
        "QC: (%{customdata[2]:.4g}, %{customdata[3]:.4g}) mm<br>"
        "QC max: %{customdata[4]:.4g} mm<br>"
        "u range: [%{customdata[5]:.4f}, %{customdata[6]:.4f}]<br>"
        "edge margin: %{customdata[7]:.4f}<br>"
        "N_R: %{customdata[8]:.0f}<extra></extra>"
    )
    cloud_hovertemplate = (
        "%{customdata[0]}<br>"
        f"{axis_titles[0]}: %{{x:.6f}} deg<br>"
        f"{axis_titles[1]}: %{{y:.6f}} deg<br>"
        f"{axis_titles[2]}: %{{z:.6f}} deg<br>"
        "sweep: %{customdata[1]:.6f} deg<br>"
        "QC: (%{customdata[2]:.4g}, %{customdata[3]:.4g}) mm<br>"
        "QC max: %{customdata[4]:.4g} mm<br>"
        "u range: [%{customdata[5]:.4f}, %{customdata[6]:.4f}]<br>"
        "edge margin: %{customdata[7]:.4f}<br>"
        "N_R: %{customdata[8]:.0f}<br>"
        "cloud point: %{customdata[9]:.0f}<br>"
        "offset radius: %{customdata[10]:.4f} deg<br>"
        "source: curve %{customdata[11]:.0f}, point %{customdata[12]:.0f}<br>"
        "source sweep: %{customdata[13]:.6f} deg<br>"
        "target sweep: %{customdata[14]:.6f} deg<br>"
        "sweep delta: %{customdata[15]:.6f} deg<br>"
        "source QC max: %{customdata[17]:.4g} mm<extra></extra>"
    )
    tube_hovertemplate = (
        "%{fullData.name}<br>"
        f"{axis_titles[0]}: %{{x:.6f}} deg<br>"
        f"{axis_titles[1]}: %{{y:.6f}} deg<br>"
        f"{axis_titles[2]}: %{{z:.6f}} deg<br>"
        "ring: %{customdata[0]:.0f}<br>"
        "source point: %{customdata[2]:.0f}<br>"
        "radius: %{customdata[3]:.4f} deg<br>"
        "QC: (%{customdata[4]:.4g}, %{customdata[5]:.4g}) mm<br>"
        "QC max: %{customdata[6]:.4g} mm<br>"
        "u range: [%{customdata[7]:.4f}, %{customdata[8]:.4f}]<br>"
        "edge margin: %{customdata[9]:.4f}<extra></extra>"
    )

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for surface_idx, surface_label in enumerate(labels):
            if any(item["surface_index"] == surface_idx for item in prepared):
                fig.add_trace(go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker={
                        "size": max(float(marker_size) + 3.0, 7.0),
                        "symbol": symbols[surface_idx % len(symbols)],
                        "color": "rgba(90,90,90,0.85)",
                    },
                    name=surface_label,
                    showlegend=True,
                    hoverinfo="skip",
                ))

        for item in prepared:
            angles = item["angles"]
            axis_indices = item["axis_indices"]
            x_values = angles[:, axis_indices[0]]
            y_values = angles[:, axis_indices[1]]
            z_values = angles[:, axis_indices[2]]
            sweep_value = item["sweep_value"]
            line_color = _rgba_from_colormap_value(sweep_value, cmin, cmax, alpha=opacity)
            fig.add_trace(go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="lines+markers",
                marker={
                    "size": float(marker_size),
                    "symbol": item["symbol"],
                    "color": [sweep_value] * len(x_values),
                    "coloraxis": "coloraxis",
                },
                line={"width": 4, "color": line_color},
                customdata=item["customdata"],
                hovertemplate=hovertemplate,
                name=f"{item['surface_label']} {sweep_value:.6f}",
                showlegend=False,
                opacity=float(opacity),
            ))
            if show_start_markers:
                start_index = item["start_index"]
                fig.add_trace(go.Scatter3d(
                    x=[x_values[start_index]],
                    y=[y_values[start_index]],
                    z=[z_values[start_index]],
                    mode="markers",
                    marker={
                        "size": max(float(marker_size) + 3.0, 7.0),
                        "symbol": item["symbol"],
                        "color": [sweep_value],
                        "coloraxis": "coloraxis",
                        "line": {"color": "black", "width": 2},
                    },
                    name=f"{item['surface_label']} start",
                    hovertemplate=f"{item['surface_label']} slice start<extra></extra>",
                    showlegend=False,
                ))

        for item in reference_items:
            fig.add_trace(go.Scatter3d(
                x=[item["x"]],
                y=[item["y"]],
                z=[item["z"]],
                mode="markers",
                marker={
                    "size": float(reference_marker_size),
                    "color": "black",
                    "symbol": "diamond",
                    "line": {"color": "white", "width": 2},
                },
                customdata=item["customdata"],
                hovertemplate=reference_hovertemplate,
                name=f"{item['surface_label']} base",
                showlegend=True,
            ))

        for item in prepared_tubes:
            fig.add_trace(go.Mesh3d(
                x=item["x"],
                y=item["y"],
                z=item["z"],
                i=item["i"],
                j=item["j"],
                k=item["k"],
                color=tube_color,
                opacity=float(tube_opacity),
                flatshading=False,
                customdata=item["customdata"],
                hovertemplate=tube_hovertemplate,
                name=item["label"],
                showlegend=True,
            ))

        for item in prepared_clouds:
            fig.add_trace(go.Scatter3d(
                x=item["x"],
                y=item["y"],
                z=item["z"],
                mode="markers",
                marker={
                    "size": float(cloud_marker_size),
                    "color": item["sweep_values"],
                    "coloraxis": "coloraxis",
                    "symbol": "circle",
                },
                opacity=float(cloud_opacity),
                customdata=item["customdata"],
                hovertemplate=cloud_hovertemplate,
                name=item["label"],
                showlegend=True,
            ))

        fig.update_layout(
            title=plot_title,
            width=None if width == "100%" else width,
            height=int(height),
            scene=_plotly_3d_scene(axis_titles, axis_ranges=axis_ranges),
            coloraxis={
                "colorscale": "Viridis",
                "cmin": cmin,
                "cmax": cmax,
                "colorbar": {"title": colorbar_title},
            },
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            legend={"itemsizing": "constant"},
        )
        if show:
            if renderer is not None:
                fig.show(renderer=renderer)
            else:
                try:
                    from IPython.display import display
                    display(fig)
                except Exception:
                    fig.show()
        return fig
    except ImportError:
        import json
        import uuid

        traces = []
        for surface_idx, surface_label in enumerate(labels):
            if any(item["surface_index"] == surface_idx for item in prepared):
                traces.append({
                    "type": "scatter3d",
                    "mode": "markers",
                    "x": [None],
                    "y": [None],
                    "z": [None],
                    "marker": {
                        "size": max(float(marker_size) + 3.0, 7.0),
                        "symbol": symbols[surface_idx % len(symbols)],
                        "color": "rgba(90,90,90,0.85)",
                    },
                    "name": surface_label,
                    "showlegend": True,
                    "hoverinfo": "skip",
                })

        for item in prepared:
            angles = item["angles"]
            axis_indices = item["axis_indices"]
            x_values = angles[:, axis_indices[0]]
            y_values = angles[:, axis_indices[1]]
            z_values = angles[:, axis_indices[2]]
            sweep_value = item["sweep_value"]
            line_color = _rgba_from_colormap_value(sweep_value, cmin, cmax, alpha=opacity)
            traces.append({
                "type": "scatter3d",
                "mode": "lines+markers",
                "x": x_values.tolist(),
                "y": y_values.tolist(),
                "z": z_values.tolist(),
                "marker": {
                    "size": float(marker_size),
                    "symbol": item["symbol"],
                    "color": [sweep_value] * len(x_values),
                    "coloraxis": "coloraxis",
                },
                "line": {"width": 4, "color": line_color},
                "customdata": item["customdata"],
                "hovertemplate": hovertemplate,
                "name": f"{item['surface_label']} {sweep_value:.6f}",
                "showlegend": False,
                "opacity": float(opacity),
            })
            if show_start_markers:
                start_index = item["start_index"]
                traces.append({
                    "type": "scatter3d",
                    "mode": "markers",
                    "x": [float(x_values[start_index])],
                    "y": [float(y_values[start_index])],
                    "z": [float(z_values[start_index])],
                    "marker": {
                        "size": max(float(marker_size) + 3.0, 7.0),
                        "symbol": item["symbol"],
                        "color": [sweep_value],
                        "coloraxis": "coloraxis",
                        "line": {"color": "black", "width": 2},
                    },
                    "name": f"{item['surface_label']} start",
                    "hovertemplate": f"{item['surface_label']} slice start<extra></extra>",
                    "showlegend": False,
                })

        for item in reference_items:
            traces.append({
                "type": "scatter3d",
                "mode": "markers",
                "x": [item["x"]],
                "y": [item["y"]],
                "z": [item["z"]],
                "marker": {
                    "size": float(reference_marker_size),
                    "color": "black",
                    "symbol": "diamond",
                    "line": {"color": "white", "width": 2},
                },
                "customdata": item["customdata"],
                "hovertemplate": reference_hovertemplate,
                "name": f"{item['surface_label']} base",
                "showlegend": True,
            })

        for item in prepared_tubes:
            traces.append({
                "type": "mesh3d",
                "x": item["x"].tolist(),
                "y": item["y"].tolist(),
                "z": item["z"].tolist(),
                "i": item["i"].tolist(),
                "j": item["j"].tolist(),
                "k": item["k"].tolist(),
                "color": tube_color,
                "opacity": float(tube_opacity),
                "flatshading": False,
                "customdata": item["customdata"].tolist(),
                "hovertemplate": tube_hovertemplate,
                "name": item["label"],
                "showlegend": True,
            })

        for item in prepared_clouds:
            traces.append({
                "type": "scatter3d",
                "mode": "markers",
                "x": item["x"].tolist(),
                "y": item["y"].tolist(),
                "z": item["z"].tolist(),
                "marker": {
                    "size": float(cloud_marker_size),
                    "color": item["sweep_values"].tolist(),
                    "coloraxis": "coloraxis",
                    "symbol": "circle",
                },
                "opacity": float(cloud_opacity),
                "customdata": item["customdata"],
                "hovertemplate": cloud_hovertemplate,
                "name": item["label"],
                "showlegend": True,
            })

        root_id = "centered-surfaces-3d-" + uuid.uuid4().hex
        payload = {
            "data": traces,
            "layout": {
                "title": plot_title,
                "height": int(height),
                "scene": _plotly_3d_scene(axis_titles, axis_ranges=axis_ranges),
                "coloraxis": {
                    "colorscale": "Viridis",
                    "cmin": cmin,
                    "cmax": cmax,
                    "colorbar": {"title": colorbar_title},
                },
                "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
                "legend": {"itemsizing": "constant"},
            },
            "config": {"responsive": True, "displaylogo": False},
        }
        html = f"""
<div id="{root_id}" style="width:{width};height:{int(height)}px;"></div>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>
(function() {{
  const payload = {json.dumps(payload)};
  const root = document.getElementById("{root_id}");
  if (root && window.Plotly) {{
    window.Plotly.newPlot(root, payload.data, payload.layout, payload.config);
  }}
}})();
</script>
"""
        try:
            from IPython.display import HTML, display
            html_obj = HTML(html)
            if show:
                display(html_obj)
            return html_obj
        except Exception:
            if show:
                print("Plotly Python is not installed; returning an HTML fragment instead.")
            return html


def plot_centered_quadcell_curve_diagnostics(curve):
    """Plot QC and reflection-u diagnostics along a centered angle curve."""
    points = curve.get("points", [])
    if not points:
        raise ValueError("curve contains no points to plot.")

    coordinate = np.array([point["coordinate"] for point in points], dtype=float)
    qc1 = np.array([point["qc"][0] for point in points], dtype=float)
    qc2 = np.array([point["qc"][1] for point in points], dtype=float)
    min_u = np.array([point["min_u"] for point in points], dtype=float)
    max_u = np.array([point["max_u"] for point in points], dtype=float)
    margins = np.array([point["closest_edge_margin"] for point in points], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_qc, ax_u = axes
    ax_qc.plot(coordinate, qc1, marker=".", label="QC1")
    ax_qc.plot(coordinate, qc2, marker=".", label="QC2")
    qc_tolerance = curve.get("qc_tolerance")
    if qc_tolerance is not None:
        ax_qc.axhline(float(qc_tolerance), color="black", linestyle=":", linewidth=1)
        ax_qc.axhline(-float(qc_tolerance), color="black", linestyle=":", linewidth=1)
    ax_qc.set_xlabel("curve coordinate (deg)")
    ax_qc.set_ylabel("QC offset (mm)")
    ax_qc.set_title("Quadcell Centering Along Curve")
    ax_qc.grid(True, linewidth=0.3)
    ax_qc.legend()

    ax_u.plot(coordinate, min_u, marker=".", label="min u")
    ax_u.plot(coordinate, max_u, marker=".", label="max u")
    ax_u.plot(coordinate, margins, marker=".", linestyle="--", label="closest edge margin")
    ax_u.axhline(float(curve.get("u_min", 0.1)), color="black", linestyle=":", linewidth=1)
    ax_u.axhline(float(curve.get("u_max", 0.9)), color="black", linestyle=":", linewidth=1)
    ax_u.set_xlabel("curve coordinate (deg)")
    ax_u.set_title("Reflection Positions Along Curve")
    ax_u.grid(True, linewidth=0.3)
    ax_u.legend()

    fig.tight_layout()
    return fig, axes


def plan_reflection_count_change(M1, M2, M3, M4,
                                 target_N_R,
                                 n_tries=2000,
                                 angle_perturb=0.3,
                                 seed=0,
                                 u_min=0.1,
                                 u_max=0.9,
                                 sigma_edge=0.1,
                                 final_qc_tolerance=0.25,
                                 min_angle_step=1e-12):
    """Plan a rotation-only move to a centered configuration with target_N_R.

    Quadcell path constraints are intentionally not enforced here because the
    exiting beam may leave/re-enter the detectors discontinuously when the
    reflection topology changes.
    """
    M_start = (
        np.array(M1, dtype=float),
        np.array(M2, dtype=float),
        np.array(M3, dtype=float),
        np.array(M4, dtype=float),
    )
    x_start = pack_variables(*M_start)
    start_reflections = get_reflection_count(*M_start)

    M_target, res = center_quadcells(
        *M_start,
        N_R=target_N_R,
        n_tries=n_tries,
        angle_perturb=angle_perturb,
        seed=seed,
        u_min=u_min,
        u_max=u_max,
        sigma_edge=sigma_edge,
        final_qc_tolerance=final_qc_tolerance,
    )
    x_target = pack_variables(*M_target)
    target_reflections = get_reflection_count(*M_target)

    steps = []
    x_current = x_start.copy()
    angle_axes = [1, 3, 5, 7]
    total_angle_motion = float(np.linalg.norm((x_target - x_start)[angle_axes]))
    cumulative_angle_motion = 0.0

    for axis_index in angle_axes:
        amount = float(x_target[axis_index] - x_current[axis_index])
        if abs(amount) <= min_angle_step:
            continue

        x_next = variables_with_axis_move(x_current, axis_index, amount)
        cumulative_angle_motion += abs(amount)
        fraction = (
            1.0 if total_angle_motion <= 1e-12
            else min(1.0, cumulative_angle_motion / total_angle_motion)
        )
        step = make_actuation_step(
            len(steps) + 1,
            fraction,
            x_current,
            x_next,
            *M_start,
            max_qc_error=1e12,
            max_qc_difference=None,
            motion_samples_per_step=1,
            u_min=u_min,
            u_max=u_max,
            include_edge_ends=False,
            enforce_edge_bounds=False,
            constraint_tolerance=0.0,
        )
        step["reflection_count_change_move"] = True
        step["target_N_R"] = int(target_N_R)
        step["qc_path_unconstrained"] = True
        steps.append(step)
        x_current = x_next

    final_qc1_error, final_qc2_error = quadcell_errors_from_variables(x_target, *M_start)
    final_edge_summary = reflection_edge_summary(x_target, *M_start, include_ends=False)

    plan = {
        "steps": steps,
        "n_steps": len(steps),
        "reflection_count_change": True,
        "rotation_only": True,
        "qc_path_unconstrained": True,
        "start_reflections": int(start_reflections),
        "target_reflections": int(target_reflections),
        "target_N_R": int(target_N_R),
        "reflection_count_verified": False,
        "start_mirrors": M_start,
        "target_mirrors": M_target,
        "start_x": x_start,
        "target_x": x_target,
        "final_OPD": float(OPD_from_variables(x_target, *M_start)),
        "final_qc1_error": float(final_qc1_error),
        "final_qc2_error": float(final_qc2_error),
        "final_qc_difference": float(final_qc1_error - final_qc2_error),
        "final_qc_max_abs": float(max(abs(final_qc1_error), abs(final_qc2_error))),
        "final_qc_tolerance": None if final_qc_tolerance is None else float(final_qc_tolerance),
        "selection_mode": getattr(res, "selection_mode", None),
        "final_angle_change_total_abs": getattr(res, "final_angle_change_total_abs", None),
        "center_matching_start_count": getattr(res, "matching_start_count", None),
        "center_valid_solution_count": getattr(res, "valid_solution_count", None),
        "centered_solution_count": getattr(res, "centered_solution_count", None),
        "u_min": float(u_min),
        "u_max": float(u_max),
        "min_reflection_u": float(final_edge_summary["min_u"]),
        "max_reflection_u": float(final_edge_summary["max_u"]),
        "reflection_u_values": final_edge_summary["u_values"],
        "failure_reason": None,
    }

    return M_target, res, plan


def plan_reflection_count_reacquisition(M1, M2, M3, M4,
                                        target_N_R,
                                        qc_reacquire_limit=2.0,
                                        angle_scan_limit=1.0,
                                        scan_samples=1001,
                                        max_first_leg_candidates=50,
                                        min_angle_step=1e-12):
    """Find a short rotation-only plan that gets target_N_R back on the QCs.

    This is intentionally not a fully centered endpoint solve. It searches for
    the smallest one-actuator move that reaches target_N_R with both quadcells
    inside qc_reacquire_limit. If that fails, it tries a two-actuator sequence
    where the first move reaches target_N_R with at least one QC in range, then
    the second move gets both QCs in range.
    """
    M_start = (
        np.array(M1, dtype=float),
        np.array(M2, dtype=float),
        np.array(M3, dtype=float),
        np.array(M4, dtype=float),
    )
    x_start = pack_variables(*M_start)
    target_N_R = int(target_N_R)
    qc_reacquire_limit = float(qc_reacquire_limit)
    angle_scan_limit = float(abs(angle_scan_limit))
    scan_samples = max(3, int(scan_samples))
    max_first_leg_candidates = max(1, int(max_first_leg_candidates))

    angle_axes = [
        ("M1.dangle", 1),
        ("M2.dangle", 3),
        ("M3.dangle", 5),
        ("M4.dangle", 7),
    ]
    scan_amounts = np.linspace(0.0, angle_scan_limit, scan_samples)[1:]
    start_reflections = get_reflection_count(*M_start)
    start_qc1, start_qc2 = quadcell_errors_from_variables(x_start, *M_start)

    def metrics(x):
        mirrors = unpack_variables(x, *M_start)
        qc1, qc2 = quadcell_errors_from_variables(x, *M_start)
        reflection_count = get_reflection_count(*mirrors)
        qc = np.array([qc1, qc2], dtype=float)
        return {
            "mirrors": mirrors,
            "reflection_count": int(reflection_count),
            "qc": qc,
            "both_qc_in_range": bool(np.all(np.abs(qc) <= qc_reacquire_limit)),
            "one_qc_in_range": bool(np.any(np.abs(qc) <= qc_reacquire_limit)),
            "qc_norm": float(np.linalg.norm(qc)),
            "qc_max_abs": float(np.max(np.abs(qc))),
        }

    def build_plan(path, search_mode, candidates_checked, failure_reason=None):
        steps = []
        total_motion = sum(abs(float(path[i][2])) for i in range(1, len(path)))
        cumulative = 0.0
        x_prev = np.array(path[0][0], dtype=float)
        for item in path[1:]:
            x_next, axis_index, amount = item
            amount = float(amount)
            if abs(amount) <= min_angle_step:
                continue
            cumulative += abs(amount)
            fraction = 1.0 if total_motion <= 1e-12 else min(1.0, cumulative / total_motion)
            step = make_actuation_step(
                len(steps) + 1,
                fraction,
                x_prev,
                x_next,
                *M_start,
                max_qc_error=qc_reacquire_limit,
                max_qc_difference=None,
                motion_samples_per_step=1,
                include_edge_ends=False,
                enforce_edge_bounds=False,
                constraint_tolerance=0.0,
            )
            step["reflection_count_reacquisition_move"] = True
            step["target_N_R"] = int(target_N_R)
            step["qc_reacquire_limit"] = float(qc_reacquire_limit)
            step["qc_path_unconstrained"] = True
            steps.append(step)
            x_prev = np.array(x_next, dtype=float)

        x_final = np.array(path[-1][0], dtype=float)
        final_metrics = metrics(x_final)
        final_edge_summary = reflection_edge_summary(x_final, *M_start, include_ends=False)
        plan = {
            "steps": steps,
            "n_steps": len(steps),
            "reflection_count_reacquisition": True,
            "reflection_count_change": True,
            "rotation_only": True,
            "qc_path_unconstrained": True,
            "search_mode": search_mode,
            "target_N_R": int(target_N_R),
            "start_reflections": int(start_reflections),
            "target_reflections": int(final_metrics["reflection_count"]),
            "reflection_count_verified": False,
            "qc_reacquire_limit": float(qc_reacquire_limit),
            "angle_scan_limit": float(angle_scan_limit),
            "scan_samples": int(scan_samples),
            "candidates_checked": int(candidates_checked),
            "start_mirrors": M_start,
            "target_mirrors": final_metrics["mirrors"],
            "start_x": x_start,
            "target_x": x_final,
            "final_OPD": float(OPD_from_variables(x_final, *M_start)),
            "final_qc1_error": float(final_metrics["qc"][0]),
            "final_qc2_error": float(final_metrics["qc"][1]),
            "final_qc_difference": float(final_metrics["qc"][0] - final_metrics["qc"][1]),
            "final_qc_max_abs": float(final_metrics["qc_max_abs"]),
            "min_reflection_u": float(final_edge_summary["min_u"]),
            "max_reflection_u": float(final_edge_summary["max_u"]),
            "reflection_u_values": final_edge_summary["u_values"],
            "failure_reason": failure_reason,
        }
        res = SimpleNamespace(
            success=failure_reason is None,
            message=(
                "Reflection-count reacquisition plan found."
                if failure_reason is None else failure_reason
            ),
        )
        return final_metrics["mirrors"], res, plan

    if start_reflections == target_N_R and max(abs(start_qc1), abs(start_qc2)) <= qc_reacquire_limit:
        return build_plan(
            [(x_start, None, 0.0)],
            "already_reacquired",
            0,
            failure_reason=None,
        )

    one_axis_candidates = []
    first_leg_candidates = []
    candidates_checked = 0

    for label, axis_index in angle_axes:
        for direction in (1.0, -1.0):
            for amount_abs in scan_amounts:
                amount = float(direction * amount_abs)
                x_trial = variables_with_axis_move(x_start, axis_index, amount)
                trial = metrics(x_trial)
                candidates_checked += 1
                if trial["reflection_count"] != target_N_R:
                    continue
                if trial["both_qc_in_range"]:
                    one_axis_candidates.append((
                        abs(amount),
                        trial["qc_norm"],
                        axis_index,
                        amount,
                        x_trial,
                    ))
                if trial["one_qc_in_range"]:
                    first_leg_candidates.append((
                        abs(amount),
                        trial["qc_max_abs"],
                        trial["qc_norm"],
                        axis_index,
                        amount,
                        x_trial,
                    ))

    if one_axis_candidates:
        one_axis_candidates.sort(key=lambda row: (row[0], row[1]))
        _, _, axis_index, amount, x_final = one_axis_candidates[0]
        return build_plan(
            [
                (x_start, None, 0.0),
                (x_final, axis_index, amount),
            ],
            "one_axis",
            candidates_checked,
            failure_reason=None,
        )

    first_leg_candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    first_leg_candidates = first_leg_candidates[:max_first_leg_candidates]
    two_axis_candidates = []

    for first in first_leg_candidates:
        _, _, _, first_axis, first_amount, x_first = first
        for label, second_axis in angle_axes:
            if second_axis == first_axis:
                continue
            for direction in (1.0, -1.0):
                for amount_abs in scan_amounts:
                    second_amount = float(direction * amount_abs)
                    x_final = variables_with_axis_move(x_first, second_axis, second_amount)
                    trial = metrics(x_final)
                    candidates_checked += 1
                    if trial["reflection_count"] != target_N_R:
                        continue
                    if not trial["both_qc_in_range"]:
                        continue
                    total_motion = abs(float(first_amount)) + abs(float(second_amount))
                    two_axis_candidates.append((
                        total_motion,
                        trial["qc_norm"],
                        first_axis,
                        first_amount,
                        x_first,
                        second_axis,
                        second_amount,
                        x_final,
                    ))

    if two_axis_candidates:
        two_axis_candidates.sort(key=lambda row: (row[0], row[1]))
        _, _, first_axis, first_amount, x_first, second_axis, second_amount, x_final = two_axis_candidates[0]
        return build_plan(
            [
                (x_start, None, 0.0),
                (x_first, first_axis, first_amount),
                (x_final, second_axis, second_amount),
            ],
            "two_axis",
            candidates_checked,
            failure_reason=None,
        )

    failure_reason = (
        "No one- or two-actuator reacquisition plan found for "
        f"N_R={target_N_R} within +/-{qc_reacquire_limit} mm over "
        f"+/-{angle_scan_limit} deg."
    )
    return build_plan(
        [(x_start, None, 0.0)],
        "failed",
        candidates_checked,
        failure_reason=failure_reason,
    )


def plan_reflection_count_staged_reacquisition(M1, M2, M3, M4,
                                               target_N_R,
                                               qc_reacquire_limit=3.0,
                                               stage_qc_limit=3.0,
                                               angle_scan_limit=1.0,
                                               scan_samples=1001,
                                               stage_n_tries=2000,
                                               stage_angle_perturb=0.3,
                                               stage_seed=0,
                                               stage_search_mode="random",
                                               forced_stage_samples=41,
                                               forced_stage_free_angle_regularization=0.02,
                                               forced_stage_max_nfev=160,
                                               stage_sigma_edge=0.02,
                                               target_center_after_jump=True,
                                               target_qc_tolerance=0.5,
                                               target_center_u_min=None,
                                               target_center_u_max=None,
                                               max_target_jump_center_candidates=12,
                                               max_stage_candidates=80,
                                               stage_max_axis_splits=80,
                                               stage_waypoint_depth=4,
                                               stage_motion_samples_per_step=15,
                                               stage_fast_motion_samples_per_step=5,
                                               u_min=0.1,
                                               u_max=0.9,
                                               min_angle_step=1e-12,
                                               profile_callback=None):
    """Plan same-N_R staging followed by a target-N_R jump.

    The topology-changing jump is intentionally one rotation actuator. When
    target_center_after_jump is true, loose target-N_R landings are accepted
    only if they can be followed by a target-N_R path that centers both
    quadcells while respecting reflection-u bounds.
    """
    M_start = (
        np.array(M1, dtype=float),
        np.array(M2, dtype=float),
        np.array(M3, dtype=float),
        np.array(M4, dtype=float),
    )
    x_start = pack_variables(*M_start)
    target_N_R = int(target_N_R)
    qc_reacquire_limit = float(qc_reacquire_limit)
    stage_qc_limit = float(stage_qc_limit)
    angle_scan_limit = float(abs(angle_scan_limit))
    scan_samples = max(3, int(scan_samples))
    stage_n_tries = max(0, int(stage_n_tries))
    stage_search_mode = str(stage_search_mode)
    if stage_search_mode not in {"random", "forced"}:
        raise ValueError("stage_search_mode must be 'random' or 'forced'.")
    forced_stage_samples = max(3, int(forced_stage_samples))
    forced_stage_free_angle_regularization = float(forced_stage_free_angle_regularization)
    forced_stage_max_nfev = max(20, int(forced_stage_max_nfev))
    stage_sigma_edge = float(stage_sigma_edge)
    target_center_after_jump = bool(target_center_after_jump)
    target_qc_tolerance = float(target_qc_tolerance)
    target_center_u_min = u_min if target_center_u_min is None else float(target_center_u_min)
    target_center_u_max = u_max if target_center_u_max is None else float(target_center_u_max)
    max_target_jump_center_candidates = max(1, int(max_target_jump_center_candidates))
    max_stage_candidates = max(1, int(max_stage_candidates))
    stage_max_axis_splits = max(1, int(stage_max_axis_splits))
    stage_waypoint_depth = max(0, int(stage_waypoint_depth))
    stage_motion_samples_per_step = max(1, int(stage_motion_samples_per_step))
    stage_fast_motion_samples_per_step = max(1, int(stage_fast_motion_samples_per_step))

    angle_axes = [
        ("M1.dangle", 1),
        ("M2.dangle", 3),
        ("M3.dangle", 5),
        ("M4.dangle", 7),
    ]
    angle_axis_indices = np.array([axis for _, axis in angle_axes], dtype=int)
    angle_axis_to_local = {int(axis): idx for idx, axis in enumerate(angle_axis_indices)}
    scan_amounts = np.linspace(0.0, angle_scan_limit, scan_samples)[1:]
    start_reflections = get_reflection_count(*M_start)
    rng = np.random.default_rng(stage_seed)
    strategy_label = (
        "forced_stage_one_axis"
        if stage_search_mode == "forced"
        else "staged_one_axis"
    )
    planner_t0 = time.perf_counter()
    planner_timing = {}

    def profile_plan(message):
        if profile_callback is not None:
            profile_callback(message)

    def add_timing(name, elapsed):
        planner_timing[name] = float(planner_timing.get(name, 0.0) + elapsed)

    def finalize_timing():
        planner_timing["total"] = float(time.perf_counter() - planner_t0)
        return dict(planner_timing)

    def metrics(x, qc_limit):
        mirrors = unpack_variables(x, *M_start)
        qc1, qc2 = quadcell_errors_from_variables(x, *M_start)
        qc = np.array([qc1, qc2], dtype=float)
        reflection_count = get_reflection_count(*mirrors)
        return {
            "mirrors": mirrors,
            "reflection_count": int(reflection_count),
            "qc": qc,
            "both_qc_in_range": bool(np.all(np.abs(qc) <= float(qc_limit))),
            "qc_norm": float(np.linalg.norm(qc)),
            "qc_max_abs": float(np.max(np.abs(qc))),
        }

    def near_miss_payload(item):
        if item is None:
            return None
        return {
            "axis_index": int(item["axis_index"]),
            "actuator": actuator_label(int(item["axis_index"])),
            "amount": float(item["amount"]),
            "qc": np.array(item["qc"], dtype=float).tolist(),
            "qc_norm": float(item["qc_norm"]),
            "qc_max_abs": float(item["qc_max_abs"]),
            "reflection_count": int(item["reflection_count"]),
        }

    def better_near_miss(candidate, best):
        if candidate is None:
            return best
        if best is None:
            return candidate
        candidate_key = (
            float(candidate["qc_max_abs"]),
            float(candidate["qc_norm"]),
            abs(float(candidate["amount"])),
        )
        best_key = (
            float(best["qc_max_abs"]),
            float(best["qc_norm"]),
            abs(float(best["amount"])),
        )
        return candidate if candidate_key < best_key else best

    def scan_one_axis_target_jump(x_origin):
        candidates = []
        best_near_miss = None
        candidates_checked = 0
        target_count_hits = 0

        for label, axis_index in angle_axes:
            for direction in (1.0, -1.0):
                for amount_abs in scan_amounts:
                    amount = float(direction * amount_abs)
                    x_trial = variables_with_axis_move(x_origin, axis_index, amount)
                    trial = metrics(x_trial, qc_reacquire_limit)
                    candidates_checked += 1
                    if trial["reflection_count"] != target_N_R:
                        continue

                    target_count_hits += 1
                    near_miss = {
                        "axis_index": int(axis_index),
                        "amount": float(amount),
                        "x": x_trial,
                        "qc": trial["qc"],
                        "qc_norm": float(trial["qc_norm"]),
                        "qc_max_abs": float(trial["qc_max_abs"]),
                        "reflection_count": int(trial["reflection_count"]),
                    }
                    best_near_miss = better_near_miss(near_miss, best_near_miss)
                    if trial["both_qc_in_range"]:
                        candidates.append((
                            abs(amount),
                            trial["qc_max_abs"],
                            trial["qc_norm"],
                            int(axis_index),
                            float(amount),
                            x_trial,
                            trial,
                        ))

        candidate_payloads = []
        if candidates:
            candidates.sort(key=lambda row: (row[0], row[1], row[2]))
            for _, _, _, axis_index_i, amount_i, x_final_i, metrics_i in candidates:
                candidate_payloads.append({
                    "axis_index": int(axis_index_i),
                    "amount": float(amount_i),
                    "x_final": np.array(x_final_i, dtype=float),
                    "metrics": metrics_i,
                })
            _, _, _, axis_index, amount, x_final, final_metrics = candidates[0]
            return {
                "found": True,
                "axis_index": int(axis_index),
                "amount": float(amount),
                "x_final": x_final,
                "metrics": final_metrics,
                "candidates": candidate_payloads,
                "candidates_checked": int(candidates_checked),
                "target_count_hits": int(target_count_hits),
                "best_near_miss": best_near_miss,
            }

        return {
            "found": False,
            "candidates": [],
            "candidates_checked": int(candidates_checked),
            "target_count_hits": int(target_count_hits),
            "best_near_miss": best_near_miss,
        }

    def make_target_jump_step(step_index, x_previous, jump):
        x_final = np.array(jump["x_final"], dtype=float)
        step = make_actuation_step(
            step_index,
            1.0,
            x_previous,
            x_final,
            *M_start,
            max_qc_error=qc_reacquire_limit,
            max_qc_difference=None,
            motion_samples_per_step=1,
            u_min=u_min,
            u_max=u_max,
            include_edge_ends=False,
            enforce_edge_bounds=False,
            constraint_tolerance=0.0,
        )
        step["reflection_count_reacquisition_move"] = True
        step["target_jump_step"] = True
        step["target_N_R"] = int(target_N_R)
        step["qc_reacquire_limit"] = float(qc_reacquire_limit)
        step["qc_path_unconstrained"] = True
        step["reacquire_stop_mode"] = "qc_only"
        return step

    def evaluate_centered_target_jump(jump, diagnostics):
        x_jump = np.array(jump["x_final"], dtype=float)
        jump_metrics = metrics(x_jump, qc_reacquire_limit)

        if not target_center_after_jump:
            return {
                "found": True,
                "jump": jump,
                "center_steps": [],
                "center_plan": None,
                "x_final": x_jump,
                "metrics": jump_metrics,
                "center_motion": 0.0,
                "edge_summary": reflection_edge_summary(x_jump, *M_start, include_ends=False),
            }

        diagnostics["target_center_attempts"] += 1
        solve_t0 = time.perf_counter()
        try:
            x_centered, center_res = solve_recenter_angles(
                x_jump,
                *M_start,
                target_reflections=int(target_N_R),
                max_qc_error=target_qc_tolerance,
                u_min=target_center_u_min,
                u_max=target_center_u_max,
                sigma_edge=stage_sigma_edge,
                include_edge_ends=False,
                verbose=0,
            )
        except Exception as exc:
            add_timing("target_center_solve", time.perf_counter() - solve_t0)
            diagnostics["target_center_failures"] += 1
            diagnostics["last_target_center_failure"] = str(exc)
            return {"found": False, "failure_reason": str(exc)}

        add_timing("target_center_solve", time.perf_counter() - solve_t0)
        centered_metrics = metrics(x_centered, target_qc_tolerance)
        if (
            not getattr(center_res, "success", False) or
            centered_metrics["reflection_count"] != int(target_N_R) or
            not centered_metrics["both_qc_in_range"]
        ):
            diagnostics["target_center_failures"] += 1
            diagnostics["last_target_center_failure"] = getattr(
                center_res,
                "message",
                "target-N_R centering solve did not meet constraints",
            )
            return {
                "found": False,
                "failure_reason": diagnostics["last_target_center_failure"],
            }

        diagnostics["target_center_solve_successes"] += 1

        center_steps = []
        path_t0 = time.perf_counter()
        x_routed, center_plan = append_waypoint_constrained_path_steps(
            center_steps,
            x_jump,
            x_centered,
            *M_start,
            max_axis_splits=stage_max_axis_splits,
            max_waypoint_depth=stage_waypoint_depth,
            max_qc_error=qc_reacquire_limit,
            max_qc_difference=None,
            preserve_reflection_count=True,
            motion_samples_per_step=stage_motion_samples_per_step,
            fast_motion_samples_per_step=stage_fast_motion_samples_per_step,
            u_min=target_center_u_min,
            u_max=target_center_u_max,
            enforce_edge_bounds=True,
            include_edge_ends=False,
            constraint_tolerance=0.0,
        )
        add_timing("target_center_path", time.perf_counter() - path_t0)

        if center_plan.get("failure_reason") is not None:
            diagnostics["target_center_path_failures"] += 1
            diagnostics["last_target_center_failure"] = center_plan["failure_reason"]
            return {
                "found": False,
                "failure_reason": center_plan["failure_reason"],
            }
        if np.linalg.norm(np.array(x_routed, dtype=float) - np.array(x_centered, dtype=float)) > 1e-8:
            diagnostics["target_center_path_failures"] += 1
            diagnostics["last_target_center_failure"] = "target center path did not reach centered endpoint"
            return {
                "found": False,
                "failure_reason": diagnostics["last_target_center_failure"],
            }

        diagnostics["target_center_path_successes"] += 1
        center_motion = float(
            sum(abs(float(step.get("command_value", 0.0))) for step in center_steps)
        )
        return {
            "found": True,
            "jump": jump,
            "center_steps": center_steps,
            "center_plan": center_plan,
            "x_final": np.array(x_centered, dtype=float),
            "metrics": centered_metrics,
            "center_motion": center_motion,
            "edge_summary": reflection_edge_summary(x_centered, *M_start, include_ends=False),
        }

    def build_success_plan(stage_steps, stage_plan, x_stage, jump, search_mode, diagnostics, centered_target=None):
        steps = []
        for source_step in stage_steps:
            step = dict(source_step)
            step["step"] = len(steps) + 1
            step["reflection_count_stage_move"] = True
            step["target_jump_step"] = False
            step["target_N_R"] = int(target_N_R)
            step["stage_reflections"] = int(start_reflections)
            step["qc_stage_limit"] = float(stage_qc_limit)
            steps.append(step)

        target_jump_step = make_target_jump_step(len(steps) + 1, x_stage, jump)
        target_jump_step["disable_reacquire_stop"] = bool(
            target_center_after_jump and centered_target is not None
        )
        steps.append(target_jump_step)

        center_steps = []
        center_plan = None
        center_motion = 0.0
        if centered_target is not None:
            center_plan = centered_target.get("center_plan")
            center_motion = float(centered_target.get("center_motion", 0.0))
            for source_step in centered_target.get("center_steps", []):
                step = dict(source_step)
                step["step"] = len(steps) + 1
                step["reflection_count_target_center_move"] = True
                step["target_center_step"] = True
                step["target_jump_step"] = False
                step["target_N_R"] = int(target_N_R)
                step["qc_center_tolerance"] = float(target_qc_tolerance)
                step["qc_reacquire_limit"] = float(qc_reacquire_limit)
                steps.append(step)
                center_steps.append(step)

        x_jump = np.array(jump["x_final"], dtype=float)
        jump_metrics = metrics(x_jump, qc_reacquire_limit)
        x_final = (
            np.array(centered_target["x_final"], dtype=float)
            if centered_target is not None else
            x_jump
        )
        final_metrics = metrics(x_final, qc_reacquire_limit)
        stage_metrics = metrics(x_stage, stage_qc_limit)
        final_edge_summary = reflection_edge_summary(x_final, *M_start, include_ends=False)
        stage_motion = float(sum(abs(float(step.get("command_value", 0.0))) for step in stage_steps))
        jump_motion = float(abs(jump["amount"]))
        centered_plan_enabled = bool(target_center_after_jump and centered_target is not None)

        plan = {
            "steps": steps,
            "n_steps": len(steps),
            "reflection_count_reacquisition": True,
            "reflection_count_change": True,
            "rotation_only": True,
            "qc_path_unconstrained": True,
            "reacquisition_strategy": strategy_label,
            "search_mode": search_mode,
            "stage_search_mode": stage_search_mode,
            "target_N_R": int(target_N_R),
            "start_reflections": int(start_reflections),
            "stage_reflections": int(stage_metrics["reflection_count"]),
            "target_reflections": int(final_metrics["reflection_count"]),
            "reflection_count_verified": False,
            "requires_inverse_refresh": not centered_plan_enabled,
            "qc_only_reacquire_stop": not centered_plan_enabled,
            "target_center_after_jump": bool(target_center_after_jump),
            "target_centered_plan": bool(centered_plan_enabled),
            "target_qc_tolerance": float(target_qc_tolerance),
            "target_center_u_min": float(target_center_u_min),
            "target_center_u_max": float(target_center_u_max),
            "suggested_next_step": (
                None if centered_plan_enabled
                else "take light/dark images and run optimize_inverse"
            ),
            "qc_reacquire_limit": float(qc_reacquire_limit),
            "stage_qc_limit": float(stage_qc_limit),
            "angle_scan_limit": float(angle_scan_limit),
            "scan_samples": int(scan_samples),
            "stage_n_tries": int(stage_n_tries),
            "stage_angle_perturb": float(stage_angle_perturb),
            "forced_stage_samples": int(forced_stage_samples),
            "forced_stage_free_angle_regularization": float(forced_stage_free_angle_regularization),
            "forced_stage_max_nfev": int(forced_stage_max_nfev),
            "max_stage_candidates": int(max_stage_candidates),
            "planner_timing": finalize_timing(),
            "candidates_checked": int(diagnostics.get("total_candidates_checked", 0)),
            "direct_candidates_checked": int(diagnostics.get("direct_candidates_checked", 0)),
            "direct_target_count_hits": int(diagnostics.get("direct_target_count_hits", 0)),
            "staging_candidates_checked": int(diagnostics.get("staging_candidates_checked", 0)),
            "same_reflection_stage_candidates": int(diagnostics.get("same_reflection_stage_candidates", 0)),
            "qc_valid_stage_candidates": int(diagnostics.get("qc_valid_stage_candidates", 0)),
            "reachable_staging_candidates": int(diagnostics.get("reachable_staging_candidates", 0)),
            "stage_jump_candidates_checked": int(diagnostics.get("stage_jump_candidates_checked", 0)),
            "stage_target_count_hits": int(diagnostics.get("stage_target_count_hits", 0)),
            "stage_recenter_attempts": int(diagnostics.get("stage_recenter_attempts", 0)),
            "stage_recenter_successes": int(diagnostics.get("stage_recenter_successes", 0)),
            "target_center_attempts": int(diagnostics.get("target_center_attempts", 0)),
            "target_center_solve_successes": int(diagnostics.get("target_center_solve_successes", 0)),
            "target_center_path_successes": int(diagnostics.get("target_center_path_successes", 0)),
            "target_center_failures": int(diagnostics.get("target_center_failures", 0)),
            "target_center_path_failures": int(diagnostics.get("target_center_path_failures", 0)),
            "last_target_center_failure": diagnostics.get("last_target_center_failure"),
            "best_near_miss_qc": near_miss_payload(diagnostics.get("best_near_miss")),
            "start_mirrors": M_start,
            "stage_mirrors": stage_metrics["mirrors"],
            "target_mirrors": final_metrics["mirrors"],
            "start_x": x_start,
            "stage_x": np.array(x_stage, dtype=float),
            "target_jump_x": x_jump,
            "target_x": x_final,
            "stage_plan": stage_plan,
            "target_jump_step": target_jump_step,
            "target_center_steps": center_steps,
            "target_center_plan": center_plan,
            "stage_motion_total_abs": float(stage_motion),
            "target_jump_motion_abs": float(jump_motion),
            "target_center_motion_total_abs": float(center_motion),
            "final_OPD": float(OPD_from_variables(x_final, *M_start)),
            "target_jump_qc1_error": float(jump_metrics["qc"][0]),
            "target_jump_qc2_error": float(jump_metrics["qc"][1]),
            "target_jump_qc_max_abs": float(jump_metrics["qc_max_abs"]),
            "final_qc1_error": float(final_metrics["qc"][0]),
            "final_qc2_error": float(final_metrics["qc"][1]),
            "final_qc_difference": float(final_metrics["qc"][0] - final_metrics["qc"][1]),
            "final_qc_max_abs": float(final_metrics["qc_max_abs"]),
            "min_reflection_u": float(final_edge_summary["min_u"]),
            "max_reflection_u": float(final_edge_summary["max_u"]),
            "reflection_u_values": final_edge_summary["u_values"],
            "failure_reason": None,
        }
        res = SimpleNamespace(
            success=True,
            message=(
                "Staged one-actuator reflection-count plan found with target-N_R centering."
                if centered_plan_enabled else
                "Staged one-actuator reflection-count reacquisition plan found; "
                "inverse refresh required after QC reacquisition."
            ),
        )
        profile_plan(
            f"{search_mode} success total_dt={planner_timing['total']:.3f}s "
            f"checked={plan['candidates_checked']} reachable_stages="
            f"{plan['reachable_staging_candidates']}"
        )
        return final_metrics["mirrors"], res, plan

    phase_t0 = time.perf_counter()
    direct_jump = scan_one_axis_target_jump(x_start)
    add_timing("direct_scan", time.perf_counter() - phase_t0)
    profile_plan(
        f"direct_scan dt={planner_timing['direct_scan']:.3f}s "
        f"checked={direct_jump['candidates_checked']} "
        f"target_hits={direct_jump['target_count_hits']} "
        f"found={direct_jump['found']}"
    )
    diagnostics = {
        "direct_candidates_checked": int(direct_jump["candidates_checked"]),
        "direct_target_count_hits": int(direct_jump["target_count_hits"]),
        "staging_candidates_checked": 0,
        "same_reflection_stage_candidates": 0,
        "qc_valid_stage_candidates": 0,
        "reachable_staging_candidates": 0,
        "stage_jump_candidates_checked": 0,
        "stage_target_count_hits": 0,
        "stage_recenter_attempts": 0,
        "stage_recenter_successes": 0,
        "target_center_attempts": 0,
        "target_center_solve_successes": 0,
        "target_center_path_successes": 0,
        "target_center_failures": 0,
        "target_center_path_failures": 0,
        "last_target_center_failure": None,
        "best_near_miss": direct_jump.get("best_near_miss"),
    }
    if direct_jump["found"]:
        for jump_candidate in direct_jump.get("candidates", [])[:max_target_jump_center_candidates]:
            centered_target = evaluate_centered_target_jump(jump_candidate, diagnostics)
            if not centered_target.get("found", False):
                continue
            diagnostics["total_candidates_checked"] = direct_jump["candidates_checked"]
            return build_success_plan(
                [],
                None,
                x_start,
                jump_candidate,
                "direct_one_axis_centered" if target_center_after_jump else "direct_one_axis",
                diagnostics,
                centered_target=centered_target if target_center_after_jump else None,
            )

    def append_stage_prefilter_candidate(stage_prefilter, x_stage):
        stage_metrics = metrics(x_stage, stage_qc_limit)
        if stage_metrics["reflection_count"] != start_reflections:
            return
        diagnostics["same_reflection_stage_candidates"] += 1
        if not stage_metrics["both_qc_in_range"]:
            return
        diagnostics["qc_valid_stage_candidates"] += 1
        stage_motion = float(np.sum(np.abs(x_stage[angle_axis_indices] - x_start[angle_axis_indices])))
        stage_prefilter.append((
            stage_motion,
            stage_metrics["qc_max_abs"],
            stage_metrics["qc_norm"],
            x_stage,
        ))

    def solve_forced_stage(axis_index, forced_amount):
        fixed_local = angle_axis_to_local[int(axis_index)]
        free_local = [idx for idx in range(len(angle_axis_indices)) if idx != fixed_local]
        theta_start = np.array(x_start[angle_axis_indices], dtype=float)
        theta_fixed = theta_start.copy()
        theta_fixed[fixed_local] = theta_start[fixed_local] + float(forced_amount)
        free0 = theta_start[free_local].copy()

        def angles_from_free(free_angles):
            theta = theta_fixed.copy()
            theta[free_local] = np.array(free_angles, dtype=float)
            return theta

        def forced_residuals(free_angles):
            theta = angles_from_free(free_angles)
            residuals = center_quadcells_residuals(
                theta,
                *M_start,
                target_reflections=int(start_reflections),
                u_min=u_min,
                u_max=u_max,
                sigma_edge=stage_sigma_edge,
            )
            regularization = (
                forced_stage_free_angle_regularization *
                (np.array(free_angles, dtype=float) - theta_start[free_local])
            )
            return np.concatenate([residuals, regularization])

        res = least_squares(
            fun=forced_residuals,
            x0=free0,
            loss="linear",
            f_scale=1.0,
            x_scale="jac",
            max_nfev=forced_stage_max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        x_stage = x_start.copy()
        x_stage[angle_axis_indices] = angles_from_free(res.x)
        return x_stage, res

    phase_t0 = time.perf_counter()
    stage_prefilter = []
    if stage_search_mode == "random":
        for _ in range(stage_n_tries):
            x_stage = x_start.copy()
            x_stage[angle_axis_indices] += rng.uniform(
                -float(stage_angle_perturb),
                float(stage_angle_perturb),
                size=len(angle_axis_indices),
            )
            diagnostics["staging_candidates_checked"] += 1
            append_stage_prefilter_candidate(stage_prefilter, x_stage)
    else:
        forced_amounts = np.linspace(
            -float(stage_angle_perturb),
            float(stage_angle_perturb),
            forced_stage_samples,
        )
        forced_amounts = [
            float(amount)
            for amount in forced_amounts
            if abs(float(amount)) > min_angle_step
        ]
        for _, axis_index in angle_axes:
            for forced_amount in forced_amounts:
                diagnostics["staging_candidates_checked"] += 1
                diagnostics["stage_recenter_attempts"] += 1
                solve_t0 = time.perf_counter()
                x_stage, recenter_res = solve_forced_stage(axis_index, forced_amount)
                add_timing("stage_recenter_solve", time.perf_counter() - solve_t0)
                if getattr(recenter_res, "success", False):
                    diagnostics["stage_recenter_successes"] += 1
                append_stage_prefilter_candidate(stage_prefilter, x_stage)
    add_timing("stage_generation", time.perf_counter() - phase_t0)
    profile_plan(
        f"{stage_search_mode}_stage_generation "
        f"dt={planner_timing['stage_generation']:.3f}s "
        f"checked={diagnostics['staging_candidates_checked']} "
        f"same_N_R={diagnostics['same_reflection_stage_candidates']} "
        f"qc_valid={diagnostics['qc_valid_stage_candidates']}"
    )

    stage_prefilter.sort(key=lambda row: (row[0], row[1], row[2]))
    staged_candidates = []

    for stage_motion, _, _, x_stage in stage_prefilter[:max_stage_candidates]:
        stage_steps = []
        phase_t0 = time.perf_counter()
        x_routed, stage_plan = append_waypoint_constrained_path_steps(
            stage_steps,
            x_start,
            x_stage,
            *M_start,
            max_axis_splits=stage_max_axis_splits,
            max_waypoint_depth=stage_waypoint_depth,
            max_qc_error=stage_qc_limit,
            max_qc_difference=None,
            preserve_reflection_count=True,
            motion_samples_per_step=stage_motion_samples_per_step,
            fast_motion_samples_per_step=stage_fast_motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            enforce_edge_bounds=False,
            include_edge_ends=False,
            constraint_tolerance=0.0,
        )
        add_timing("stage_path_validation", time.perf_counter() - phase_t0)
        if stage_plan.get("failure_reason") is not None:
            continue
        if np.linalg.norm(np.array(x_routed, dtype=float) - np.array(x_stage, dtype=float)) > 1e-8:
            continue

        diagnostics["reachable_staging_candidates"] += 1
        phase_t0 = time.perf_counter()
        jump = scan_one_axis_target_jump(x_stage)
        add_timing("stage_jump_scan", time.perf_counter() - phase_t0)
        diagnostics["stage_jump_candidates_checked"] += int(jump["candidates_checked"])
        diagnostics["stage_target_count_hits"] += int(jump["target_count_hits"])
        diagnostics["best_near_miss"] = better_near_miss(
            jump.get("best_near_miss"),
            diagnostics["best_near_miss"],
        )
        if not jump["found"]:
            continue

        for jump_candidate in jump.get("candidates", [])[:max_target_jump_center_candidates]:
            centered_target = evaluate_centered_target_jump(jump_candidate, diagnostics)
            if not centered_target.get("found", False):
                continue

            edge_margin = float(centered_target["edge_summary"]["closest_edge_margin"])
            score = (
                float(stage_motion),
                abs(float(jump_candidate["amount"])),
                float(centered_target.get("center_motion", 0.0)),
                float(centered_target["metrics"]["qc_max_abs"]),
                -edge_margin if np.isfinite(edge_margin) else float("inf"),
            )
            staged_candidates.append((score, stage_steps, stage_plan, x_stage, jump_candidate, centered_target))
            if stage_search_mode == "forced":
                diagnostics["total_candidates_checked"] = (
                    diagnostics["direct_candidates_checked"] +
                    diagnostics["staging_candidates_checked"] +
                    diagnostics["stage_jump_candidates_checked"]
                )
                return build_success_plan(
                    stage_steps,
                    stage_plan,
                    x_stage,
                    jump_candidate,
                    "forced_stage_one_axis_centered" if target_center_after_jump else "forced_stage_one_axis",
                    diagnostics,
                    centered_target=centered_target if target_center_after_jump else None,
                )
            break

    if staged_candidates:
        staged_candidates.sort(key=lambda row: row[0])
        _, stage_steps, stage_plan, x_stage, jump, centered_target = staged_candidates[0]
        diagnostics["total_candidates_checked"] = (
            diagnostics["direct_candidates_checked"] +
            diagnostics["staging_candidates_checked"] +
            diagnostics["stage_jump_candidates_checked"]
        )
        return build_success_plan(
            stage_steps,
            stage_plan,
            x_stage,
            jump,
            (
                "forced_stage_one_axis_centered"
                if stage_search_mode == "forced" and target_center_after_jump else
                "forced_stage_one_axis"
                if stage_search_mode == "forced" else
                "staged_one_axis_centered"
                if target_center_after_jump else
                "staged_one_axis"
            ),
            diagnostics,
            centered_target=centered_target if target_center_after_jump else None,
        )

    diagnostics["total_candidates_checked"] = (
        diagnostics["direct_candidates_checked"] +
        diagnostics["staging_candidates_checked"] +
        diagnostics["stage_jump_candidates_checked"]
    )
    if target_center_after_jump and diagnostics.get("target_center_attempts", 0) > 0:
        failure_reason = (
            "No staged one-actuator reacquisition plan found for "
            f"N_R={target_N_R} that could also center both quadcells within "
            f"+/-{target_qc_tolerance} mm while preserving reflection-u bounds. "
            f"Target-centering u bounds were [{target_center_u_min}, {target_center_u_max}]. "
            f"Loose target landings checked for centering: "
            f"{diagnostics['target_center_attempts']}. "
            f"Last centering/path failure: {diagnostics.get('last_target_center_failure')}"
        )
    else:
        failure_reason = (
            "No staged one-actuator reacquisition plan found for "
            f"N_R={target_N_R} within +/-{qc_reacquire_limit} mm after "
            f"{diagnostics['staging_candidates_checked']} staging samples."
        )
    start_metrics = metrics(x_start, qc_reacquire_limit)
    edge_summary = reflection_edge_summary(x_start, *M_start, include_ends=False)
    plan = {
        "steps": [],
        "n_steps": 0,
        "reflection_count_reacquisition": True,
        "reflection_count_change": True,
        "rotation_only": True,
        "qc_path_unconstrained": True,
        "reacquisition_strategy": strategy_label,
        "search_mode": "failed",
        "stage_search_mode": stage_search_mode,
        "target_N_R": int(target_N_R),
        "start_reflections": int(start_reflections),
        "target_reflections": int(start_metrics["reflection_count"]),
        "reflection_count_verified": False,
        "requires_inverse_refresh": False,
        "qc_only_reacquire_stop": True,
        "target_center_after_jump": bool(target_center_after_jump),
        "target_centered_plan": False,
        "target_qc_tolerance": float(target_qc_tolerance),
        "target_center_u_min": float(target_center_u_min),
        "target_center_u_max": float(target_center_u_max),
        "qc_reacquire_limit": float(qc_reacquire_limit),
        "stage_qc_limit": float(stage_qc_limit),
        "angle_scan_limit": float(angle_scan_limit),
        "scan_samples": int(scan_samples),
        "stage_n_tries": int(stage_n_tries),
        "stage_angle_perturb": float(stage_angle_perturb),
        "forced_stage_samples": int(forced_stage_samples),
        "forced_stage_free_angle_regularization": float(forced_stage_free_angle_regularization),
        "forced_stage_max_nfev": int(forced_stage_max_nfev),
        "max_stage_candidates": int(max_stage_candidates),
        "planner_timing": finalize_timing(),
        "candidates_checked": int(diagnostics["total_candidates_checked"]),
        "direct_candidates_checked": int(diagnostics["direct_candidates_checked"]),
        "direct_target_count_hits": int(diagnostics["direct_target_count_hits"]),
        "staging_candidates_checked": int(diagnostics["staging_candidates_checked"]),
        "same_reflection_stage_candidates": int(diagnostics["same_reflection_stage_candidates"]),
        "qc_valid_stage_candidates": int(diagnostics["qc_valid_stage_candidates"]),
        "reachable_staging_candidates": int(diagnostics["reachable_staging_candidates"]),
        "stage_jump_candidates_checked": int(diagnostics["stage_jump_candidates_checked"]),
        "stage_target_count_hits": int(diagnostics["stage_target_count_hits"]),
        "stage_recenter_attempts": int(diagnostics["stage_recenter_attempts"]),
        "stage_recenter_successes": int(diagnostics["stage_recenter_successes"]),
        "target_center_attempts": int(diagnostics["target_center_attempts"]),
        "target_center_solve_successes": int(diagnostics["target_center_solve_successes"]),
        "target_center_path_successes": int(diagnostics["target_center_path_successes"]),
        "target_center_failures": int(diagnostics["target_center_failures"]),
        "target_center_path_failures": int(diagnostics["target_center_path_failures"]),
        "last_target_center_failure": diagnostics.get("last_target_center_failure"),
        "best_near_miss_qc": near_miss_payload(diagnostics["best_near_miss"]),
        "start_mirrors": M_start,
        "target_mirrors": M_start,
        "start_x": x_start,
        "target_jump_x": None,
        "target_x": x_start.copy(),
        "stage_plan": None,
        "target_jump_step": None,
        "target_center_steps": [],
        "target_center_plan": None,
        "final_OPD": float(OPD_from_variables(x_start, *M_start)),
        "final_qc1_error": float(start_metrics["qc"][0]),
        "final_qc2_error": float(start_metrics["qc"][1]),
        "final_qc_difference": float(start_metrics["qc"][0] - start_metrics["qc"][1]),
        "final_qc_max_abs": float(start_metrics["qc_max_abs"]),
        "min_reflection_u": float(edge_summary["min_u"]),
        "max_reflection_u": float(edge_summary["max_u"]),
        "reflection_u_values": edge_summary["u_values"],
        "failure_reason": failure_reason,
    }
    res = SimpleNamespace(success=False, message=failure_reason)
    profile_plan(
        f"{stage_search_mode}_stage failed total_dt={planner_timing['total']:.3f}s "
        f"checked={plan['candidates_checked']} reachable_stages="
        f"{plan['reachable_staging_candidates']}"
    )
    return M_start, res, plan

#simulation(M1_opt[0], M1_opt[1],
#                      M2_opt[0], M2_opt[1], 
#                      M3_opt[0], M3_opt[1], 
#                      M4_opt[0], M4_opt[1],
#                      M1_opt[2], M2_opt[2], M3_opt[2], M4_opt[2]), best_res

def choose_OPD(target_OPD, M1, M2, M3, M4,
               return_actuation_plan=True,
               n_actuation_steps=None,
               max_axis_splits=20,
               qc_plan_limit=1.5,
               qc_detector_limit=3.9,
               qc_hardware_stop=3.5,
               max_qc_difference=None,
               preserve_reflection_count=True,
               motion_samples_per_step=25,
               moving_linear_stages=("M1", "M2", "M3"),
               max_OPD_step=20.0,
               u_min=0.1,
               u_max=0.9,
               linear_u_min=0.05,
               linear_u_max=0.95,
               sigma_edge=0.02,
               enforce_edge_bounds=True,
               include_edge_ends=False,
               constraint_tolerance=0.0,
               target_OPD_tolerance=0.05,
               final_qc_tolerance=0.5,
               final_center_qc_threshold=0.5,
               final_OPD_relaxed_tolerance=0.5,
               final_center_qc_priority=True,
               fast_recenter_path=True,
               fast_recenter_motion_samples_per_step=5,
               final_endpoint_waypoint_depth=4,
               auto_recenter_start=True,
               recenter_constraint_tolerance=0.25,
               optimizer_verbose=0,
               M1_linear_loc=None,
               M2_linear_loc=None,
               M3_linear_loc=None,
               profile=False,
               profile_sink=None,
               **legacy_qc_kwargs):
    if "max_qc_error" in legacy_qc_kwargs:
        qc_plan_limit = legacy_qc_kwargs.pop("max_qc_error")
    if "qc_hard_limit" in legacy_qc_kwargs:
        qc_hardware_stop = legacy_qc_kwargs.pop("qc_hard_limit")
    if "final_endpoint_path_qc_limit" in legacy_qc_kwargs:
        qc_plan_limit = legacy_qc_kwargs.pop("final_endpoint_path_qc_limit")
    if len(legacy_qc_kwargs) > 0:
        unknown = ", ".join(sorted(legacy_qc_kwargs))
        raise TypeError(f"choose_OPD() got unexpected keyword argument(s): {unknown}")

    qc_plan_limit = float(qc_plan_limit)
    qc_detector_limit = float(qc_detector_limit)
    qc_hardware_stop = float(qc_hardware_stop)
    max_qc_error = qc_plan_limit

    x_start = pack_variables(M1, M2, M3, M4)
    start_OPD = metrics_from_variables(x_start, M1, M2, M3, M4)[1]

    provided_linear_stage_locs = any(
        loc is not None for loc in [M1_linear_loc, M2_linear_loc, M3_linear_loc]
    )
    use_linear_stage_limits = provided_linear_stage_locs

    if provided_linear_stage_locs:
        M1_linear_loc = 0.0 if M1_linear_loc is None else float(M1_linear_loc)
        M2_linear_loc = 0.0 if M2_linear_loc is None else float(M2_linear_loc)
        M3_linear_loc = 0.0 if M3_linear_loc is None else float(M3_linear_loc)

    if return_actuation_plan:
        if not provided_linear_stage_locs:
            assumed_midpoint = LINEAR_STAGE_TRAVEL_MM / 2.0
            M1_linear_loc = assumed_midpoint
            M2_linear_loc = assumed_midpoint
            M3_linear_loc = assumed_midpoint
        correction_max_axis_splits = max_axis_splits if n_actuation_steps is None else n_actuation_steps

        mirrors_opt, final_res, actuation_plan = plan_OPD_linear_then_recenter(
            target_OPD,
            M1, M2, M3, M4,
            M1_linear_loc, M2_linear_loc, M3_linear_loc,
            qc_plan_limit=qc_plan_limit,
            qc_detector_limit=qc_detector_limit,
            qc_hardware_stop=qc_hardware_stop,
            max_qc_difference=max_qc_difference,
            preserve_reflection_count=preserve_reflection_count,
            motion_samples_per_step=motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            sigma_edge=sigma_edge,
            enforce_edge_bounds=enforce_edge_bounds,
            include_edge_ends=include_edge_ends,
            constraint_tolerance=constraint_tolerance,
            optimizer_verbose=optimizer_verbose,
            target_OPD_tolerance=target_OPD_tolerance,
            final_qc_tolerance=final_qc_tolerance,
            final_center_qc_threshold=final_center_qc_threshold,
            final_OPD_relaxed_tolerance=final_OPD_relaxed_tolerance,
            final_center_qc_priority=final_center_qc_priority,
            correction_max_axis_splits=correction_max_axis_splits,
            fast_recenter_path=fast_recenter_path,
            fast_recenter_motion_samples_per_step=fast_recenter_motion_samples_per_step,
            linear_u_min=linear_u_min,
            linear_u_max=linear_u_max,
            final_endpoint_waypoint_depth=final_endpoint_waypoint_depth,
            linear_stage_order=moving_linear_stages,
            profile=profile,
            profile_sink=profile_sink
        )
        actuation_plan["linear_stage_locs_were_provided"] = provided_linear_stage_locs
        if not provided_linear_stage_locs:
            actuation_plan["assumed_initial_linear_stage_locs"] = {
                "M1": LINEAR_STAGE_TRAVEL_MM / 2.0,
                "M2": LINEAR_STAGE_TRAVEL_MM / 2.0,
                "M3": LINEAR_STAGE_TRAVEL_MM / 2.0
            }
        return mirrors_opt, final_res, actuation_plan

    segment_max_OPD_step = max_OPD_step
    if use_linear_stage_limits and return_actuation_plan and max_OPD_step is not None:
        segment_max_OPD_step = max_OPD_step

    if segment_max_OPD_step is None or abs(target_OPD - start_OPD) <= segment_max_OPD_step:
        segment_targets = [target_OPD]
    else:
        n_segments = int(np.ceil(abs(target_OPD - start_OPD) / segment_max_OPD_step))
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
            variable_bounds = None
            recenter_stages = moving_linear_stages
            if use_linear_stage_limits:
                variable_bounds = linear_stage_x_bounds(
                    current_M1, current_M2, current_M3, current_M4,
                    M1_linear_loc, M2_linear_loc, M3_linear_loc
                )
                recenter_stages = ("M1", "M2", "M3")

            x_recentered, final_res = solve_OPD_configuration(
                start_OPD,
                current_M1, current_M2, current_M3, current_M4,
                moving_linear_stages=recenter_stages,
                variable_bounds=variable_bounds,
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
                final_res = set_OPD_result_full_x(final_res, M1_opt, M2_opt, M3_opt, M4_opt)
                return (M1_opt, M2_opt, M3_opt, M4_opt), final_res, actuation_plan

            previous_mirrors = (current_M1, current_M2, current_M3, current_M4)
            current_M1, current_M2, current_M3, current_M4 = unpack_variables(
                x_recentered,
                current_M1, current_M2, current_M3, current_M4
            )
            if use_linear_stage_limits:
                M1_linear_loc, M2_linear_loc, M3_linear_loc = update_linear_stage_locs(
                    previous_mirrors,
                    (current_M1, current_M2, current_M3, current_M4),
                    M1_linear_loc, M2_linear_loc, M3_linear_loc
                )

    segment_index = 0
    min_adaptive_OPD_step = 0.25

    while segment_index < len(segment_targets):
        segment_target = segment_targets[segment_index]
        if use_linear_stage_limits:
            current_OPD = metrics_from_variables(
                pack_variables(current_M1, current_M2, current_M3, current_M4),
                current_M1, current_M2, current_M3, current_M4
            )[1]
            stage_order = ("M1", "M2", "M3") if segment_target >= current_OPD else ("M3", "M2", "M1")
            segment_accepted = False
            failed_segment_plans = []

            for stage_name in stage_order:
                x_segment_start = pack_variables(current_M1, current_M2, current_M3, current_M4)
                variable_bounds = linear_stage_x_bounds(
                    current_M1, current_M2, current_M3, current_M4,
                    M1_linear_loc, M2_linear_loc, M3_linear_loc
                )
                x_segment_target, final_res = solve_OPD_configuration(
                    segment_target,
                    current_M1, current_M2, current_M3, current_M4,
                    moving_linear_stages=(stage_name,),
                    variable_bounds=variable_bounds,
                    u_min=u_min,
                    u_max=u_max,
                    sigma_edge=sigma_edge,
                    enforce_edge_bounds=enforce_edge_bounds,
                    include_edge_ends=include_edge_ends,
                    verbose=optimizer_verbose
                )

                if np.allclose(x_segment_target, x_segment_start, atol=1e-9, rtol=0):
                    current_OPD = metrics_from_variables(
                        x_segment_start,
                        current_M1, current_M2, current_M3, current_M4
                    )[1]
                    if abs(current_OPD - segment_target) <= constraint_tolerance:
                        x_opt = x_segment_target
                        break
                    continue

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
                    segment_plan["linear_stage"] = stage_name
                    segment_plan["linear_stage_locs_start"] = {
                        "M1": M1_linear_loc,
                        "M2": M2_linear_loc,
                        "M3": M3_linear_loc
                    }
                    if segment_plan["failure_reason"] is not None:
                        failed_segment_plans.append(segment_plan)
                        continue
                    segment_plans.append(segment_plan)

                previous_mirrors = (current_M1, current_M2, current_M3, current_M4)
                current_M1, current_M2, current_M3, current_M4 = unpack_variables(
                    x_segment_target,
                    current_M1, current_M2, current_M3, current_M4
                )
                M1_linear_loc, M2_linear_loc, M3_linear_loc = update_linear_stage_locs(
                    previous_mirrors,
                    (current_M1, current_M2, current_M3, current_M4),
                    M1_linear_loc, M2_linear_loc, M3_linear_loc
                )
                if return_actuation_plan:
                    segment_plans[-1]["linear_stage_locs_end"] = {
                        "M1": M1_linear_loc,
                        "M2": M2_linear_loc,
                        "M3": M3_linear_loc
                    }

                x_opt = x_segment_target
                segment_accepted = True
                current_OPD = metrics_from_variables(
                    pack_variables(current_M1, current_M2, current_M3, current_M4),
                    current_M1, current_M2, current_M3, current_M4
                )[1]
                if abs(current_OPD - segment_target) <= constraint_tolerance:
                    break

            if not segment_accepted:
                current_OPD = metrics_from_variables(
                    pack_variables(current_M1, current_M2, current_M3, current_M4),
                    current_M1, current_M2, current_M3, current_M4
                )[1]
                if abs(segment_target - current_OPD) > min_adaptive_OPD_step:
                    midpoint_OPD = current_OPD + 0.5 * (segment_target - current_OPD)
                    segment_targets.insert(segment_index, midpoint_OPD)
                    continue

                if return_actuation_plan and failed_segment_plans:
                    segment_plans.append(failed_segment_plans[-1])
                x_opt = pack_variables(current_M1, current_M2, current_M3, current_M4)
                break

            segment_index += 1
            continue

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
        segment_index += 1
    
    M1_opt, M2_opt, M3_opt, M4_opt = unpack_variables(x_opt, current_M1, current_M2, current_M3, current_M4)
    final_res = set_OPD_result_full_x(final_res, M1_opt, M2_opt, M3_opt, M4_opt)

    if not return_actuation_plan:
        return (M1_opt, M2_opt, M3_opt, M4_opt), final_res

    actuation_plan = combine_actuation_plans(segment_plans)
    if actuation_plan is None:
        actuation_plan = build_actuation_plan_summary(
            [], x_start, x_start, M1, M2, M3, M4,
            get_reflection_count(M1, M2, M3, M4),
            get_reflection_count(M1, M2, M3, M4),
            True,
            None,
            max_qc_error=max_qc_error,
            max_qc_difference=max_qc_difference,
            motion_samples_per_step=motion_samples_per_step,
            u_min=u_min,
            u_max=u_max,
            include_edge_ends=include_edge_ends,
            search_mode="failed",
            split_count=0,
            failure_reason="No actuation segments were accepted."
        )
    if use_linear_stage_limits:
        actuation_plan["final_linear_stage_locs"] = {
            "M1": M1_linear_loc,
            "M2": M2_linear_loc,
            "M3": M3_linear_loc
        }

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

    #ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, labelspacing=1., framealpha=0.9)
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

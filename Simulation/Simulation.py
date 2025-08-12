import numpy as np
from scipy.special import fresnelimport numpy as np
from scipy.special import fresnel
from scipy.io import savemat
from multiprocessing import Pool, cpu_count
import time
import os

# ---------- Parameter Definitions ----------
plane_height_range = (10000, 12000)
plane_size_range = (20, 50)
plane_speed_range = (200, 300)
direction_angle_range = (-90, 90)
wavelength_fixed = 1.0e-2
alpha_initial_range = (0, 0.03675)
offset_ratios = (0, 2)
orbit_height = 5.5e5

r_Earth = 6.371e6
G = 6.67430e-11
M_Earth = 5.972e24
T_satellite = 2 * np.pi * np.sqrt((orbit_height + r_Earth) ** 3 / (G * M_Earth))

time_step = 0.00051
start_time = -0.3
total_time = 0.6
end_time = start_time + total_time
time_array = np.arange(start_time, end_time, time_step)
Num_data_point = len(time_array)

# ---------- Control Parameters ----------
snr_levels_db = [10, 20, 30, 40]  # Noise levels, representing different SNR values
N_samples_per_bin = 5000

# ---------- Parameter Combination Generation ----------
base_param_list = []
for _ in range(N_samples_per_bin):
    ratio = np.random.uniform(*offset_ratios)
    plane_height = np.random.uniform(*plane_height_range)
    plane_size = np.random.uniform(*plane_size_range)
    plane_speed = np.random.uniform(*plane_speed_range)
    direction_angle = np.random.uniform(*direction_angle_range)
    alpha_initial = np.random.uniform(*alpha_initial_range)
    base_param_list.append((plane_height, plane_size, plane_speed,
                            direction_angle, wavelength_fixed, alpha_initial, ratio))

param_combinations = []
for base_param in base_param_list:
    for snr_db in snr_levels_db:
        param_combinations.append(base_param + (snr_db,))


# ---------- Main Function ----------
def compute_instance(params):
    plane_height, plane_size, plane_speed, direction_angle, wavelength, alpha_initial, ratio, snr_db = params
    speed_y = plane_speed * np.sin(np.deg2rad(direction_angle))
    speed_x = plane_speed * np.cos(np.deg2rad(direction_angle))

    alpha0 = alpha_initial
    theta0 = np.pi / 2 - np.arctan(((orbit_height + r_Earth) * np.sin(alpha0)) /
                                   ((orbit_height + r_Earth) * np.cos(alpha0) - r_Earth))
    dPlane0 = plane_height / np.sin(theta0) + (0 - plane_height / np.tan(theta0)) * np.cos(theta0)
    d1_0 = np.sqrt((orbit_height + r_Earth)**2 + r_Earth**2 -
                   2 * r_Earth * (orbit_height + r_Earth) * np.cos(alpha0))
    d2_0 = dPlane0
    R_F0 = np.sqrt(wavelength * d2_0)

    offset_Y0 = - ratio * R_F0
    offset_X0 = 0
    raw_signal = np.zeros(Num_data_point)

    for m, t in enumerate(time_array):
        distanceY = offset_Y0 + speed_y * t
        distanceX = offset_X0 + speed_x * t
        alpha = alpha_initial - t * 2 * np.pi / T_satellite
        theta = np.pi / 2 - np.arctan(((orbit_height + r_Earth) * np.sin(alpha)) /
                                      ((orbit_height + r_Earth) * np.cos(alpha) - r_Earth))

        dPlane = np.sqrt(plane_height**2 + distanceX**2 + distanceY**2)
        sqrt_factor = np.sqrt(2 / wavelength / dPlane)
        distance = np.sqrt(distanceX**2 + distanceY**2)

        s1 = sqrt_factor * plane_size / 2
        s2 = sqrt_factor * (plane_size / 2 - distance)
        s3 = sqrt_factor * (plane_size / 2 + distance)
        C1, S1 = fresnel(s1)
        C2, S2 = fresnel(s2)
        C3, S3 = fresnel(s3)

        temp = 1 + 1j / 2 * (2 * C1 + 1j * 2 * S1) * (C2 + C3 + 1j * S2 + 1j * S3)
        raw_signal[m] = np.real(temp)  # Can be replaced with np.abs(temp) if more appropriate

    # ---------- Add Gaussian noise according to dB definition ----------
    P_signal = np.mean(raw_signal ** 2)
    noise_power = P_signal / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0.0, noise_std, size=raw_signal.shape)
    noisy_signal = raw_signal + noise

    # ---------- Mean Normalization ----------
    clean_reference = np.ones_like(raw_signal)  # Clean signal is ideally “1”
    normalized_signal = noisy_signal / (np.mean(clean_reference) + 1e-12)

    # ---------- Jitter (perturb time axis) ----------
    if np.random.rand() < 0.1:
        jittered_time_array = time_array + np.random.normal(0.0, 0.0001, size=time_array.shape)
        normalized_signal = np.interp(time_array, jittered_time_array, normalized_signal)

    # ---------- Convert to dB power ----------
    output_FN_F = 10 * np.log10(np.clip(normalized_signal ** 2, 1e-12, None))
    label = 'large'
    return {'data': output_FN_F, 'label': label, 'fz_bin': 2, 'snr': snr_db}


# ---------- Parallel Execution and Save to Multiple Files ----------
if __name__ == '__main__':
    start = time.time()
    grouped_results = {}

    with Pool(min(6, cpu_count())) as pool:
        results = pool.map(compute_instance, param_combinations)

    for r in results:
        key = (r['fz_bin'], r['snr'])
        if key not in grouped_results:
            grouped_results[key] = {'data': [], 'label': []}
        grouped_results[key]['data'].append(r['data'])
        grouped_results[key]['label'].append(r['label'])

    for (fz, snr), content in grouped_results.items():
        data = np.array(content['data'])
        labels = np.array(content['label'])
        filename = f"/content/drive/MyDrive/MPE_Research/Data/Train_new/FZDataset_Large_Fz{fz}_SNR{snr}.mat"
        savemat(filename, {
            'all_data': data,
            'all_labels': labels,
            'fz_bin': fz,
            'snr': snr_db
        })
        print(f"Saved file: {os.path.basename(filename)}, Samples: {len(data)}")

    print(f"Total time: {(time.time()-start)/60:.2f} minutes")

from scipy.io import savemat
from multiprocessing import Pool, cpu_count
import time
import os

# ---------- Parameter Definitions ----------
plane_height_range = (10000, 12000)
plane_size_range = (20, 50)
plane_speed_range = (200, 300)
direction_angle_range = (-90, 90)
wavelength_fixed = 1.0e-2
alpha_initial_range = (0, 0.03675)
offset_ratios = (0, 2)
orbit_height = 5.5e5

r_Earth = 6.371e6
G = 6.67430e-11
M_Earth = 5.972e24
T_satellite = 2 * np.pi * np.sqrt((orbit_height + r_Earth) ** 3 / (G * M_Earth))

time_step = 0.00051
start_time = -0.3
total_time = 0.6
end_time = start_time + total_time
time_array = np.arange(start_time, end_time, time_step)
Num_data_point = len(time_array)

# ---------- Control Parameters ----------
snr_levels_db = [10, 20, 30, 40]  # Noise levels, representing different SNR values
N_samples_per_bin = 5000

# ---------- Parameter Combination Generation ----------
base_param_list = []
for _ in range(N_samples_per_bin):
    ratio = np.random.uniform(*offset_ratios)
    plane_height = np.random.uniform(*plane_height_range)
    plane_size = np.random.uniform(*plane_size_range)
    plane_speed = np.random.uniform(*plane_speed_range)
    direction_angle = np.random.uniform(*direction_angle_range)
    alpha_initial = np.random.uniform(*alpha_initial_range)
    base_param_list.append((plane_height, plane_size, plane_speed,
                            direction_angle, wavelength_fixed, alpha_initial, ratio))

param_combinations = []
for base_param in base_param_list:
    for snr_db in snr_levels_db:
        param_combinations.append(base_param + (snr_db,))


# ---------- Main Function ----------
def compute_instance(params):
    plane_height, plane_size, plane_speed, direction_angle, wavelength, alpha_initial, ratio, snr_db = params
    speed_y = plane_speed * np.sin(np.deg2rad(direction_angle))
    speed_x = plane_speed * np.cos(np.deg2rad(direction_angle))

    alpha0 = alpha_initial
    theta0 = np.pi / 2 - np.arctan(((orbit_height + r_Earth) * np.sin(alpha0)) /
                                   ((orbit_height + r_Earth) * np.cos(alpha0) - r_Earth))
    dPlane0 = plane_height / np.sin(theta0) + (0 - plane_height / np.tan(theta0)) * np.cos(theta0)
    d1_0 = np.sqrt((orbit_height + r_Earth)**2 + r_Earth**2 -
                   2 * r_Earth * (orbit_height + r_Earth) * np.cos(alpha0))
    d2_0 = dPlane0
    R_F0 = np.sqrt(wavelength * d2_0)

    offset_Y0 = - ratio * R_F0
    offset_X0 = 0
    raw_signal = np.zeros(Num_data_point)

    for m, t in enumerate(time_array):
        distanceY = offset_Y0 + speed_y * t
        distanceX = offset_X0 + speed_x * t
        alpha = alpha_initial - t * 2 * np.pi / T_satellite
        theta = np.pi / 2 - np.arctan(((orbit_height + r_Earth) * np.sin(alpha)) /
                                      ((orbit_height + r_Earth) * np.cos(alpha) - r_Earth))

        dPlane = np.sqrt(plane_height**2 + distanceX**2 + distanceY**2)
        sqrt_factor = np.sqrt(2 / wavelength / dPlane)
        distance = np.sqrt(distanceX**2 + distanceY**2)

        s1 = sqrt_factor * plane_size / 2
        s2 = sqrt_factor * (plane_size / 2 - distance)
        s3 = sqrt_factor * (plane_size / 2 + distance)
        C1, S1 = fresnel(s1)
        C2, S2 = fresnel(s2)
        C3, S3 = fresnel(s3)

        temp = 1 + 1j / 2 * (2 * C1 + 1j * 2 * S1) * (C2 + C3 + 1j * S2 + 1j * S3)
        raw_signal[m] = np.real(temp)  # Can be replaced with np.abs(temp) if more appropriate

    # ---------- Add Gaussian noise according to dB definition ----------
    P_signal = np.mean(raw_signal ** 2)
    noise_power = P_signal / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0.0, noise_std, size=raw_signal.shape)
    noisy_signal = raw_signal + noise

    # ---------- Mean Normalization ----------
    clean_reference = np.ones_like(raw_signal)  # Clean signal is ideally “1”
    normalized_signal = noisy_signal / (np.mean(clean_reference) + 1e-12)

    # ---------- Jitter (perturb time axis) ----------
    if np.random.rand() < 0.1:
        jittered_time_array = time_array + np.random.normal(0.0, 0.0001, size=time_array.shape)
        normalized_signal = np.interp(time_array, jittered_time_array, normalized_signal)

    # ---------- Convert to dB power ----------
    output_FN_F = 10 * np.log10(np.clip(normalized_signal ** 2, 1e-12, None))
    label = 'large'
    return {'data': output_FN_F, 'label': label, 'fz_bin': 2, 'snr': snr_db}


# ---------- Parallel Execution and Save to Multiple Files ----------
if __name__ == '__main__':
    start = time.time()
    grouped_results = {}

    with Pool(min(6, cpu_count())) as pool:
        results = pool.map(compute_instance, param_combinations)

    for r in results:
        key = (r['fz_bin'], r['snr'])
        if key not in grouped_results:
            grouped_results[key] = {'data': [], 'label': []}
        grouped_results[key]['data'].append(r['data'])
        grouped_results[key]['label'].append(r['label'])

    for (fz, snr), content in grouped_results.items():
        data = np.array(content['data'])
        labels = np.array(content['label'])
        filename = f"/content/drive/MyDrive/MPE_Research/Data/Train_new/FZDataset_Large_Fz{fz}_SNR{snr}.mat"
        savemat(filename, {
            'all_data': data,
            'all_labels': labels,
            'fz_bin': fz,
            'snr': snr_db
        })
        print(f"Saved file: {os.path.basename(filename)}, Samples: {len(data)}")

    print(f"Total time: {(time.time()-start)/60:.2f} minutes")

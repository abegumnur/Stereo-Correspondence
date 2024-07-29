import cv2
import numpy as np
from skimage.segmentation import slic, felzenszwalb
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
import os


# Function to load images
def load_images(folder):
    images = []
    for i in range(9):
        img = cv2.imread(f'{folder}/im{i}.ppm')
        if img is None:
            print(f"Image not found: {folder}/im{i}.ppm")
            continue
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images


# Function to perform oversegmentation
def oversegment(image, method='slic', **kwargs):
    if method == 'slic':
        n_segments = kwargs.get('n_segments', 250)
        compactness = kwargs.get('compactness', 10)
        image_lab = rgb2lab(image)
        segments = slic(image_lab, n_segments=n_segments, compactness=compactness)
    elif method == 'felzenszwalb':
        scale = kwargs.get('scale', 100)
        sigma = kwargs.get('sigma', 0.5)
        min_size = kwargs.get('min_size', 50)
        segments = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
    else:
        raise ValueError("Unsupported oversegmentation method")
    return segments


# Function to compute the Sum of Absolute Differences (SAD) between two color vectors
def compute_sad(color1, color2):
    return np.sum(np.abs(np.array(color1) - np.array(color2)))


# Function to match segments between left and right images
def match_segments(left_image, right_image, left_segments, right_segments):
    matches = []
    # Calculate mean color and row for each segment in the left image
    left_segment_means = {
        i: (
        cv2.mean(left_image, mask=(left_segments == i).astype(np.uint8))[:3], np.mean(np.where(left_segments == i)[0]))
        for i in np.unique(left_segments)
    }
    # Calculate mean color and row for each segment in the right image
    right_segment_means = {
        j: (cv2.mean(right_image, mask=(right_segments == j).astype(np.uint8))[:3],
            np.mean(np.where(right_segments == j)[0]))
        for j in np.unique(right_segments)
    }

    # Match segments between left and right images based on color similarity and row alignment
    for i, (left_mean, left_row) in left_segment_means.items():
        best_match = None
        best_sad = float('inf')
        for j, (right_mean, right_row) in right_segment_means.items():
            if abs(left_row - right_row) < 5:  # Ensure the match is on the epipolar line (or close to it)
                sad = compute_sad(left_mean, right_mean)  # Calculate SAD
                if sad < best_sad:
                    best_sad = sad
                    best_match = j
        if best_match is not None:
            matches.append((i, best_match))
    return matches


# Function to load a disparity map and scale it
def load_disparity_map(disp_path, scale_factor=8):
    disparity_map = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    if disparity_map is None:
        raise FileNotFoundError(f"Disparity map not found: {disp_path}")
    return disparity_map / scale_factor


# Function to compute the Mean Absolute Error (MAE) between estimated and ground truth disparity
def compute_disparity_error(estimated_disparity, ground_truth):
    error = np.abs(estimated_disparity - ground_truth)
    mae = np.mean(error)
    return mae


# Function to plot results
def plot_results(image, segments_slic, segments_felzenszwalb, ground_truth, disparity_slic, disparity_felzenszwalb,
                 title):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(segments_slic, cmap='viridis')  # Change colormap for segments
    axes[0, 1].set_title('SLIC Segments')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(segments_felzenszwalb, cmap='viridis')  # Change colormap for segments
    axes[0, 2].set_title('Felzenszwalb Segments')
    axes[0, 2].axis('off')

    im = axes[1, 0].imshow(ground_truth, cmap='viridis', vmin=4, vmax=16)  # Change colormap for disparity
    axes[1, 0].set_title('Ground Truth Disparity')
    axes[1, 0].axis('off')
    fig.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].imshow(disparity_slic, cmap='viridis', vmin=4, vmax=16)  # Change colormap for disparity
    axes[1, 1].set_title('SLIC Estimated Disparity')
    axes[1, 1].axis('off')
    fig.colorbar(im, ax=axes[1, 1])

    im = axes[1, 2].imshow(disparity_felzenszwalb, cmap='viridis', vmin=4, vmax=16)  # Change colormap for disparity
    axes[1, 2].set_title('Felzenszwalb Estimated Disparity')
    axes[1, 2].axis('off')
    fig.colorbar(im, ax=axes[1, 2])

    fig.suptitle(title)
    plt.show()


# Function to plot average MAE results
def plot_average_mae(results):
    datasets = list(results['SLIC'].keys())
    slic_mae = [results['SLIC'][dataset] for dataset in datasets]
    felzenszwalb_mae = [results['Felzenszwalb'][dataset] for dataset in datasets]

    x = np.arange(len(datasets))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, slic_mae, width, label='SLIC')
    rects2 = ax.bar(x + width / 2, felzenszwalb_mae, width, label='Felzenszwalb')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Datasets')
    ax.set_ylabel('MAE')
    ax.set_title('Average MAE by Dataset and Oversegmentation Method')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    fig.tight_layout()
    plt.show()


# Paths to datasets
datasets = ['barn1', 'barn2', 'bull', 'poster', 'sawtooth', 'venus']
base_path = 'C:/Users/BEGUM/PycharmProjects/CSE463/dataset'

# Define the ground-truth disparity maps
disparity_maps = {
    'im2.ppm': 'disp2.pgm',
    'im6.ppm': 'disp6.pgm'
}

# Initialize results dictionary
results = {'SLIC': {}, 'Felzenszwalb': {}}

# Process each dataset
for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    dataset_path = os.path.join(base_path, dataset)

    images = load_images(dataset_path)
    mae_slic_total = 0
    mae_felzenszwalb_total = 0
    count = 0

    for left_image_file, disparity_map_file in disparity_maps.items():
        print(f"Processing image: {left_image_file} with disparity map: {disparity_map_file}")
        ground_truth_path = os.path.join(dataset_path, disparity_map_file)

        # Load ground-truth disparity map
        ground_truth = load_disparity_map(ground_truth_path)

        for i in range(9):
            left_image = images[i]

            # Initialize the combined disparity map with zeros
            combined_disparity_slic = np.zeros_like(ground_truth, dtype=float)
            combined_disparity_felzenszwalb = np.zeros_like(ground_truth, dtype=float)
            count_map_slic = np.zeros_like(ground_truth, dtype=float)
            count_map_felzenszwalb = np.zeros_like(ground_truth, dtype=float)

            for j in range(9):
                if i == j:
                    continue
                right_image = images[j]

                # Apply SLIC segmentation
                left_segments_slic = oversegment(left_image, method='slic')
                right_segments_slic = oversegment(right_image, method='slic')

                # Apply Felzenszwalb segmentation
                left_segments_felzenszwalb = oversegment(left_image, method='felzenszwalb')
                right_segments_felzenszwalb = oversegment(right_image, method='felzenszwalb')

                # Run correspondence algorithm for SLIC
                matches_slic = match_segments(left_image, right_image, left_segments_slic, right_segments_slic)

                # Estimate disparity based on SLIC matches
                for left_seg, right_seg in matches_slic:
                    left_mask = (left_segments_slic == left_seg)
                    disp_value = np.mean(ground_truth[left_mask])
                    combined_disparity_slic[left_mask] += disp_value
                    count_map_slic[left_mask] += 1

                # Run correspondence algorithm for Felzenszwalb
                matches_felzenszwalb = match_segments(left_image, right_image, left_segments_felzenszwalb,
                                                      right_segments_felzenszwalb)

                # Estimate disparity based on Felzenszwalb matches
                for left_seg, right_seg in matches_felzenszwalb:
                    left_mask = (left_segments_felzenszwalb == left_seg)
                    disp_value = np.mean(ground_truth[left_mask])
                    combined_disparity_felzenszwalb[left_mask] += disp_value
                    count_map_felzenszwalb[left_mask] += 1

            # Compute the average disparity by dividing by the count map
            average_disparity_slic = np.divide(combined_disparity_slic, count_map_slic,
                                               out=np.zeros_like(combined_disparity_slic),
                                               where=count_map_slic != 0)
            average_disparity_felzenszwalb = np.divide(combined_disparity_felzenszwalb, count_map_felzenszwalb,
                                                       out=np.zeros_like(combined_disparity_felzenszwalb),
                                                       where=count_map_felzenszwalb != 0)

            # Compute the Mean Absolute Error (MAE) for SLIC
            mae_slic = compute_disparity_error(average_disparity_slic, ground_truth)
            mae_slic_total += mae_slic
            count += 1

            # Compute the Mean Absolute Error (MAE) for Felzenszwalb
            mae_felzenszwalb = compute_disparity_error(average_disparity_felzenszwalb, ground_truth)
            mae_felzenszwalb_total += mae_felzenszwalb

        # Compute and print the average MAE for the dataset
        if count > 0:
            average_mae_slic = mae_slic_total / count
            average_mae_felzenszwalb = mae_felzenszwalb_total / count
            results['SLIC'][dataset] = average_mae_slic
            results['Felzenszwalb'][dataset] = average_mae_felzenszwalb
            print(f'Average MAE for {dataset} using SLIC: {average_mae_slic}')
            print(f'Average MAE for {dataset} using Felzenszwalb: {average_mae_felzenszwalb}')

            # Plot the results for each dataset
            plot_results(images[0], left_segments_slic, left_segments_felzenszwalb, ground_truth,
                         average_disparity_slic, average_disparity_felzenszwalb, f'{dataset} Average Results')

# Plot average MAE results
plot_average_mae(results)

#!/usr/bin/env python3
"""
Test XOR/XNOR stereo vision on real stereo image pairs
Compares geometric approach against ground truth disparity
"""

import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import pearsonr

# Create test directory
TEST_DIR = Path("outputs/stereo_test")
TEST_DIR.mkdir(parents=True, exist_ok=True)


def download_middlebury_sample():
    """Download a sample stereo pair from Middlebury dataset"""
    print("Downloading Middlebury stereo test images...")

    base_url = "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/"
    files = {"left": "im2.png", "right": "im6.png", "disp": "disp2.png"}

    paths = {}
    for key, filename in files.items():
        url = base_url + filename
        save_path = TEST_DIR / f"cones_{key}.png"

        if not save_path.exists():
            try:
                print(f"  Fetching {filename}...")
                urllib.request.urlretrieve(url, save_path)
                paths[key] = save_path
            except Exception as e:
                print(f"  Error downloading {filename}: {e}")
                return None
        else:
            print(f"  Using cached {filename}")
            paths[key] = save_path

    return paths


def load_and_preprocess(img_path, target_size=None):
    """Load image and convert to grayscale"""
    img = Image.open(img_path).convert("L")
    if target_size:
        img = img.resize(target_size, Image.BILINEAR)
    return np.array(img, dtype=np.float32)


def binarize_edges(img, method="sobel", threshold_percentile=50):
    """
    Binarize image using edge detection

    Methods:
    - sobel: Sobel gradient magnitude
    - laplacian: Laplacian edge detection
    - simple: Simple gradient
    """
    if method == "sobel":
        sx = ndimage.sobel(img, axis=0)
        sy = ndimage.sobel(img, axis=1)
        edges = np.hypot(sx, sy)
    elif method == "laplacian":
        edges = np.abs(ndimage.laplace(img))
    elif method == "simple":
        edges = np.abs(np.gradient(img)[0]) + np.abs(np.gradient(img)[1])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Threshold at percentile
    threshold = np.percentile(edges, threshold_percentile)
    binary = (edges > threshold).astype(np.uint8)

    return binary, edges


def compute_xor_xnor_stereo(left_binary, right_binary, max_disparity=64):
    """
    Apply XOR/XNOR kernel across different horizontal shifts
    to find best correspondence (disparity)
    """
    h, w = left_binary.shape

    # For each disparity level, compute XOR and XNOR
    disparity_map = np.zeros((h, w), dtype=np.float32)
    confidence_map = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            best_disp = 0
            best_score = -1

            # Search for best match in right image
            for d in range(min(max_disparity, x)):
                # Compare left[x] with right[x-d]
                if x - d >= 0:
                    # Get local window
                    window_size = 5
                    y_start = max(0, y - window_size // 2)
                    y_end = min(h, y + window_size // 2 + 1)
                    x_start = max(0, x - window_size // 2)
                    x_end = min(w, x + window_size // 2 + 1)

                    left_window = left_binary[y_start:y_end, x_start:x_end]

                    x_right_start = max(0, x - d - window_size // 2)
                    x_right_end = min(w, x - d + window_size // 2 + 1)
                    right_window = right_binary[y_start:y_end, x_right_start:x_right_end]

                    # Make windows same size
                    min_h = min(left_window.shape[0], right_window.shape[0])
                    min_w = min(left_window.shape[1], right_window.shape[1])

                    if min_h > 0 and min_w > 0:
                        left_window = left_window[:min_h, :min_w]
                        right_window = right_window[:min_h, :min_w]

                        # XNOR (agreement) - higher is better match
                        xnor = np.logical_not(np.logical_xor(left_window, right_window))
                        score = np.mean(xnor)

                        if score > best_score:
                            best_score = score
                            best_disp = d

            disparity_map[y, x] = best_disp
            confidence_map[y, x] = best_score

    return disparity_map, confidence_map


def evaluate_results(predicted_disp, ground_truth_disp, confidence_map=None):
    """
    Compare predicted disparity against ground truth
    """
    # Resize ground truth to match predicted if needed
    if predicted_disp.shape != ground_truth_disp.shape:
        from PIL import Image

        gt_img = Image.fromarray(ground_truth_disp)
        gt_img = gt_img.resize((predicted_disp.shape[1], predicted_disp.shape[0]), Image.BILINEAR)
        ground_truth_disp = np.array(gt_img, dtype=np.float32)

    # Mask out invalid pixels (disparity = 0 in ground truth often means invalid)
    valid_mask = ground_truth_disp > 0

    if confidence_map is not None:
        # Only use high-confidence predictions
        conf_threshold = np.percentile(confidence_map[valid_mask], 25)
        valid_mask = valid_mask & (confidence_map > conf_threshold)

    pred_valid = predicted_disp[valid_mask]
    gt_valid = ground_truth_disp[valid_mask]

    # Compute metrics
    mae = np.mean(np.abs(pred_valid - gt_valid))
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))

    # Correlation
    if len(pred_valid) > 0:
        correlation, p_value = pearsonr(pred_valid.flatten(), gt_valid.flatten())
    else:
        correlation, p_value = 0, 1

    # Percentage of pixels within threshold
    threshold = 3  # pixels
    accuracy = np.mean(np.abs(pred_valid - gt_valid) < threshold) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "p_value": p_value,
        "accuracy_3px": accuracy,
        "n_valid": len(pred_valid),
    }


def visualize_results(
    left_img,
    right_img,
    left_binary,
    right_binary,
    predicted_disp,
    ground_truth_disp,
    confidence_map,
    metrics,
):
    """Create comprehensive visualization"""

    fig = plt.figure(figsize=(16, 12))

    # Original images
    plt.subplot(3, 3, 1)
    plt.imshow(left_img, cmap="gray")
    plt.title("Left Image")
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.imshow(right_img, cmap="gray")
    plt.title("Right Image")
    plt.axis("off")

    # Ground truth
    plt.subplot(3, 3, 3)
    plt.imshow(ground_truth_disp, cmap="jet")
    plt.colorbar()
    plt.title("Ground Truth Disparity")
    plt.axis("off")

    # Binarized edges
    plt.subplot(3, 3, 4)
    plt.imshow(left_binary, cmap="gray")
    plt.title("Left Binary (Edges)")
    plt.axis("off")

    plt.subplot(3, 3, 5)
    plt.imshow(right_binary, cmap="gray")
    plt.title("Right Binary (Edges)")
    plt.axis("off")

    # XOR visualization
    xor_viz = np.logical_xor(left_binary, right_binary).astype(np.uint8)
    plt.subplot(3, 3, 6)
    plt.imshow(xor_viz, cmap="gray")
    plt.title("Z = L XOR R (Disparity Field)")
    plt.axis("off")

    # Predicted disparity
    plt.subplot(3, 3, 7)
    plt.imshow(predicted_disp, cmap="jet")
    plt.colorbar()
    plt.title("XOR/XNOR Predicted Disparity")
    plt.axis("off")

    # Confidence map
    plt.subplot(3, 3, 8)
    plt.imshow(confidence_map, cmap="viridis")
    plt.colorbar()
    plt.title("XNOR Confidence (Agreement)")
    plt.axis("off")

    # Error map
    plt.subplot(3, 3, 9)
    if predicted_disp.shape == ground_truth_disp.shape:
        error = np.abs(predicted_disp - ground_truth_disp)
        error[ground_truth_disp == 0] = 0  # Mask invalid
        plt.imshow(error, cmap="hot", vmin=0, vmax=10)
        plt.colorbar()
        plt.title(f'Absolute Error\nMAE={metrics["mae"]:.2f} px')
    else:
        plt.text(0.5, 0.5, "Size mismatch", ha="center", va="center")
    plt.axis("off")

    plt.suptitle(
        f"XOR/XNOR Stereo Vision Test\n"
        + f'Correlation: {metrics["correlation"]:.3f} | '
        + f'RMSE: {metrics["rmse"]:.2f}px | '
        + f'3px Accuracy: {metrics["accuracy_3px"]:.1f}%',
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Testing XOR/XNOR Geometric Stereo Vision")
    print("=" * 60)

    # Download test data
    paths = download_middlebury_sample()
    if paths is None:
        print("\n❌ Failed to download test images")
        return

    print("\n✓ Test images ready")

    # Load images
    print("\nLoading and preprocessing images...")
    left_img = load_and_preprocess(paths["left"], target_size=(200, 300))
    right_img = load_and_preprocess(paths["right"], target_size=(200, 300))
    ground_truth = load_and_preprocess(paths["disp"])

    print(f"  Image size: {left_img.shape}")

    # Binarize using edge detection
    print("\nBinarizing edges...")
    left_binary, left_edges = binarize_edges(left_img, method="sobel", threshold_percentile=60)
    right_binary, right_edges = binarize_edges(right_img, method="sobel", threshold_percentile=60)
    print(f"  Left edges: {np.sum(left_binary)} pixels")
    print(f"  Right edges: {np.sum(right_binary)} pixels")

    # Apply XOR/XNOR stereo matching
    print("\nApplying XOR/XNOR stereo kernel...")
    print("  (This may take a minute for dense matching...)")
    predicted_disp, confidence_map = compute_xor_xnor_stereo(
        left_binary, right_binary, max_disparity=40
    )
    print(f"  Disparity range: {predicted_disp.min():.1f} - {predicted_disp.max():.1f} px")

    # Evaluate
    print("\nEvaluating against ground truth...")
    metrics = evaluate_results(predicted_disp, ground_truth, confidence_map)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"  Correlation:      {metrics['correlation']:.4f} (p={metrics['p_value']:.2e})")
    print(f"  MAE:              {metrics['mae']:.2f} pixels")
    print(f"  RMSE:             {metrics['rmse']:.2f} pixels")
    print(f"  3px Accuracy:     {metrics['accuracy_3px']:.1f}%")
    print(f"  Valid pixels:     {metrics['n_valid']:,}")

    # Interpret results
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)

    if metrics["correlation"] > 0.5:
        print("✓ STRONG correlation with ground truth!")
        print("  → XOR/XNOR approach captures disparity structure")
    elif metrics["correlation"] > 0.3:
        print("≈ MODERATE correlation with ground truth")
        print("  → Approach has signal but needs refinement")
    elif metrics["correlation"] > 0.1:
        print("⚠ WEAK correlation with ground truth")
        print("  → Limited correspondence matching ability")
    else:
        print("✗ NO meaningful correlation")
        print("  → XOR/XNOR on edges doesn't recover disparity well")

    print(f"\nFor reference, state-of-the-art stereo achieves ~95%+ at 3px threshold")

    # Visualize
    print("\nGenerating visualization...")
    fig = visualize_results(
        left_img,
        right_img,
        left_binary,
        right_binary,
        predicted_disp,
        ground_truth,
        confidence_map,
        metrics,
    )

    output_path = TEST_DIR / "xor_xnor_stereo_results.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    # Save metrics
    metrics_path = TEST_DIR / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("XOR/XNOR Stereo Vision Test Results\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"  Saved: {metrics_path}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

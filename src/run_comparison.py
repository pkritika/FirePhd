"""
Compare Baseline vs Improved SAM2 Fire Segmentation

This script runs both approaches side-by-side and creates comparison visualizations.
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fire_segmentation_baseline import FireSegmentationBaseline
from fire_segmentation_improved import FireSegmentationImproved


def compare_segmentations(infrared_video, visible_video, checkpoint_path, output_dir):
    """
    Run both baseline and improved segmentation and compare results.

    Args:
        infrared_video: Path to infrared/thermal video
        visible_video: Path to visible-light video
        checkpoint_path: Path to SAM2 checkpoint
        output_dir: Directory to save comparison results
    """
    print("=" * 80)
    print("BASELINE APPROACH: Color-based positive points only")
    print("=" * 80)

    baseline_dir = os.path.join(output_dir, "baseline")
    baseline = FireSegmentationBaseline(checkpoint_path)
    baseline_segments = baseline.segment_video(
        infrared_video,
        baseline_dir,
        num_points=15
    )

    print("\n" + "=" * 80)
    print("IMPROVED APPROACH: With vegetation detection and negative points")
    print("=" * 80)

    improved_dir = os.path.join(output_dir, "improved")
    improved = FireSegmentationImproved(checkpoint_path)
    improved_segments = improved.segment_video(
        infrared_video,
        visible_video,
        improved_dir,
        num_fire_points=15,
        num_veg_points=15,
        num_bg_points=10
    )

    # Create side-by-side comparison
    print("\n" + "=" * 80)
    print("Creating comparison visualizations...")
    print("=" * 80)

    create_comparison_visualizations(
        baseline_dir,
        improved_dir,
        baseline_segments,
        improved_segments,
        output_dir
    )

    print(f"\nComparison complete! Results saved to {output_dir}")


def create_comparison_visualizations(baseline_dir, improved_dir,
                                     baseline_segments, improved_segments,
                                     output_dir):
    """
    Create side-by-side comparison of baseline vs improved results.

    Args:
        baseline_dir: Directory with baseline results
        improved_dir: Directory with improved results
        baseline_segments: Baseline segmentation masks
        improved_segments: Improved segmentation masks
        output_dir: Output directory
    """
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Get frame paths
    baseline_frames = sorted(Path(baseline_dir).glob("frames/frame_*.jpg"))
    improved_thermal = sorted(Path(improved_dir).glob("thermal_frames/frame_*.jpg"))
    improved_visible = sorted(Path(improved_dir).glob("visible_frames/frame_*.jpg"))

    for idx, (b_frame, i_thermal, i_visible) in enumerate(zip(baseline_frames, improved_thermal, improved_visible)):
        if idx not in baseline_segments or idx not in improved_segments:
            continue

        # Read frames
        baseline_frame = cv2.imread(str(b_frame))
        baseline_frame_rgb = cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2RGB)

        thermal_frame = cv2.imread(str(i_thermal))
        thermal_rgb = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2RGB)

        visible_frame = cv2.imread(str(i_visible))
        visible_rgb = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2RGB)

        # Get masks
        baseline_mask = baseline_segments[idx][1][0] if 1 in baseline_segments[idx] else None
        improved_mask = improved_segments[idx][1][0] if 1 in improved_segments[idx] else None

        if baseline_mask is None or improved_mask is None:
            continue

        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Row 1: Baseline
        axes[0, 0].imshow(baseline_frame_rgb)
        axes[0, 0].set_title(f"Thermal Frame {idx}")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(baseline_mask, cmap='gray')
        axes[0, 1].set_title("BASELINE: Mask (Positive Points Only)")
        axes[0, 1].axis('off')

        overlay_baseline = baseline_frame_rgb.copy()
        overlay_baseline[baseline_mask] = overlay_baseline[baseline_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[0, 2].imshow(overlay_baseline.astype(np.uint8))
        axes[0, 2].set_title("BASELINE: Overlay")
        axes[0, 2].axis('off')

        # Row 2: Improved
        axes[1, 0].imshow(visible_rgb)
        axes[1, 0].set_title(f"Visible Light Frame {idx}")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(improved_mask, cmap='gray')
        axes[1, 1].set_title("IMPROVED: Mask (With Negative Points)")
        axes[1, 1].axis('off')

        overlay_improved = thermal_rgb.copy()
        overlay_improved[improved_mask] = overlay_improved[improved_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[1, 2].imshow(overlay_improved.astype(np.uint8))
        axes[1, 2].set_title("IMPROVED: Overlay")
        axes[1, 2].axis('off')

        # Add difference visualization
        fig.suptitle(f"Frame {idx}: Baseline vs Improved Comparison", fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f"comparison_{idx:05d}.jpg"), dpi=100)
        plt.close()

        # Create difference map
        create_difference_visualization(baseline_mask, improved_mask, idx, comparison_dir)

    print(f"Comparison visualizations saved to {comparison_dir}")


def create_difference_visualization(baseline_mask, improved_mask, frame_idx, output_dir):
    """
    Create visualization showing differences between baseline and improved masks.

    Args:
        baseline_mask: Baseline segmentation mask
        improved_mask: Improved segmentation mask
        frame_idx: Frame index
        output_dir: Output directory
    """
    # Calculate differences
    # False positives (baseline): In baseline but not in improved (likely vegetation)
    false_positives = baseline_mask & ~improved_mask

    # False negatives (baseline): In improved but not in baseline (missed fire)
    false_negatives = ~baseline_mask & improved_mask

    # True positives: In both
    true_positives = baseline_mask & improved_mask

    # Create RGB difference map
    diff_map = np.zeros((*baseline_mask.shape, 3), dtype=np.uint8)
    diff_map[true_positives] = [0, 255, 0]      # Green: Correct fire detection
    diff_map[false_positives] = [255, 0, 0]     # Red: False positives (vegetation)
    diff_map[false_negatives] = [0, 0, 255]     # Blue: Missed fire

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(baseline_mask, cmap='gray')
    axes[0].set_title("Baseline Mask")
    axes[0].axis('off')

    axes[1].imshow(improved_mask, cmap='gray')
    axes[1].set_title("Improved Mask")
    axes[1].axis('off')

    axes[2].imshow(diff_map)
    axes[2].set_title("Difference (Green=Correct, Red=False+, Blue=Missed)")
    axes[2].axis('off')

    # Calculate metrics
    fp_count = np.sum(false_positives)
    fn_count = np.sum(false_negatives)
    tp_count = np.sum(true_positives)

    fig.suptitle(
        f"Frame {frame_idx} Difference Analysis\n"
        f"True Positives: {tp_count} | False Positives Removed: {fp_count} | Missed Fire: {fn_count}",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"difference_{frame_idx:05d}.jpg"), dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs improved SAM2 fire segmentation")
    parser.add_argument("--infrared", type=str, required=True, help="Path to infrared video")
    parser.add_argument("--visible", type=str, required=True, help="Path to visible-light video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    compare_segmentations(
        args.infrared,
        args.visible,
        args.checkpoint,
        args.output
    )


if __name__ == "__main__":
    # Example usage when run directly
    checkpoint_path = "/Users/kritikapant/Desktop/PHD/Refined Sam/checkpoints/sam2_hiera_small.pt"
    infrared_video = "/Users/kritikapant/Desktop/PHD/Refined Sam/FireSentry-Benchmark-Dataset/Region A/Infrared Videos/Fire0.mp4"
    visible_video = "/Users/kritikapant/Desktop/PHD/Refined Sam/FireSentry-Benchmark-Dataset/Region A/Visible Light/Fire0.avi"
    output_dir = "/Users/kritikapant/Desktop/PHD/Refined Sam/results/comparison"

    compare_segmentations(infrared_video, visible_video, checkpoint_path, output_dir)

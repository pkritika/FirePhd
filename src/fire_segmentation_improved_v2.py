"""
Improved SAM2 Fire Segmentation with Vegetation Detection (V2)
Handles both video files and frame sequences for visible-light data.
"""

import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


class FireSegmentationImprovedV2:
    """
    Improved fire segmentation using SAM2 with both positive and negative points.

    This version handles:
    - Thermal video (MP4) with hot region detection → positive points
    - Visible-light frames (JPG sequence) with vegetation detection → negative points
    """

    def __init__(self, checkpoint_path, model_config="configs/sam2.1/sam2.1_hiera_s.yaml"):
        """Initialize SAM2 video predictor."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.predictor = build_sam2_video_predictor(
            model_config,
            checkpoint_path,
            device=self.device
        )

    def extract_frames_from_video(self, video_path, output_dir, max_frames=None):
        """Extract frames from video file."""
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            frame_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_idx += 1

        cap.release()
        print(f"Extracted {len(frame_paths)} frames from {video_path}")
        return frame_paths

    def copy_visible_frames(self, visible_frames_dir, output_dir, num_frames=None):
        """
        Copy visible-light frames to output directory with consistent naming.

        Args:
            visible_frames_dir: Directory containing visible light frames (JPG)
            output_dir: Destination directory
            num_frames: Number of frames to copy (None = all)

        Returns:
            List of frame paths
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get all JPG files sorted by name
        visible_frames = sorted(Path(visible_frames_dir).glob("*.jpg"))

        if num_frames:
            visible_frames = visible_frames[:num_frames]

        frame_paths = []
        for idx, src_path in enumerate(visible_frames):
            dst_path = os.path.join(output_dir, f"{idx:05d}.jpg")
            shutil.copy2(src_path, dst_path)
            frame_paths.append(dst_path)

        print(f"Copied {len(frame_paths)} visible light frames to {output_dir}")
        return frame_paths

    def detect_fire_points_thermal(self, thermal_frame, num_points=10):
        """Detect fire regions from thermal frame using brightness."""
        gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)

        # Find hot regions
        threshold = np.percentile(gray, 95)  # Top 5% hottest
        hot_mask = gray > threshold

        hot_coords = np.argwhere(hot_mask)

        if len(hot_coords) == 0:
            return np.array([])

        # Sample points
        if len(hot_coords) > num_points:
            indices = np.random.choice(len(hot_coords), num_points, replace=False)
            sampled_coords = hot_coords[indices]
        else:
            sampled_coords = hot_coords

        # Convert from (row, col) to (x, y)
        points = np.column_stack([sampled_coords[:, 1], sampled_coords[:, 0]])

        return points

    def detect_vegetation_points(self, visible_frame, num_points=10):
        """
        Detect vegetation using green color in visible-light frame.

        Returns NEGATIVE points to exclude vegetation from segmentation.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2HSV)

        # Green vegetation range
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        vegetation_mask = cv2.inRange(hsv, lower_green, upper_green)

        veg_coords = np.argwhere(vegetation_mask > 0)

        if len(veg_coords) == 0:
            return np.array([])

        # Sample points from vegetation
        if len(veg_coords) > num_points:
            indices = np.random.choice(len(veg_coords), num_points, replace=False)
            sampled_coords = veg_coords[indices]
        else:
            sampled_coords = veg_coords

        points = np.column_stack([sampled_coords[:, 1], sampled_coords[:, 0]])

        return points

    def detect_background_points(self, visible_frame, thermal_frame, num_points=5):
        """Detect background regions (cold + not vegetation)."""
        # Find cold regions in thermal
        gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        cold_threshold = np.percentile(gray, 30)
        cold_mask = gray < cold_threshold

        # Find non-green regions in visible
        hsv = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        not_vegetation_mask = cv2.inRange(hsv, lower_green, upper_green) == 0

        # Background = cold AND not vegetation
        background_mask = cold_mask & not_vegetation_mask

        bg_coords = np.argwhere(background_mask)

        if len(bg_coords) == 0:
            return np.array([])

        if len(bg_coords) > num_points:
            indices = np.random.choice(len(bg_coords), num_points, replace=False)
            sampled_coords = bg_coords[indices]
        else:
            sampled_coords = bg_coords

        points = np.column_stack([sampled_coords[:, 1], sampled_coords[:, 0]])

        return points

    def combine_points_and_labels(self, fire_points, vegetation_points, background_points):
        """
        Combine positive (fire) and negative (vegetation, background) points.

        Returns:
            points: Combined (x, y) coordinates
            labels: 1=fire (positive), 0=not-fire (negative)
        """
        all_points = []
        all_labels = []

        if len(fire_points) > 0:
            all_points.append(fire_points)
            all_labels.extend([1] * len(fire_points))

        if len(vegetation_points) > 0:
            all_points.append(vegetation_points)
            all_labels.extend([0] * len(vegetation_points))

        if len(background_points) > 0:
            all_points.append(background_points)
            all_labels.extend([0] * len(background_points))

        if len(all_points) == 0:
            return np.array([]), np.array([])

        points = np.vstack(all_points)
        labels = np.array(all_labels, dtype=np.int32)

        return points, labels

    def segment_video(self, infrared_video_path, visible_frames_dir, output_dir,
                     num_fire_points=10, num_veg_points=10, num_bg_points=5,
                     max_frames=50):
        """
        Segment fire using thermal video + visible frame sequence.

        Args:
            infrared_video_path: Path to thermal video (MP4)
            visible_frames_dir: Directory with visible light frames (JPG sequence)
            output_dir: Output directory
            num_fire_points: Number of positive fire points
            num_veg_points: Number of negative vegetation points
            num_bg_points: Number of negative background points
            max_frames: Maximum frames to process (to save time/memory)
        """
        # Extract thermal frames
        thermal_frames_dir = os.path.join(output_dir, "thermal_frames")
        thermal_frame_paths = self.extract_frames_from_video(
            infrared_video_path, thermal_frames_dir, max_frames
        )

        # Copy visible frames with matching count
        visible_output_dir = os.path.join(output_dir, "visible_frames")
        visible_frame_paths = self.copy_visible_frames(
            visible_frames_dir, visible_output_dir, len(thermal_frame_paths)
        )

        # Verify same number of frames
        assert len(thermal_frame_paths) == len(visible_frame_paths), \
            f"Frame count mismatch: thermal={len(thermal_frame_paths)}, visible={len(visible_frame_paths)}"

        print(f"\nProcessing {len(thermal_frame_paths)} frames")

        # Initialize SAM2 with thermal frames
        inference_state = self.predictor.init_state(video_path=thermal_frames_dir)

        # Read first frames
        first_thermal = cv2.imread(thermal_frame_paths[0])
        first_visible = cv2.imread(visible_frame_paths[0])

        # Resize visible to match thermal dimensions
        thermal_h, thermal_w = first_thermal.shape[:2]
        first_visible = cv2.resize(first_visible, (thermal_w, thermal_h))

        # Detect points
        print("\nDetecting point prompts...")
        fire_points = self.detect_fire_points_thermal(first_thermal, num_fire_points)
        vegetation_points = self.detect_vegetation_points(first_visible, num_veg_points)
        background_points = self.detect_background_points(first_visible, first_thermal, num_bg_points)

        # Combine points
        points, labels = self.combine_points_and_labels(fire_points, vegetation_points, background_points)

        print(f"\nPoint Summary:")
        print(f"  Fire points (positive): {len(fire_points)}")
        print(f"  Vegetation points (negative): {len(vegetation_points)}")
        print(f"  Background points (negative): {len(background_points)}")
        print(f"  Total: {len(points)} points")

        if len(points) == 0:
            print("ERROR: No points detected! Check thresholds.")
            return {}

        # Add points to SAM2
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )

        # Propagate through video
        print("\nPropagating segmentation...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Visualize
        self.visualize_results(thermal_frame_paths, visible_frame_paths,
                              video_segments, points, labels, output_dir)

        return video_segments

    def visualize_results(self, thermal_paths, visible_paths, segments,
                         points, labels, output_dir):
        """Create visualizations comparing thermal and visible with segmentation."""
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Sample every 5th frame to avoid too many images
        frame_indices = list(range(0, len(thermal_paths), 5))

        for frame_idx in frame_indices:
            if frame_idx not in segments:
                continue

            thermal = cv2.imread(thermal_paths[frame_idx])
            thermal_rgb = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)

            visible = cv2.imread(visible_paths[frame_idx])
            # Resize visible to match thermal
            thermal_h, thermal_w = thermal.shape[:2]
            visible = cv2.resize(visible, (thermal_w, thermal_h))
            visible_rgb = cv2.cvtColor(visible, cv2.COLOR_BGR2RGB)

            masks = segments[frame_idx]
            if 1 not in masks:
                continue

            mask = masks[1][0]

            # Create figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # Row 1: Thermal
            axes[0, 0].imshow(thermal_rgb)
            axes[0, 0].set_title(f"Thermal Frame {frame_idx}", fontsize=12)
            axes[0, 0].axis('off')

            axes[0, 1].imshow(mask, cmap='hot')
            axes[0, 1].set_title("Fire Segmentation Mask", fontsize=12)
            axes[0, 1].axis('off')

            overlay_thermal = thermal_rgb.copy()
            overlay_thermal[mask] = overlay_thermal[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
            axes[0, 2].imshow(overlay_thermal.astype(np.uint8))
            axes[0, 2].set_title("Thermal + Fire Overlay", fontsize=12)
            axes[0, 2].axis('off')

            # Row 2: Visible
            axes[1, 0].imshow(visible_rgb)
            axes[1, 0].set_title(f"Visible Light Frame {frame_idx}", fontsize=12)
            axes[1, 0].axis('off')

            # Show points on first frame only
            if frame_idx == 0:
                visible_with_points = visible_rgb.copy()
                for pt, label in zip(points, labels):
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < visible_with_points.shape[1] and 0 <= y < visible_with_points.shape[0]:
                        color = (0, 255, 0) if label == 1 else (255, 0, 0)
                        cv2.circle(visible_with_points, (x, y), 7, color, -1)
                        cv2.circle(visible_with_points, (x, y), 8, (255, 255, 255), 2)  # White outline
                axes[1, 1].imshow(visible_with_points)
                axes[1, 1].set_title("Point Prompts\n(Green=Fire, Red=Not-Fire)", fontsize=12)
            else:
                axes[1, 1].imshow(mask, cmap='hot')
                axes[1, 1].set_title("Fire Mask", fontsize=12)
            axes[1, 1].axis('off')

            overlay_visible = visible_rgb.copy()
            overlay_visible[mask] = overlay_visible[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
            axes[1, 2].imshow(overlay_visible.astype(np.uint8))
            axes[1, 2].set_title("Visible + Fire Overlay", fontsize=12)
            axes[1, 2].axis('off')

            fire_px = np.sum(mask)
            fig.suptitle(
                f"Improved Fire Segmentation - Frame {frame_idx}\n"
                f"Fire pixels detected: {fire_px:,}",
                fontsize=14, fontweight='bold'
            )

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"result_{frame_idx:05d}.jpg"), dpi=100)
            plt.close()

        print(f"\nVisualizations saved to {vis_dir}")


def main():
    """Test the improved V2 implementation."""
    checkpoint_path = "/Users/kritikapant/Desktop/PHD/Refined Sam/checkpoints/sam2_hiera_small.pt"

    # Use first thermal video
    infrared_video = "/Users/kritikapant/Desktop/PHD/Refined Sam/FireSentry-Benchmark-Dataset/Region A/Infrared Videos/video_001.mp4"

    # Use visible light frame sequence
    visible_frames_dir = "/Users/kritikapant/Desktop/PHD/Refined Sam/FireSentry-Benchmark-Dataset/Region A/Visible Light"

    output_dir = "/Users/kritikapant/Desktop/PHD/Refined Sam/results/improved_v2"

    print("=" * 80)
    print("IMPROVED SAM2 FIRE SEGMENTATION")
    print("Using thermal video + visible light frames")
    print("=" * 80)

    segmenter = FireSegmentationImprovedV2(checkpoint_path)
    segments = segmenter.segment_video(
        infrared_video,
        visible_frames_dir,
        output_dir,
        num_fire_points=15,
        num_veg_points=15,
        num_bg_points=10,
        max_frames=50  # Process first 50 frames
    )

    print(f"\nSegmentation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Check visualizations in: {output_dir}/visualizations/")


if __name__ == "__main__":
    main()

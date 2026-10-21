"""
Improved SAM2 Fire Segmentation with Vegetation Detection
Uses both thermal and visible-light videos to provide better prompts to SAM2.
"""

import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
from pathlib import Path


class FireSegmentationImproved:
    """
    Improved fire segmentation using SAM2 with both positive and negative points.

    This approach uses:
    - Thermal video: Detect hot regions (positive points for fire)
    - Visible-light video: Detect vegetation/background (negative points)

    This should reduce false positives where vegetation is incorrectly identified as fire.
    """

    def __init__(self, checkpoint_path, model_config="configs/sam2.1/sam2.1_hiera_s.yaml"):
        """
        Initialize SAM2 video predictor.

        Args:
            checkpoint_path: Path to SAM2 model checkpoint
            model_config: Model configuration name
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Build SAM2 video predictor
        self.predictor = build_sam2_video_predictor(
            model_config,
            checkpoint_path,
            device=self.device
        )

    def extract_frames_from_video(self, video_path, output_dir):
        """
        Extract frames from video file.

        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames

        Returns:
            List of frame paths
        """
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_idx += 1

        cap.release()
        print(f"Extracted {len(frame_paths)} frames from {video_path}")
        return frame_paths

    def detect_fire_points_thermal(self, thermal_frame, num_points=10):
        """
        Detect fire regions from thermal/infrared frame.

        Uses brightness/temperature as indicator of fire.

        Args:
            thermal_frame: Thermal frame (BGR format)
            num_points: Number of positive points to sample

        Returns:
            points: Array of (x, y) coordinates for fire regions
        """
        # Convert to grayscale to analyze intensity
        gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)

        # Find very hot regions (top percentile)
        threshold = np.percentile(gray, 95)  # Top 5% hottest pixels
        hot_mask = gray > threshold

        # Get coordinates of hot pixels
        hot_coords = np.argwhere(hot_mask)

        if len(hot_coords) == 0:
            # No hot regions, return empty
            return np.array([])

        # Sample points from hot regions
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
        Detect vegetation regions from visible-light frame.

        Uses green color detection (vegetation has high green channel values).
        These will be used as NEGATIVE points to tell SAM2 to exclude vegetation.

        Args:
            visible_frame: Visible-light frame (BGR format)
            num_points: Number of negative points to sample

        Returns:
            points: Array of (x, y) coordinates for vegetation regions
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2HSV)

        # Define green color range for vegetation
        # Hue: 35-85 (green range)
        # Saturation: 40-255 (avoid very desaturated/grayish)
        # Value: 40-255 (avoid very dark)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        # Create mask for green regions (vegetation)
        vegetation_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Get coordinates of vegetation pixels
        veg_coords = np.argwhere(vegetation_mask > 0)

        if len(veg_coords) == 0:
            # No vegetation detected, return empty
            return np.array([])

        # Sample points from vegetation regions
        if len(veg_coords) > num_points:
            indices = np.random.choice(len(veg_coords), num_points, replace=False)
            sampled_coords = veg_coords[indices]
        else:
            sampled_coords = veg_coords

        # Convert from (row, col) to (x, y)
        points = np.column_stack([sampled_coords[:, 1], sampled_coords[:, 0]])

        return points

    def detect_background_points(self, visible_frame, thermal_frame, num_points=5):
        """
        Detect background regions (non-fire, non-vegetation).

        Uses regions that are:
        - Not hot in thermal (low temperature)
        - Not green in visible (not vegetation)

        Args:
            visible_frame: Visible-light frame
            thermal_frame: Thermal frame
            num_points: Number of background negative points

        Returns:
            points: Array of (x, y) coordinates for background
        """
        # Find cold regions in thermal
        gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        cold_threshold = np.percentile(gray, 30)  # Bottom 70% coldest
        cold_mask = gray < cold_threshold

        # Find non-green regions in visible
        hsv = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        not_vegetation_mask = cv2.inRange(hsv, lower_green, upper_green) == 0

        # Background = cold AND not vegetation
        background_mask = cold_mask & not_vegetation_mask

        # Get coordinates
        bg_coords = np.argwhere(background_mask)

        if len(bg_coords) == 0:
            return np.array([])

        # Sample points
        if len(bg_coords) > num_points:
            indices = np.random.choice(len(bg_coords), num_points, replace=False)
            sampled_coords = bg_coords[indices]
        else:
            sampled_coords = bg_coords

        # Convert from (row, col) to (x, y)
        points = np.column_stack([sampled_coords[:, 1], sampled_coords[:, 0]])

        return points

    def combine_points_and_labels(self, fire_points, vegetation_points, background_points):
        """
        Combine positive (fire) and negative (vegetation, background) points.

        Args:
            fire_points: Positive points from fire detection
            vegetation_points: Negative points from vegetation detection
            background_points: Negative points from background

        Returns:
            points: Combined array of (x, y) coordinates
            labels: Array of labels (1=positive/fire, 0=negative/not-fire)
        """
        all_points = []
        all_labels = []

        # Add fire points (positive, label=1)
        if len(fire_points) > 0:
            all_points.append(fire_points)
            all_labels.extend([1] * len(fire_points))

        # Add vegetation points (negative, label=0)
        if len(vegetation_points) > 0:
            all_points.append(vegetation_points)
            all_labels.extend([0] * len(vegetation_points))

        # Add background points (negative, label=0)
        if len(background_points) > 0:
            all_points.append(background_points)
            all_labels.extend([0] * len(background_points))

        if len(all_points) == 0:
            return np.array([]), np.array([])

        points = np.vstack(all_points)
        labels = np.array(all_labels, dtype=np.int32)

        return points, labels

    def segment_video(self, infrared_video_path, visible_video_path, output_dir,
                     num_fire_points=10, num_veg_points=10, num_bg_points=5):
        """
        Segment fire in video using improved approach with visible-light reference.

        Args:
            infrared_video_path: Path to infrared/thermal video
            visible_video_path: Path to corresponding visible-light video
            output_dir: Directory to save results
            num_fire_points: Number of positive fire points
            num_veg_points: Number of negative vegetation points
            num_bg_points: Number of negative background points
        """
        # Extract frames from both videos
        thermal_frames_dir = os.path.join(output_dir, "thermal_frames")
        visible_frames_dir = os.path.join(output_dir, "visible_frames")

        thermal_frame_paths = self.extract_frames_from_video(infrared_video_path, thermal_frames_dir)
        visible_frame_paths = self.extract_frames_from_video(visible_video_path, visible_frames_dir)

        # Verify same number of frames
        assert len(thermal_frame_paths) == len(visible_frame_paths), \
            "Thermal and visible videos must have same number of frames"

        # Initialize inference state with thermal frames
        inference_state = self.predictor.init_state(video_path=thermal_frames_dir)

        # Read first frames
        first_thermal = cv2.imread(thermal_frame_paths[0])
        first_visible = cv2.imread(visible_frame_paths[0])

        # Detect points
        print("Detecting fire points from thermal video...")
        fire_points = self.detect_fire_points_thermal(first_thermal, num_fire_points)

        print("Detecting vegetation points from visible-light video...")
        vegetation_points = self.detect_vegetation_points(first_visible, num_veg_points)

        print("Detecting background points...")
        background_points = self.detect_background_points(first_visible, first_thermal, num_bg_points)

        # Combine points
        points, labels = self.combine_points_and_labels(fire_points, vegetation_points, background_points)

        print(f"\nPoint Summary:")
        print(f"  Fire points (positive): {len(fire_points)}")
        print(f"  Vegetation points (negative): {len(vegetation_points)}")
        print(f"  Background points (negative): {len(background_points)}")
        print(f"  Total points: {len(points)}")
        print(f"  Labels: {labels}")

        # Add points to first frame
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )

        # Propagate through video
        print("\nPropagating segmentation through video...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Save visualization
        self.visualize_results(thermal_frame_paths, visible_frame_paths, video_segments,
                              points, labels, output_dir)

        return video_segments

    def visualize_results(self, thermal_frame_paths, visible_frame_paths, video_segments,
                         points, labels, output_dir):
        """
        Visualize segmentation results with both thermal and visible frames.

        Args:
            thermal_frame_paths: List of thermal frame paths
            visible_frame_paths: List of visible frame paths
            video_segments: Dictionary of segmentation masks
            points: Point prompts used
            labels: Labels for points (1=positive, 0=negative)
            output_dir: Directory to save visualizations
        """
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        for frame_idx in range(len(thermal_frame_paths)):
            if frame_idx not in video_segments:
                continue

            # Read frames
            thermal = cv2.imread(thermal_frame_paths[frame_idx])
            thermal_rgb = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)

            visible = cv2.imread(visible_frame_paths[frame_idx])
            visible_rgb = cv2.cvtColor(visible, cv2.COLOR_BGR2RGB)

            # Get mask
            masks = video_segments[frame_idx]
            if 1 in masks:
                mask = masks[1][0]

                # Create visualization
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))

                # Row 1: Thermal
                axes[0, 0].imshow(thermal_rgb)
                axes[0, 0].set_title(f"Thermal Frame {frame_idx}")
                axes[0, 0].axis('off')

                axes[0, 1].imshow(mask, cmap='gray')
                axes[0, 1].set_title("Segmentation Mask")
                axes[0, 1].axis('off')

                overlay_thermal = thermal_rgb.copy()
                overlay_thermal[mask] = overlay_thermal[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
                axes[0, 2].imshow(overlay_thermal.astype(np.uint8))
                axes[0, 2].set_title("Thermal + Mask Overlay")
                axes[0, 2].axis('off')

                # Row 2: Visible
                axes[1, 0].imshow(visible_rgb)
                axes[1, 0].set_title(f"Visible Frame {frame_idx}")
                axes[1, 0].axis('off')

                # Show points on first frame
                if frame_idx == 0:
                    visible_with_points = visible_rgb.copy()
                    for pt, label in zip(points, labels):
                        color = (0, 255, 0) if label == 1 else (255, 0, 0)  # Green=positive, Red=negative
                        cv2.circle(visible_with_points, tuple(pt.astype(int)), 5, color, -1)
                    axes[1, 1].imshow(visible_with_points)
                    axes[1, 1].set_title("Point Prompts (Green=Fire, Red=Not-Fire)")
                else:
                    axes[1, 1].imshow(mask, cmap='gray')
                    axes[1, 1].set_title("Segmentation Mask")
                axes[1, 1].axis('off')

                overlay_visible = visible_rgb.copy()
                overlay_visible[mask] = overlay_visible[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
                axes[1, 2].imshow(overlay_visible.astype(np.uint8))
                axes[1, 2].set_title("Visible + Mask Overlay")
                axes[1, 2].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"result_{frame_idx:05d}.jpg"), dpi=100)
                plt.close()

        print(f"Visualizations saved to {vis_dir}")


def main():
    """Test the improved implementation."""
    # Setup paths
    checkpoint_path = "/Users/kritikapant/Desktop/PHD/Refined Sam/checkpoints/sam2_hiera_small.pt"

    # Example video paths
    infrared_video = "/Users/kritikapant/Desktop/PHD/Refined Sam/FireSentry-Benchmark-Dataset/Region A/Infrared Videos/Fire0.mp4"
    visible_video = "/Users/kritikapant/Desktop/PHD/Refined Sam/FireSentry-Benchmark-Dataset/Region A/Visible Light/Fire0.avi"

    output_dir = "/Users/kritikapant/Desktop/PHD/Refined Sam/results/improved"

    # Run segmentation
    segmenter = FireSegmentationImproved(checkpoint_path)
    segments = segmenter.segment_video(
        infrared_video,
        visible_video,
        output_dir,
        num_fire_points=15,
        num_veg_points=15,
        num_bg_points=10
    )

    print(f"\nSegmentation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

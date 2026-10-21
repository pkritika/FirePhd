"""
Baseline SAM2 Fire Segmentation
Uses basic color-based positive point selection from thermal video only.
"""

import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
from pathlib import Path


class FireSegmentationBaseline:
    """
    Baseline fire segmentation using SAM2 with color-based positive points.

    This approach uses only thermal video and selects positive points based on
    high-intensity (bright/hot) regions that likely represent fire.
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

    def detect_fire_points_color_based(self, frame, num_points=5):
        """
        Detect fire regions using simple color-based thresholding.

        This is a basic approach that selects bright/hot pixels as potential fire.
        LIMITATION: This can incorrectly identify vegetation or other bright objects.

        Args:
            frame: Input frame (BGR format)
            num_points: Number of positive points to select

        Returns:
            points: Array of (x, y) coordinates
            labels: Array of labels (all 1 for positive points)
        """
        # Convert to grayscale to find bright regions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold to find bright/hot regions (simple approach)
        # Higher intensity = hotter = likely fire
        threshold = np.percentile(gray, 95)  # Top 5% brightest pixels
        hot_mask = gray > threshold

        # Find coordinates of hot pixels
        hot_coords = np.argwhere(hot_mask)

        if len(hot_coords) == 0:
            # No hot regions found, sample random points
            h, w = gray.shape
            points = np.random.randint(0, min(h, w), size=(num_points, 2))
        else:
            # Sample points from hot regions
            if len(hot_coords) > num_points:
                indices = np.random.choice(len(hot_coords), num_points, replace=False)
                sampled_coords = hot_coords[indices]
            else:
                sampled_coords = hot_coords

            # Convert from (row, col) to (x, y)
            points = np.column_stack([sampled_coords[:, 1], sampled_coords[:, 0]])

        # All positive points (label=1)
        labels = np.ones(len(points), dtype=np.int32)

        return points, labels

    def segment_video(self, infrared_video_path, output_dir, num_points=5):
        """
        Segment fire in video using baseline approach.

        Args:
            infrared_video_path: Path to infrared/thermal video
            output_dir: Directory to save results
            num_points: Number of positive points to use
        """
        # Extract frames
        frames_dir = os.path.join(output_dir, "frames")
        frame_paths = self.extract_frames_from_video(infrared_video_path, frames_dir)

        # Initialize inference state with frame directory
        inference_state = self.predictor.init_state(video_path=frames_dir)

        # Read first frame to get positive points
        first_frame = cv2.imread(frame_paths[0])
        points, labels = self.detect_fire_points_color_based(first_frame, num_points)

        print(f"Using {len(points)} positive points for segmentation")
        print(f"Points: {points}")
        print(f"Labels: {labels}")

        # Add points to first frame (frame_idx=0, obj_id=1)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )

        # Propagate through video
        print("Propagating segmentation through video...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Save visualization
        self.visualize_results(frame_paths, video_segments, output_dir)

        return video_segments

    def visualize_results(self, frame_paths, video_segments, output_dir):
        """
        Visualize segmentation results.

        Args:
            frame_paths: List of frame file paths
            video_segments: Dictionary of segmentation masks
            output_dir: Directory to save visualizations
        """
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        for frame_idx, frame_path in enumerate(frame_paths):
            if frame_idx not in video_segments:
                continue

            # Read original frame
            frame = cv2.imread(frame_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get mask for this frame
            masks = video_segments[frame_idx]
            if 1 in masks:
                mask = masks[1][0]  # obj_id=1, first mask

                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original frame
                axes[0].imshow(frame_rgb)
                axes[0].set_title(f"Original Frame {frame_idx}")
                axes[0].axis('off')

                # Mask
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title(f"Segmentation Mask")
                axes[1].axis('off')

                # Overlay
                overlay = frame_rgb.copy()
                overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
                axes[2].imshow(overlay.astype(np.uint8))
                axes[2].set_title(f"Overlay")
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"result_{frame_idx:05d}.jpg"))
                plt.close()

        print(f"Visualizations saved to {vis_dir}")


def main():
    """Test the baseline implementation."""
    # Setup paths
    checkpoint_path = "/Users/kritikapant/Desktop/PHD/Refined Sam/checkpoints/sam2_hiera_small.pt"

    # Example video path - update this to your actual video
    infrared_video = "/Users/kritikapant/Desktop/PHD/Refined Sam/FireSentry-Benchmark-Dataset/Region A/Infrared Videos/Fire0.mp4"

    output_dir = "/Users/kritikapant/Desktop/PHD/Refined Sam/results/baseline"

    # Run segmentation
    segmenter = FireSegmentationBaseline(checkpoint_path)
    segments = segmenter.segment_video(infrared_video, output_dir, num_points=10)

    print(f"\nSegmentation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

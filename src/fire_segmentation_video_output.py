"""
SAM2 Fire Segmentation with VIDEO Output
Processes video input and generates segmented VIDEO output.
"""

import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor


class FireSegmentationVideoOutput:
    """
    Fire segmentation using SAM2 with VIDEO input and VIDEO output.

    Key features:
    - Processes thermal MP4 video directly
    - Outputs segmented VIDEO (not individual frames)
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

    def get_video_info(self, video_path):
        """Get video information."""
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return width, height, frame_count, fps

    def read_video_frame(self, video_path, frame_idx):
        """Read specific frame from video."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx}")

        return frame

    def read_visible_frame(self, visible_frames_dir, frame_idx):
        """Read visible light frame from directory."""
        from pathlib import Path
        visible_frames = sorted(Path(visible_frames_dir).glob("*.jpg"))

        if frame_idx >= len(visible_frames):
            raise RuntimeError(f"Frame {frame_idx} out of range")

        frame = cv2.imread(str(visible_frames[frame_idx]))
        if frame is None:
            raise RuntimeError(f"Failed to read frame")

        return frame

    def detect_fire_points(self, thermal_frame, num_points=10):
        """Detect fire points from thermal frame."""
        gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, 95)  # Top 5% hottest
        hot_mask = gray > threshold
        hot_coords = np.argwhere(hot_mask)

        if len(hot_coords) == 0:
            return np.array([])

        if len(hot_coords) > num_points:
            indices = np.random.choice(len(hot_coords), num_points, replace=False)
            sampled = hot_coords[indices]
        else:
            sampled = hot_coords

        return np.column_stack([sampled[:, 1], sampled[:, 0]])

    def detect_vegetation_points(self, visible_frame, num_points=10):
        """Detect vegetation points (negative)."""
        hsv = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        veg_mask = cv2.inRange(hsv, lower_green, upper_green)
        veg_coords = np.argwhere(veg_mask > 0)

        if len(veg_coords) == 0:
            return np.array([])

        if len(veg_coords) > num_points:
            indices = np.random.choice(len(veg_coords), num_points, replace=False)
            sampled = veg_coords[indices]
        else:
            sampled = veg_coords

        return np.column_stack([sampled[:, 1], sampled[:, 0]])

    def detect_background_points(self, visible_frame, thermal_frame, num_points=5):
        """Detect background points (negative)."""
        gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        cold_threshold = np.percentile(gray, 30)
        cold_mask = gray < cold_threshold

        hsv = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        not_veg_mask = cv2.inRange(hsv, lower_green, upper_green) == 0

        bg_mask = cold_mask & not_veg_mask
        bg_coords = np.argwhere(bg_mask)

        if len(bg_coords) == 0:
            return np.array([])

        if len(bg_coords) > num_points:
            indices = np.random.choice(len(bg_coords), num_points, replace=False)
            sampled = bg_coords[indices]
        else:
            sampled = bg_coords

        return np.column_stack([sampled[:, 1], sampled[:, 0]])

    def combine_points_and_labels(self, fire_pts, veg_pts, bg_pts):
        """Combine positive and negative points."""
        all_points = []
        all_labels = []

        if len(fire_pts) > 0:
            all_points.append(fire_pts)
            all_labels.extend([1] * len(fire_pts))

        if len(veg_pts) > 0:
            all_points.append(veg_pts)
            all_labels.extend([0] * len(veg_pts))

        if len(bg_pts) > 0:
            all_points.append(bg_pts)
            all_labels.extend([0] * len(bg_pts))

        if len(all_points) == 0:
            return np.array([]), np.array([])

        return np.vstack(all_points), np.array(all_labels, dtype=np.int32)

    def segment_video(self, infrared_video, visible_frames_dir, output_video_path,
                     num_fire_points=15, num_veg_points=15, num_bg_points=10):
        """
        Segment fire and output as VIDEO file.

        Args:
            infrared_video: Path to thermal video (MP4)
            visible_frames_dir: Directory with visible frames
            output_video_path: Output video path (MP4)
            num_fire_points: Positive fire points
            num_veg_points: Negative vegetation points
            num_bg_points: Negative background points

        Returns:
            Path to output video
        """
        print("\n" + "="*80)
        print("SAM2 FIRE SEGMENTATION - VIDEO OUTPUT")
        print("="*80)

        # Get video info
        print(f"\nAnalyzing input video: {infrared_video}")
        width, height, frame_count, fps = self.get_video_info(infrared_video)
        print(f"  Resolution: {width}x{height}")
        print(f"  Frames: {frame_count}")
        print(f"  FPS: {fps}")

        # Read first frame for point detection
        print("\nDetecting point prompts from first frame...")
        first_thermal = self.read_video_frame(infrared_video, 0)
        first_visible = self.read_visible_frame(visible_frames_dir, 0)
        first_visible = cv2.resize(first_visible, (width, height))

        # Detect points
        fire_pts = self.detect_fire_points(first_thermal, num_fire_points)
        veg_pts = self.detect_vegetation_points(first_visible, num_veg_points)
        bg_pts = self.detect_background_points(first_visible, first_thermal, num_bg_points)

        points, labels = self.combine_points_and_labels(fire_pts, veg_pts, bg_pts)

        print(f"  Fire points (positive): {len(fire_pts)}")
        print(f"  Vegetation points (negative): {len(veg_pts)}")
        print(f"  Background points (negative): {len(bg_pts)}")
        print(f"  Total prompts: {len(points)}")

        if len(points) == 0:
            raise RuntimeError("No points detected! Check thresholds.")

        # Initialize SAM2 with video
        print(f"\nInitializing SAM2 with video...")
        inference_state = self.predictor.init_state(video_path=infrared_video)
        print(f"  Loaded {inference_state['num_frames']} frames")

        # Add prompts
        print("\nAdding point prompts...")
        self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )

        # Propagate segmentation
        print("\nPropagating segmentation through video...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        print(f"  Segmented {len(video_segments)} frames")

        # Create output video
        print(f"\nCreating output video: {output_video_path}")
        self.create_output_video(infrared_video, video_segments, output_video_path, fps)

        print(f"\n✓ Output video saved: {output_video_path}")
        print("="*80)

        return output_video_path

    def create_output_video(self, input_video, segments, output_path, fps):
        """Create output video with segmentation overlay."""
        # Open input video
        cap = cv2.VideoCapture(input_video)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        total_frames = len(segments)

        print(f"  Writing {total_frames} frames to video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get segmentation mask
            if frame_idx in segments and 1 in segments[frame_idx]:
                mask = segments[frame_idx][1][0]

                # Create overlay
                overlay = frame.copy()
                overlay[mask] = overlay[mask] * 0.5 + np.array([0, 0, 255]) * 0.5

                # Add contours for better visibility
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

                out.write(overlay.astype(np.uint8))
            else:
                out.write(frame)

            frame_idx += 1

            if frame_idx % 10 == 0:
                print(f"    Progress: {frame_idx}/{total_frames} frames")

        cap.release()
        out.release()

        print(f"  ✓ Video writing complete")


def main():
    """Main execution."""
    checkpoint_path = "/Users/kritikapant/Desktop/PHD/Refined Sam/checkpoints/sam2_hiera_small.pt"

    infrared_video = "/Users/kritikapant/Desktop/PHD/Refined Sam/FireSentry-Benchmark-Dataset/Region A/Infrared Videos/video_001.mp4"
    visible_frames_dir = "/Users/kritikapant/Desktop/PHD/Refined Sam/FireSentry-Benchmark-Dataset/Region A/Visible Light"

    output_dir = "/Users/kritikapant/Desktop/PHD/Refined Sam/results/video_output"
    os.makedirs(output_dir, exist_ok=True)

    output_video = os.path.join(output_dir, "segmented_video_001.mp4")

    # Create segmenter and process
    segmenter = FireSegmentationVideoOutput(checkpoint_path)

    result_video = segmenter.segment_video(
        infrared_video,
        visible_frames_dir,
        output_video,
        num_fire_points=15,
        num_veg_points=15,
        num_bg_points=10
    )

    print(f"\n✓ SUCCESS! Segmented video saved to:")
    print(f"  {result_video}")


if __name__ == "__main__":
    main()

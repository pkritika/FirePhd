"""
SAM2 Fire Segmentation with Smoke Filtering
Removes smoke connections and only segments actual fire regions.
"""

import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor


class FireSegmentationNoSmoke:
    """
    Fire segmentation that filters out smoke by using:
    1. Temperature thresholding - Only keep high-temperature regions
    2. Morphological operations - Remove thin smoke connections
    3. Negative points in smoke regions
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

    def detect_fire_points(self, thermal_frame, num_points=15):
        """Detect ONLY very hot fire points (excluding warm smoke)."""
        gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)

        # Use higher percentile to get only hottest regions (actual fire, not smoke)
        # Smoke is warm but not as hot as active fire
        threshold = np.percentile(gray, 98)  # Top 2% hottest (was 95)
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

    def detect_smoke_points(self, thermal_frame, num_points=15):
        """Detect warm smoke regions (negative points to exclude from fire)."""
        gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)

        # Smoke is in the warm range but not hot like fire
        # Between 70th and 98th percentile = warm but not fire
        low_threshold = np.percentile(gray, 70)
        high_threshold = np.percentile(gray, 98)

        smoke_mask = (gray > low_threshold) & (gray < high_threshold)
        smoke_coords = np.argwhere(smoke_mask)

        if len(smoke_coords) == 0:
            return np.array([])

        if len(smoke_coords) > num_points:
            indices = np.random.choice(len(smoke_coords), num_points, replace=False)
            sampled = smoke_coords[indices]
        else:
            sampled = smoke_coords

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

    def combine_points_and_labels(self, fire_pts, smoke_pts, veg_pts, bg_pts):
        """Combine positive (fire) and negative (smoke, vegetation, background) points."""
        all_points = []
        all_labels = []

        if len(fire_pts) > 0:
            all_points.append(fire_pts)
            all_labels.extend([1] * len(fire_pts))

        # Mark smoke as NEGATIVE (key improvement!)
        if len(smoke_pts) > 0:
            all_points.append(smoke_pts)
            all_labels.extend([0] * len(smoke_pts))

        if len(veg_pts) > 0:
            all_points.append(veg_pts)
            all_labels.extend([0] * len(veg_pts))

        if len(bg_pts) > 0:
            all_points.append(bg_pts)
            all_labels.extend([0] * len(bg_pts))

        if len(all_points) == 0:
            return np.array([]), np.array([])

        return np.vstack(all_points), np.array(all_labels, dtype=np.int32)

    def filter_smoke_from_mask(self, mask, thermal_frame):
        """
        Post-process mask to remove smoke regions using temperature thresholding.
        Only keep pixels that are VERY hot (actual fire).
        """
        gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)

        # Only keep pixels above fire temperature threshold
        fire_threshold = np.percentile(gray, 97)  # Top 3% hottest
        fire_temp_mask = gray > fire_threshold

        # Apply temperature filter to SAM2 mask
        filtered_mask = mask & fire_temp_mask

        return filtered_mask

    def remove_thin_connections(self, mask):
        """
        Use morphological operations to remove thin smoke connections.
        This separates fire regions connected by smoke.
        """
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Opening operation: erosion followed by dilation
        # This removes thin connections while preserving main fire regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

        # Close small holes within fire regions
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        return (closed > 0).astype(bool)

    def segment_video(self, infrared_video, visible_frames_dir, output_video_path,
                     num_fire_points=15, num_smoke_points=15, num_veg_points=10, num_bg_points=5):
        """
        Segment fire and output as VIDEO file with smoke filtering.
        """
        print("\n" + "="*80)
        print("SAM2 FIRE SEGMENTATION - NO SMOKE")
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

        # Detect points (including SMOKE as negative!)
        fire_pts = self.detect_fire_points(first_thermal, num_fire_points)
        smoke_pts = self.detect_smoke_points(first_thermal, num_smoke_points)
        veg_pts = self.detect_vegetation_points(first_visible, num_veg_points)
        bg_pts = self.detect_background_points(first_visible, first_thermal, num_bg_points)

        points, labels = self.combine_points_and_labels(fire_pts, smoke_pts, veg_pts, bg_pts)

        print(f"  Fire points (positive): {len(fire_pts)}")
        print(f"  Smoke points (NEGATIVE): {len(smoke_pts)}")  # KEY!
        print(f"  Vegetation points (negative): {len(veg_pts)}")
        print(f"  Background points (negative): {len(bg_pts)}")
        print(f"  Total prompts: {len(points)}")

        if len(points) == 0:
            raise RuntimeError("No points detected! Check thresholds.")

        # Initialize SAM2
        print(f"\nInitializing SAM2 with video...")
        inference_state = self.predictor.init_state(video_path=infrared_video)
        print(f"  Loaded {inference_state['num_frames']} frames")

        # Add prompts
        print("\nAdding point prompts (including smoke as negative)...")
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

        # Create output video with smoke filtering
        print(f"\nCreating output video with smoke filtering: {output_video_path}")
        self.create_output_video(infrared_video, video_segments, output_video_path, fps)

        print(f"\n✓ Output video saved: {output_video_path}")
        print("="*80)

        return output_video_path

    def create_output_video(self, input_video, segments, output_path, fps):
        """Create output video with segmentation overlay and smoke filtering."""
        cap = cv2.VideoCapture(input_video)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        total_frames = len(segments)

        print(f"  Writing {total_frames} frames with smoke filtering...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get segmentation mask
            if frame_idx in segments and 1 in segments[frame_idx]:
                raw_mask = segments[frame_idx][1][0]

                # FILTER SMOKE using temperature
                filtered_mask = self.filter_smoke_from_mask(raw_mask, frame)

                # REMOVE THIN CONNECTIONS using morphology
                final_mask = self.remove_thin_connections(filtered_mask)

                # Create overlay
                overlay = frame.copy()
                overlay[final_mask] = overlay[final_mask] * 0.5 + np.array([0, 0, 255]) * 0.5

                # Add contours
                mask_uint8 = (final_mask * 255).astype(np.uint8)
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
    """Test the smoke-filtered implementation."""
    checkpoint_path = "/Users/kritikapant/Desktop/PHD/Refined Sam/checkpoints/sam2_hiera_small.pt"

    infrared_video = "/Users/kritikapant/Desktop/PHD/Refined Sam/FireSentry-Benchmark-Dataset/Region A/Infrared Videos/video_001.mp4"
    visible_frames_dir = "/Users/kritikapant/Desktop/PHD/Refined Sam/FireSentry-Benchmark-Dataset/Region A/Visible Light"

    output_dir = "/Users/kritikapant/Desktop/PHD/Refined Sam/results/no_smoke"
    os.makedirs(output_dir, exist_ok=True)

    output_video = os.path.join(output_dir, "segmented_no_smoke_video_001.mp4")

    # Create segmenter and process
    segmenter = FireSegmentationNoSmoke(checkpoint_path)

    result_video = segmenter.segment_video(
        infrared_video,
        visible_frames_dir,
        output_video,
        num_fire_points=15,
        num_smoke_points=15,  # NEW: negative points for smoke
        num_veg_points=10,
        num_bg_points=5
    )

    print(f"\n✓ SUCCESS! Smoke-filtered video saved to:")
    print(f"  {result_video}")


if __name__ == "__main__":
    main()

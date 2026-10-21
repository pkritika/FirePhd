"""
Batch Process Multiple Videos for Fire Segmentation
Processes multiple infrared videos and outputs VIDEO files (not frames).
"""

import os
import sys
import time
from pathlib import Path
from fire_segmentation_video_output import FireSegmentationVideoOutput


def process_multiple_videos(
    checkpoint_path,
    infrared_dir,
    visible_frames_dir,
    output_base_dir,
    video_numbers,
    num_fire_points=15,
    num_veg_points=15,
    num_bg_points=10
):
    """
    Process multiple videos in batch - outputs VIDEO files.

    Args:
        checkpoint_path: Path to SAM2 checkpoint
        infrared_dir: Directory containing infrared videos
        visible_frames_dir: Directory containing visible light frames
        output_base_dir: Base directory for all results
        video_numbers: List of video numbers to process (e.g., [2, 3, 4, ..., 11])
        num_fire_points: Number of positive fire points
        num_veg_points: Number of negative vegetation points
        num_bg_points: Number of negative background points
    """

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Create segmenter once (reuse for all videos)
    print("=" * 80)
    print("BATCH FIRE SEGMENTATION - VIDEO OUTPUT")
    print("=" * 80)
    print(f"\nInitializing SAM2 model...")

    segmenter = FireSegmentationVideoOutput(checkpoint_path)

    # Results tracking
    results_summary = []
    total_start_time = time.time()

    # Process each video
    for idx, video_num in enumerate(video_numbers, 1):
        video_name = f"video_{video_num:03d}.mp4"
        video_path = os.path.join(infrared_dir, video_name)

        # Check if video exists
        if not os.path.exists(video_path):
            print(f"\n⚠️  [{idx}/{len(video_numbers)}] Skipping {video_name} - file not found")
            results_summary.append({
                'video': video_name,
                'status': 'NOT_FOUND',
                'time': 0,
                'frames': 0
            })
            continue

        # Output video path
        output_video = os.path.join(output_base_dir, f"segmented_{video_name}")

        print(f"\n[{idx}/{len(video_numbers)}] Processing: {video_name}")
        print("-" * 80)

        video_start_time = time.time()

        try:
            # Process video and get output video path
            result_video = segmenter.segment_video(
                video_path,
                visible_frames_dir,
                output_video,
                num_fire_points=num_fire_points,
                num_veg_points=num_veg_points,
                num_bg_points=num_bg_points
            )

            video_time = time.time() - video_start_time

            # Get video info to count frames
            import cv2
            cap = cv2.VideoCapture(result_video)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(f"✅ {video_name} complete!")
            print(f"   Output: {os.path.basename(result_video)}")
            print(f"   Frames: {num_frames}")
            print(f"   Time: {video_time:.1f}s ({video_time/num_frames:.2f}s per frame)")

            results_summary.append({
                'video': video_name,
                'status': 'SUCCESS',
                'time': video_time,
                'frames': num_frames,
                'output': result_video
            })

        except Exception as e:
            video_time = time.time() - video_start_time
            print(f"❌ {video_name} FAILED!")
            print(f"   Error: {str(e)}")

            results_summary.append({
                'video': video_name,
                'status': 'FAILED',
                'time': video_time,
                'frames': 0,
                'error': str(e)
            })

    # Print final summary
    total_time = time.time() - total_start_time

    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\nResults Summary:")
    print("-" * 80)

    successful = 0
    failed = 0
    total_frames = 0

    for result in results_summary:
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌" if result['status'] == 'FAILED' else "⚠️"
        print(f"{status_icon} {result['video']:<20} {result['status']:<12} "
              f"Frames: {result['frames']:<5} Time: {result['time']:.1f}s")

        if result['status'] == 'SUCCESS':
            successful += 1
            total_frames += result['frames']
        elif result['status'] == 'FAILED':
            failed += 1

    print("-" * 80)
    print(f"Successful: {successful}/{len(video_numbers)}")
    print(f"Failed: {failed}/{len(video_numbers)}")
    print(f"Total frames processed: {total_frames}")
    if total_frames > 0:
        print(f"Average processing time: {total_time/total_frames:.2f}s per frame")

    # Save summary to file
    summary_path = os.path.join(output_base_dir, "BATCH_SUMMARY.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BATCH FIRE SEGMENTATION SUMMARY - VIDEO OUTPUT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
        f.write(f"Videos processed: {successful + failed}/{len(video_numbers)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Total frames: {total_frames}\n\n")

        f.write("Per-Video Results:\n")
        f.write("-" * 80 + "\n")
        for result in results_summary:
            f.write(f"{result['video']:<20} {result['status']:<12} "
                   f"Frames: {result['frames']:<5} Time: {result['time']:.1f}s\n")
            if 'output' in result:
                f.write(f"  Output: {os.path.basename(result['output'])}\n")
            if 'error' in result:
                f.write(f"  Error: {result['error']}\n")

    print(f"\nSummary saved to: {summary_path}")
    print(f"\nAll segmented videos in: {output_base_dir}/")

    return results_summary


def main():
    """Process videos 2-11 from Region A (10 videos, excluding video_001 already processed)."""

    base_dir = "/Users/kritikapant/Desktop/PHD/Refined Sam"

    # Paths
    checkpoint_path = os.path.join(base_dir, "checkpoints/sam2_hiera_small.pt")
    infrared_dir = os.path.join(base_dir, "FireSentry-Benchmark-Dataset/Region A/Infrared Videos")
    visible_frames_dir = os.path.join(base_dir, "FireSentry-Benchmark-Dataset/Region A/Visible Light")
    output_base_dir = os.path.join(base_dir, "results/batch_region_a")

    # Process videos 2-11 (10 videos, excluding video_001 which was already processed)
    video_numbers = list(range(2, 12))  # [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    print(f"\nWill process {len(video_numbers)} videos:")
    for n in video_numbers:
        print(f"  - video_{n:03d}.mp4")
    print()

    # Process all videos
    results = process_multiple_videos(
        checkpoint_path=checkpoint_path,
        infrared_dir=infrared_dir,
        visible_frames_dir=visible_frames_dir,
        output_base_dir=output_base_dir,
        video_numbers=video_numbers,
        num_fire_points=15,
        num_veg_points=15,
        num_bg_points=10
    )

    print("\n" + "=" * 80)
    print("ALL DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

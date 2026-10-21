"""
Batch Process Multiple Videos with Smoke Filtering
Processes multiple infrared videos and outputs VIDEO files without smoke connections.
"""

import os
import time
from fire_segmentation_no_smoke import FireSegmentationNoSmoke


def process_multiple_videos(
    checkpoint_path,
    infrared_dir,
    visible_frames_dir,
    output_base_dir,
    video_numbers,
    num_fire_points=15,
    num_smoke_points=15,
    num_veg_points=10,
    num_bg_points=5
):
    """Process multiple videos in batch with smoke filtering."""
    
    os.makedirs(output_base_dir, exist_ok=True)

    print("=" * 80)
    print("BATCH FIRE SEGMENTATION - NO SMOKE (VIDEO OUTPUT)")
    print("=" * 80)
    print(f"\nInitializing SAM2 model...")

    segmenter = FireSegmentationNoSmoke(checkpoint_path)

    results_summary = []
    total_start_time = time.time()

    for idx, video_num in enumerate(video_numbers, 1):
        video_name = f"video_{video_num:03d}.mp4"
        video_path = os.path.join(infrared_dir, video_name)

        if not os.path.exists(video_path):
            print(f"\n[{idx}/{len(video_numbers)}] Skipping {video_name} - file not found")
            results_summary.append({
                'video': video_name,
                'status': 'NOT_FOUND',
                'time': 0,
                'frames': 0
            })
            continue

        output_video = os.path.join(output_base_dir, f"segmented_no_smoke_{video_name}")

        print(f"\n[{idx}/{len(video_numbers)}] Processing: {video_name}")
        print("-" * 80)

        video_start_time = time.time()

        try:
            result_video = segmenter.segment_video(
                video_path,
                visible_frames_dir,
                output_video,
                num_fire_points=num_fire_points,
                num_smoke_points=num_smoke_points,
                num_veg_points=num_veg_points,
                num_bg_points=num_bg_points
            )

            video_time = time.time() - video_start_time

            import cv2
            cap = cv2.VideoCapture(result_video)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(f"SUCCESS {video_name} complete!")
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
            print(f"FAILED {video_name}")
            print(f"   Error: {str(e)}")

            results_summary.append({
                'video': video_name,
                'status': 'FAILED',
                'time': video_time,
                'frames': 0,
                'error': str(e)
            })

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
        status_icon = "OK" if result['status'] == 'SUCCESS' else "FAIL" if result['status'] == 'FAILED' else "SKIP"
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

    summary_path = os.path.join(output_base_dir, "BATCH_SUMMARY_NO_SMOKE.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BATCH FIRE SEGMENTATION SUMMARY - NO SMOKE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
        f.write(f"Videos processed: {successful + failed}/{len(video_numbers)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Total frames: {total_frames}\n\n")

        f.write("Smoke Filtering Enabled:\n")
        f.write("  - Smoke points as NEGATIVE prompts\n")
        f.write("  - Temperature-based post-filtering\n")
        f.write("  - Morphological operations to remove thin connections\n\n")

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
    print(f"\nAll segmented videos (NO SMOKE) in: {output_base_dir}/")

    return results_summary


def main():
    """Process videos 2-11 from Region A with smoke filtering."""

    base_dir = "/Users/kritikapant/Desktop/PHD/Refined Sam"

    checkpoint_path = os.path.join(base_dir, "checkpoints/sam2_hiera_small.pt")
    infrared_dir = os.path.join(base_dir, "FireSentry-Benchmark-Dataset/Region A/Infrared Videos")
    visible_frames_dir = os.path.join(base_dir, "FireSentry-Benchmark-Dataset/Region A/Visible Light")
    output_base_dir = os.path.join(base_dir, "results/batch_no_smoke")

    video_numbers = list(range(2, 12))

    print(f"\nWill process {len(video_numbers)} videos with SMOKE FILTERING:")
    for n in video_numbers:
        print(f"  - video_{n:03d}.mp4")
    print()

    results = process_multiple_videos(
        checkpoint_path=checkpoint_path,
        infrared_dir=infrared_dir,
        visible_frames_dir=visible_frames_dir,
        output_base_dir=output_base_dir,
        video_numbers=video_numbers,
        num_fire_points=15,
        num_smoke_points=15,
        num_veg_points=10,
        num_bg_points=5
    )

    print("\n" + "=" * 80)
    print("ALL DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

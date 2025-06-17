import cv2
import argparse
import os

args = argparse.ArgumentParser(description="Convert JPEG frames to MP4 video")
args.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the input directory containing JPEG frames"
)
args.add_argument(
    "--output",
    type=str,
    default="Output/output.mp4",
    help="Path to save the output MP4 video file"
)

args = args.parse_args()
Input = args.input
Output = args.output

out = cv2.VideoWriter(
    Output,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30.0,  # Assuming 30 FPS
    (1440, 1080)  # Assuming 1440x1080 resolution
)

for file in os.listdir(Input):
    if file.endswith(".jpg"):
        frame = cv2.imread(os.path.join(Input, file), cv2.IMREAD_COLOR)
        if frame is not None:
            out.write(frame)
            print(f"Added {file} to video.")
            print(f"Frame shape: {frame.shape}, length: {len(frame.shape)}")
            
        else:
            print(f"Failed to read {file}.")

out.release()
print(f"Video saved as {Output}.")

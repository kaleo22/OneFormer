import cv2
import argparse

parser = argparse.ArgumentParser(description="Convert MP4 video to JPEG frames")
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the input MP4 video file"
)
parser.add_argument(
    "--output",
    type=str,
    default="output",
    help="Directory to save the output JPEG frames"
)
args = parser.parse_args()
Input = args.input
Output = args.output

cap = cv2.VideoCapture(Input)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.imwrite(f"{Output}/frame_{frame_number:04d}.jpg", frame)
    print(f"Saved frame {frame_number} as JPEG.")
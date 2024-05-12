import os
import subprocess


def cut_video(video_path):
    # Get the filename and extension
    filename, ext = os.path.splitext(video_path)

    # Create a directory to store the clips
    output_dir = filename + "_clips"
    os.makedirs(output_dir, exist_ok=True)

    # Use ffmpeg to cut the video into twenty-second clips
    command = [
        "ffmpeg",
        "-i", video_path,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", "20",
        "-reset_timestamps", "1",
        os.path.join(output_dir, filename + "_%03d" + ext)
    ]

    # Run the command
    subprocess.run(command)


if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ")
    cut_video(video_path)

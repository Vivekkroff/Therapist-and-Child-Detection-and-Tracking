# Therapist-and-Child-Detection-and-Tracking


Person Detection and Tracking Pipeline
This project implements a person detection and tracking pipeline to identify and track children and therapists in a video. The pipeline uses a pre-trained SSD MobileNet model for object detection and integrates the Deep SORT algorithm for tracking. The final output is a video with bounding boxes and unique IDs assigned to detected persons.

Features
Person Detection: Identifies persons in the video using the SSD MobileNet model.
Tracking: Assigns unique IDs to each detected person and tracks them throughout the video.
Handling Re-entries and Occlusions: Maintains tracking of individuals who exit and re-enter the frame, as well as handling partial occlusions.
Requirements
Python 3.x
opencv-python
tensorflow
tensorflow-hub
Deep SORT library (e.g., deep_sort_realtime)
Installation


pip install opencv-python tensorflow tensorflow-hub
pip install deep_sort_realtime  # or your preferred Deep SORT library
Usage
Prepare your video files: Ensure you have the input video file (e.g., test_video.mp4) in the same directory as the script or provide the correct path to it.

Run the script:


python detect_and_track.py
This script will process the input video, detect persons, track them, and save the output video with bounding boxes and unique IDs to output_video.avi.

Code Overview
detect_objects(frame): Converts the video frame to a tensor, runs object detection using the SSD MobileNet model, and returns detection boxes, scores, and classes.

draw_boxes(frame, boxes, scores, classes, track_ids, threshold=0.5): Draws bounding boxes around detected persons, labels them with their unique IDs, and displays the confidence score.

process_video(input_video_path, output_video_path): Reads the input video, performs object detection and tracking on each frame, and writes the annotated frames to the output video.




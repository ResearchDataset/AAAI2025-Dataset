# MINGLE: VLMs for Semantically Complex Region Detection in Urban Scenes
![pipeline](images/pipeline.png)

## 1 The Model of Pairwise Grouping Judgment (Stage 2)

This repository contains:
- CSV data files used for fine-tuning and evaluation
- Download and inference scripts
- Example prompts and usage instructions

The model weights are hosted on Hugging Face:
👉 [https://huggingface.co/AI-Anon/MINGLE-1.0](https://huggingface.co/AI-Anon/MINGLE-1.0)

## 2 The 100k Street-Level Grouping Dataset

We release a large-scale metadata set of 100,000 street-view images, each annotated with both individual people and social group regions. These annotations were generated using the full MINGLE pipeline and combine manual labels with validated pipeline predictions.

Note:
We do not release any image files due to copyright and privacy concerns. Instead, we provide:
	•	Metadata for each image, including platform, location info, and viewing angles
	•	Scripts for downloading and re-stitching the original images from public APIs (e.g., Google Street View, Apple Lookaround, Mapillary)

⸻

📦 Contents of the CSV

Each row in the CSV corresponds to one annotated image and includes:
	•	image_id: Unique image identifier
	•	platform: One of {apple, google, bing, mapillary}
	•	location_info: Fields such as pano ID, latitude, longitude, heading, pitch, zoom, etc.
	•	person_boxes: Bounding boxes for detected individuals in [x1,y1,x2,y2] format
	•	group_boxes: Bounding boxes for socially affiliated groups (≥2 people)
	•	group_assignments: A mapping from each person to a group ID
	•	metadata: Number of people, number of groups, average group size, etc.

These fields enable researchers to recreate the full image context and re-align group regions on top of the stitched views.

⸻

🌐 Image Source Platforms

The 100k images were originally collected from:
	•	Apple Lookaround (60%)
	•	Google Street View (15%)
	•	Bing Streetside (15%)
	•	Mapillary (10%)

Only lateral views (left/right) were used to better capture pedestrian interactions.

⸻

🛠️ Reconstructing the Images

A companion script scripts/download_and_stitch.py is provided to help reconstruct the original scenes. It takes in the metadata (e.g., pano ID and viewing angles), queries the relevant API, and outputs the RGB image suitable for visualization or inference.

⸻



# MINGLE: VLMs for Semantically Complex Region Detection in Urban Scenes
![pipeline](images/pipeline.png)

## 1 The Model of Pairwise Grouping Judgment (Stage 2)

This repository contains:
- CSV data files used for fine-tuning and evaluation
- Download and inference scripts
- Example prompts and usage instructions

The model weights are hosted on Hugging Face:
👉 [https://huggingface.co/AI-Anon/MINGLE-1.0](https://huggingface.co/AI-Anon/MINGLE-1.0)

📖 **Usage Instructions**: Detailed step-by-step instructions on how to use this model are provided in the Jupyter notebook `ipynb/pairwise_grouping.ipynb`.

## 2 The 100k Street-Level Grouping Dataset

We release a large-scale metadata set of 100,000 street-view images, each annotated with both individual people and social group regions. These annotations were generated using the full MINGLE pipeline and combine manual labels with validated pipeline predictions.

Note:
We do not release any image files due to copyright and privacy concerns. Instead, we provide:
	- Metadata for each image, including platform, location info, and viewing angles
	- Scripts for downloading and re-stitching the original images from public APIs (e.g., Google Street View, Apple Lookaround, Mapillary)

⸻

📦 Contents of the CSV

Each row in the CSV corresponds to one annotated image and includes:
	- image_id: Unique image identifier
	- group id: the id of the detected group in this picture
	- cnt_people: the number of people detected in this image
	- cnt_group: the number of the groups detected in this image
	- box_detected_person: the bounding boxes of detected person
	- group_box: the bounding boxes of detected groups
	- box_within_group: the person boxes within the group boxes
	- cnt_person_within_group: the number of persons in the groups
	- person_box_is_unreal_person: this is the check of whether the person is an unreal person
	- has unreal: whether there is any unreal person (but all have been removed already)
	- source: One of {apple, google, bing, mapillary}
	- id: the id of that pano in that source

These fields enable researchers to recreate the full image context and re-align group regions on top of the stitched views.

⸻

🌐 Image Source Platforms

The 100k images were originally collected from:
	- Apple Lookaround (60%)
	- Google Street View (15%)
	- Bing Streetside (15%)
	- Mapillary (10%)

Only lateral views (left/right) were used to better capture pedestrian interactions.

⸻

🛠️ Reconstructing the Images

A companion script scripts/download_and_stitch.py is provided to help reconstruct the original scenes. It takes in the metadata (e.g., pano ID and viewing angles), queries the relevant API, and outputs the RGB image suitable for visualization or inference.

⸻



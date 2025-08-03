import numpy as np

def compute_median_depth(depth_image, bounding_boxes):
    depth_image = depth_image.convert("L")
    
    depth_array = np.array(depth_image)
    median_depths = []
    for i, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes):
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        roi = depth_array[y_min:y_max, x_min:x_max]
        valid_pixels = roi[roi > 0].flatten()
        median_depth = np.median(valid_pixels) if valid_pixels.size > 0 else 0
        median_depths.append(int(median_depth))
    return median_depths
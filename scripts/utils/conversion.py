import cv2
import os
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import map_coordinates

def cube_to_sphere_scipy(cube_faces, width, height, order=1):
    """
    Converts cube faces to an equirectangular image using SciPy for interpolation.
    (Version with back face flip RESTORED, right face flip REMOVED)
    """
    sphere_image = np.zeros((height, width, 3), dtype=np.float32)

    # --- Pre-process faces ---
    # Restore flip for 'back', keep 'right' unflipped.
    # This combination seems necessary based on the mirroring observed.
    if 'back' in cube_faces:
        # <<< --- RE-ADD THIS LINE --- >>>
        cube_faces['back'] = cube_faces['back'].transpose(Image.FLIP_LEFT_RIGHT)
    # if 'right' in cube_faces:
    #     # <<< --- KEEP THIS COMMENTED --- >>>
    #     cube_faces['right'] = cube_faces['right'].transpose(Image.FLIP_LEFT_RIGHT)

    # Now, face_arrays will contain the correctly pre-processed faces
    face_arrays = {k: np.array(v) for k, v in cube_faces.items()}

    # Assume all faces have the same dimensions for simplicity here
    # (Add checks/handling if they can differ)
    if not face_arrays:
         raise ValueError("No cube faces provided.")
    face_h, face_w = next(iter(face_arrays.values())).shape[:2]


    # --- Calculate spherical coordinates (same as before) ---
    jj, ii = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    theta = (0.5 - jj / height) * np.pi
    phi = (ii / width) * 2 * np.pi
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta)
    z = np.cos(theta) * np.cos(phi)

    # --- Map Cartesian coordinates (same as before) ---
    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
    max_abs = np.maximum.reduce([abs_x, abs_y, abs_z])
    epsilon = 1e-12
    mask_front = (np.abs(z - max_abs) < epsilon) & (z > 0)
    mask_back  = (np.abs(z + max_abs) < epsilon) & (z < 0)
    mask_right = (np.abs(x - max_abs) < epsilon) & (x > 0)
    mask_left  = (np.abs(x + max_abs) < epsilon) & (x < 0)
    mask_top   = (np.abs(y - max_abs) < epsilon) & (y > 0)
    mask_bottom= (np.abs(y + max_abs) < epsilon) & (y < 0)

    masks = {
        'front': mask_front, 'back': mask_back,
        'left': mask_left, 'right': mask_right,
        'top': mask_top, 'bottom': mask_bottom
    }

    # --- Perform interpolation (same as before) ---
    # The u,v formulas will now work correctly because face_arrays['back']
    # has been pre-flipped, while face_arrays['right'] has not.
    for face_name, mask in masks.items():
        if face_name not in face_arrays or not np.any(mask):
            continue

        face_array = face_arrays[face_name] # Gets the correctly pre-processed face
        h, w = face_array.shape[:2]
        face_x, face_y, face_z = x[mask], y[mask], z[mask]
        u, v = None, None

        # Use the same u,v formulas as the previous version
        if face_name == 'front':   # +z
            u = 0.5 * (face_x / face_z + 1)
            v = 0.5 * (-face_y / face_z + 1)
        elif face_name == 'back':    # -z
            # Formula expects pre-flipped texture (which we restored)
            u = 0.5 * (face_x / (-face_z) + 1)
            v = 0.5 * (-face_y / (-face_z) + 1)
        elif face_name == 'right':   # +x
            # Formula expects unflipped texture
            u = 0.5 * ((-face_z) / face_x + 1)
            v = 0.5 * (-face_y / face_x + 1)
        elif face_name == 'left':    # -x
            # Assuming this also expects unflipped texture
            u = 0.5 * (face_z / (-face_x) + 1)
            v = 0.5 * (-face_y / (-face_x) + 1)
        elif face_name == 'top':     # +y
            u = 0.5 * (face_x / face_y + 1)
            v = 0.5 * (face_z / face_y + 1)
        elif face_name == 'bottom':  # -y
            u = 0.5 * (face_x / (-face_y) + 1)
            v = 0.5 * ((-face_z) / (-face_y) + 1)

        if u is not None and v is not None:
            px = u * (w - 1)
            py = v * (h - 1)
            coords = np.stack([py, px])

            for channel in range(3):
                interpolated_values = map_coordinates(
                    face_array[:, :, channel],
                    coords,
                    order=order,
                    mode='nearest',
                    prefilter=(order > 1)
                )
                sphere_image[mask, channel] = interpolated_values

    sphere_image = np.clip(sphere_image, 0, 255)
    return Image.fromarray(sphere_image.astype(np.uint8))

# --- Helper function similar to yours ---
def cube_to_panorama_pil_scipy(front, right, back, left, top, bottom=None, width=3216, height=1608, order=1):
    """Wraps cube_to_sphere_scipy."""
    cube_faces = {
        'front': front,
        'back': back,
        'left': left,
        'right': right,
        'top': top,
    }
    if bottom:
        cube_faces['bottom'] = bottom

    # Ensure all inputs are PIL Images (add error handling if needed)
    for k, v in cube_faces.items():
        if not isinstance(v, Image.Image):
             raise TypeError(f"Face '{k}' is not a PIL Image.")

    return cube_to_sphere_scipy(cube_faces, width, height, order=order)


def extract_perspective_view_scipy(
    equi_img: Image.Image,
    fov_h: float = 120.0,
    fov_v: float = 60.0,
    center_azimuth: float = 0.0, # Convention: 0=front, 90=right, 180=back, 270=left
    center_elevation: float = 0.0, # Convention: 0=horizon, positive=up
    out_w: int = 1024,
    out_h: int = 512,
    interpolation_order: int = 3 # Default to Bicubic (order=3) for higher quality
) -> Image.Image:
    """
    Extracts a perspective view from an equirectangular panorama image using SciPy.

    Args:
        equi_img: The input PIL Image (equirectangular panorama).
        fov_h: Desired horizontal field of view in degrees.
        fov_v: Desired vertical field of view in degrees.
        center_azimuth: Azimuth (yaw) angle for the center of the view (degrees).
        center_elevation: Elevation (pitch) angle for the center of the view (degrees).
        out_w: Width of the output perspective image in pixels.
        out_h: Height of the output perspective image in pixels.
        interpolation_order: Order for map_coordinates (0=Nearest, 1=Bilinear, 3=Bicubic).
                             Default is 3. Use 1 for faster bilinear interpolation.

    Returns:
        A PIL Image containing the perspective view.
    """
    # 1. Convert PIL image to NumPy array, ensure float type for interpolation
    # Using float32 is usually sufficient and memory-friendly
    equi_arr = np.asarray(equi_img, dtype=np.float32)
    H, W = equi_arr.shape[:2]
    num_channels = equi_arr.shape[2] if equi_arr.ndim == 3 else 1 # Handle grayscale

    # 2. Create normalized coordinate grid for the output image (-1 to 1)
    ys, xs = np.meshgrid(
        np.linspace(-1, 1, num=out_h, dtype=np.float32),
        np.linspace(-1, 1, num=out_w, dtype=np.float32),
        indexing='ij' # ys changes along axis 0 (rows), xs along axis 1 (columns)
    )

    # 3. Convert output normalized coords (xs, ys) to spherical coords (yaw, pitch)
    fov_h_rad = np.radians(fov_h)
    fov_v_rad = np.radians(fov_v)
    center_azimuth_rad = np.radians(center_azimuth)
    center_elevation_rad = np.radians(center_elevation)

    yaw = center_azimuth_rad + xs * (fov_h_rad / 2.0)
    pitch = center_elevation_rad + ys * (fov_v_rad / 2.0)
    pitch = np.clip(pitch, -np.pi / 2.0, np.pi / 2.0) # Clamp pitch
    phi = yaw % (2 * np.pi) # Wrap yaw/longitude to [0, 2*pi)

    # 4. Convert spherical coords (phi, pitch) to source pixel coords (x_src, y_src)
    x_src = (phi / (2 * np.pi)) * (W - 1) # Map phi [0, 2*pi) -> x [0, W-1]
    y_src = ((pitch + np.pi / 2.0) / np.pi) * (H - 1) # Map pitch [-pi/2, pi/2] -> y [0, H-1]

    # 5. Prepare coordinates for map_coordinates (shape must be (2, N))
    # Need y coordinates first (rows), then x coordinates (columns)
    coords = np.stack([y_src.ravel(), x_src.ravel()])

    # 6. Perform interpolation using map_coordinates for each channel
    persp_channels = []
    for c in range(num_channels):
        # Select the channel data (works for both grayscale and color)
        channel_data = equi_arr[..., c] if num_channels > 1 else equi_arr

        interpolated_channel = map_coordinates(
            channel_data,           # Input channel data (float32)
            coords,                 # Source coordinates for sampling (float32)
            order=interpolation_order, # Use specified interpolation order
            mode='wrap',            # Handles horizontal wrapping
            prefilter=(interpolation_order > 1) # Recommended for order > 1
        )
        persp_channels.append(interpolated_channel)

    # Stack interpolated channels back together
    if num_channels > 1:
        persp_arr_float = np.stack(persp_channels, axis=-1)
    else:
        persp_arr_float = persp_channels[0] # Grayscale case

    # 7. Reshape flattened array back to image dimensions
    persp_arr_float = persp_arr_float.reshape(out_h, out_w, num_channels) if num_channels > 1 else persp_arr_float.reshape(out_h, out_w)

    # 8. Clip values and convert back to uint8 for PIL image
    persp_arr_uint8 = np.clip(persp_arr_float, 0, 255).astype(np.uint8)

    # 9. Convert back to PIL Image
    return Image.fromarray(persp_arr_uint8)



def panorama_to_cube(panorama_path, output_folder, face_size=1024):
    """
    Convert a full equirectangular panorama image into six cube faces.
    
    Parameters:
      panorama_path (str): File path to the equirectangular image.
      output_folder (str): Directory where the cube face images will be saved.
      face_size (int): The width and height (in pixels) of each cube face image.
    
    Returns:
      dict: A dictionary mapping face names ('front', 'right', 'back', 'left', 'top', 'bottom')
            to their corresponding cube face images (as NumPy arrays).
    """
    # Read the equirectangular panorama image
    equirect_img = cv2.imread(panorama_path)
    if equirect_img is None:
        raise ValueError("Could not read the panorama image from the given path.")
    
    # Define the mapping for each cube face.
    # Each face uses a 90° field-of-view in both horizontal and vertical directions.
    faces = {
        'front':  {'center_azimuth': 0,   'center_elevation': 0},
        'right':  {'center_azimuth': 90,  'center_elevation': 0},
        'back':   {'center_azimuth': 180, 'center_elevation': 0},
        'left':   {'center_azimuth': 270, 'center_elevation': 0},
        'top':    {'center_azimuth': 0,   'center_elevation': -90},
        'bottom': {'center_azimuth': 0,   'center_elevation': 90},
    }
    
    cube_faces = {}
    for face, params in faces.items():
        cube_faces[face] = extract_perspective_view(
            equirect_img,
            fov_h=90, 
            fov_v=90,
            center_azimuth=params['center_azimuth'],
            center_elevation=params['center_elevation'],
            out_w=face_size,
            out_h=face_size
        )
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save each cube face image
    for face, img in cube_faces.items():
        cv2.imwrite(os.path.join(output_folder, f"{face}.png"), img)
    
    return cube_faces
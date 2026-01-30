import cv2
import numpy as np

def resize_image(image, width=None, height=None):
    """
    Resize the image to the specified width or height, maintaining aspect ratio.
    """
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def rgb_to_hex(rgb):
    """
    Convert RGB tuple to Hex string.
    """
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def visualize_mask(image, mask):
    """
    Overlays the mask on the image for visualization.
    """
    # Create a red overlay
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    output = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    return output

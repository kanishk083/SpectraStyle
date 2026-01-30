import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_skin_tone(image, mask):
    """
    Extracts the dominant skin tone using the provided mask.
    Filters out shadow/dark pixels and low-saturation (grey) pixels.
    Returns RGB tuple and LAB tuple.
    """
    # Extract non-zero pixels (skin pixels)
    skin_pixels_indices = np.where(mask > 0)
    
    # If no skin found, return defaults
    if len(skin_pixels_indices[0]) == 0:
        return (0, 0, 0), (0, 0, 0)
        
    skin_pixels_bgr = image[skin_pixels_indices]
    
    # === SHADOW & GREY FILTERING ===
    skin_pixels_bgr_reshaped = skin_pixels_bgr.reshape(-1, 1, 3)
    skin_pixels_hsv = cv2.cvtColor(skin_pixels_bgr_reshaped, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    h, s, v = skin_pixels_hsv[:, 0], skin_pixels_hsv[:, 1], skin_pixels_hsv[:, 2]
    
    # Filter: Remove dark and grey pixels
    v_threshold = np.percentile(v, 20)
    brightness_mask = v > v_threshold
    saturation_mask = s > 30
    valid_mask = brightness_mask & saturation_mask
    
    filtered_bgr = skin_pixels_bgr[valid_mask]
    
    if len(filtered_bgr) < 50:
        filtered_bgr = skin_pixels_bgr
    
    # Convert to RGB for K-Means
    filtered_rgb = filtered_bgr[..., ::-1]
    
    kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
    kmeans.fit(filtered_rgb)
    
    dominant_rgb = kmeans.cluster_centers_[0].astype(int)
    
    # Convert to LAB for better undertone analysis
    dominant_bgr_reshaped = np.uint8([[dominant_rgb[::-1]]])  # RGB to BGR
    dominant_lab = cv2.cvtColor(dominant_bgr_reshaped, cv2.COLOR_BGR2LAB)[0][0]
    
    return tuple(dominant_rgb), tuple(dominant_lab)

def extract_hair_color_robust(image, hair_mask, eyebrow_mask=None):
    """
    Robust hair color extraction using multi-cluster analysis.
    Handles glare/highlights by selecting the darkest significant cluster.
    
    Algorithm:
    1. K-Means with k=3 clusters (Highlight, Mid-tone, Shadow/True Color)
    2. Select darkest cluster with >= 20% area (Darkness Priority Rule)
    3. If eyebrow_mask provided, verify against eyebrow color
    
    Returns RGB tuple and LAB tuple.
    """
    pixels_indices = np.where(hair_mask > 0)
    
    if len(pixels_indices[0]) == 0:
        return (0, 0, 0), (0, 0, 0)
    
    pixels_bgr = image[pixels_indices]
    pixels_rgb = pixels_bgr[..., ::-1]
    
    # === STEP 1: Multi-Cluster Extraction (k=3) ===
    n_clusters = min(3, len(pixels_rgb))  # Handle small regions
    if n_clusters < 2:
        # Fallback for very small regions
        dominant_rgb = np.mean(pixels_rgb, axis=0).astype(int)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels_rgb)
        
        # Get cluster centers and their counts
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Convert centers to LAB for luminance comparison
        cluster_info = []
        for i, center in enumerate(centers):
            center_bgr = np.uint8([[center[::-1]]])
            center_lab = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2LAB)[0][0]
            luminance = center_lab[0]
            count = np.sum(labels == i)
            percentage = count / len(labels)
            cluster_info.append({
                'idx': i,
                'rgb': center,
                'lab': center_lab,
                'luminance': luminance,
                'percentage': percentage
            })
        
        # === STEP 2: Darkness Priority Rule ===
        # Sort by luminance (darkest first)
        cluster_info.sort(key=lambda x: x['luminance'])
        
        # Select darkest cluster with at least 20% coverage
        selected = None
        for cluster in cluster_info:
            if cluster['percentage'] >= 0.2:
                selected = cluster
                break
        
        # Fallback: if no cluster has 20%, take the darkest anyway
        if selected is None:
            selected = cluster_info[0]
        
        dominant_rgb = selected['rgb'].astype(int)
        hair_lab = tuple(selected['lab'])
    
    # Convert final RGB to LAB
    dominant_bgr_reshaped = np.uint8([[dominant_rgb[::-1]]])
    hair_lab = cv2.cvtColor(dominant_bgr_reshaped, cv2.COLOR_BGR2LAB)[0][0]
    
    # === STEP 3: Eyebrow Verification (Truth Check) ===
    if eyebrow_mask is not None:
        brow_pixels_indices = np.where(eyebrow_mask > 0)
        if len(brow_pixels_indices[0]) > 50:
            brow_pixels_bgr = image[brow_pixels_indices]
            brow_pixels_rgb = brow_pixels_bgr[..., ::-1]
            
            # Get eyebrow color (simple mean - less affected by glare)
            brow_rgb = np.mean(brow_pixels_rgb, axis=0).astype(int)
            brow_bgr = np.uint8([[brow_rgb[::-1]]])
            brow_lab = cv2.cvtColor(brow_bgr, cv2.COLOR_BGR2LAB)[0][0]
            
            hair_luminance = hair_lab[0]
            brow_luminance = brow_lab[0]
            
            # If hair is >20 units brighter than eyebrows, use eyebrow color
            # (Assumption: natural hair is never lighter than eyebrows)
            if hair_luminance > brow_luminance + 20:
                dominant_rgb = brow_rgb
                hair_lab = brow_lab
    
    return tuple(dominant_rgb), tuple(hair_lab)


def extract_hair_color(image, mask):
    """
    Backward-compatible wrapper for robust hair extraction.
    Returns RGB tuple and LAB tuple.
    """
    return extract_hair_color_robust(image, mask, eyebrow_mask=None)

def extract_eye_color(image, mask):
    """
    Extracts dominant eye/iris color from the eye mask region.
    Returns RGB tuple and LAB tuple.
    """
    pixels_indices = np.where(mask > 0)
    
    if len(pixels_indices[0]) == 0:
        return (0, 0, 0), (0, 0, 0)
    
    pixels_bgr = image[pixels_indices]
    
    # Filter out very dark pixels (pupil) and very light (sclera)
    pixels_lab = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    l_channel = pixels_lab[:, 0]
    
    # Keep mid-range luminance (iris, not pupil or sclera)
    luminance_mask = (l_channel > 30) & (l_channel < 200)
    filtered_bgr = pixels_bgr[luminance_mask]
    
    if len(filtered_bgr) < 10:
        filtered_bgr = pixels_bgr
    
    filtered_rgb = filtered_bgr[..., ::-1]
    
    kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
    kmeans.fit(filtered_rgb)
    
    dominant_rgb = kmeans.cluster_centers_[0].astype(int)
    
    dominant_bgr_reshaped = np.uint8([[dominant_rgb[::-1]]])
    dominant_lab = cv2.cvtColor(dominant_bgr_reshaped, cv2.COLOR_BGR2LAB)[0][0]
    
    return tuple(dominant_rgb), tuple(dominant_lab)

def calculate_contrast(skin_lab, hair_lab, eye_lab):
    """
    Calculates contrast level based on luminance differences.
    Uses L* channel from CIELAB.
    Returns: 'High', 'Medium', or 'Low'
    """
    skin_l = skin_lab[0]
    hair_l = hair_lab[0]
    eye_l = eye_lab[0]
    
    # Calculate max difference between any two features
    diff_skin_hair = abs(skin_l - hair_l)
    diff_skin_eye = abs(skin_l - eye_l)
    diff_hair_eye = abs(hair_l - eye_l)
    
    max_diff = max(diff_skin_hair, diff_skin_eye, diff_hair_eye)
    
    # High contrast: > 60 L* difference
    # Medium: 30-60
    # Low: < 30
    if max_diff > 60:
        return "High"
    elif max_diff > 30:
        return "Medium"
    else:
        return "Low"

def analyze_undertone_lab(skin_lab):
    """
    Analyzes skin undertone using CIELAB color space.
    For dark skin (L < 40), uses b-channel (blue-yellow) instead of a-channel.
    Returns: undertone ('Warm'/'Cool'), depth ('Light'/'Medium'/'Deep')
    """
    l, a, b = skin_lab
    
    # L = Lightness (0-100 in theory, 0-255 in OpenCV)
    # a = Green-Red axis (negative = green, positive = red)
    # b = Blue-Yellow axis (negative = blue, positive = yellow)
    
    # Depth Classification based on L*
    if l > 170:
        depth = "Light"
    elif l > 100:
        depth = "Medium"
    else:
        depth = "Deep"
    
    # Undertone Classification
    # For DARK SKIN (L < 100 in OpenCV scale): Use b-channel (blue-yellow)
    # Yellow (high b) = Warm, Blue-ish (low b) = Cool
    # For LIGHTER SKIN: Use traditional a+b balance
    
    if l < 100:  # Dark skin - use b-channel primarily
        # b > 128 means yellow undertone (warm)
        # b < 128 means blue undertone (cool)
        if b > 135:
            undertone = "Warm"
        else:
            undertone = "Cool"
    else:  # Medium/Light skin - use combined approach
        # Warm: higher a (red) AND higher b (yellow)
        # Cool: lower a (green-ish) OR lower b (blue-ish)
        warmth_score = (a - 128) + (b - 128)
        if warmth_score > 10:
            undertone = "Warm"
        else:
            undertone = "Cool"
    
    return undertone, depth

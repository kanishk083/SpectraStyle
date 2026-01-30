def recommend_12_season(undertone, depth, contrast):
    """
    Determines the 12-Season palette using the Flow System.
    
    The Flow System categorizes based on dominant characteristic:
    - Flow 1 (Depth): Deep Winter / Deep Autumn
    - Flow 2 (Light): Light Summer / Light Spring  
    - Flow 3 (Soft): Soft Summer / Soft Autumn
    - Flow 4 (Clear): Clear Winter / Clear Spring
    - Flow 5 (Cool): Cool Summer / Cool Winter
    - Flow 6 (Warm): Warm Autumn / Warm Spring
    
    Args:
        undertone: 'Warm' or 'Cool'
        depth: 'Light', 'Medium', or 'Deep'
        contrast: 'High', 'Medium', or 'Low'
    
    Returns:
        season: One of 12 seasons (e.g., 'deep_winter', 'light_spring')
    """
    
    # Flow 1: DEEP (Dark hair, rich coloring)
    if depth == "Deep":
        if undertone == "Cool":
            return "deep_winter"
        else:
            return "deep_autumn"
    
    # Flow 2: LIGHT (Light hair, light eyes, fair skin)
    if depth == "Light":
        if undertone == "Cool":
            return "light_summer"
        else:
            return "light_spring"
    
    # For Medium depth, use contrast and undertone
    
    # Flow 4: CLEAR (High contrast, bright coloring)
    if contrast == "High":
        if undertone == "Cool":
            return "clear_winter"
        else:
            return "clear_spring"
    
    # Flow 3: SOFT (Low contrast, muted coloring)
    if contrast == "Low":
        if undertone == "Cool":
            return "soft_summer"
        else:
            return "soft_autumn"
    
    # Flow 5 & 6: Pure COOL or WARM (Medium contrast, medium depth)
    if undertone == "Cool":
        return "cool_summer"  # True Summer or True Winter based on other factors
    else:
        return "warm_autumn"  # True Autumn or True Spring

def recommend_season(undertone, depth, contrast=None):
    """
    Wrapper for backward compatibility.
    If contrast is provided, uses 12-season system.
    Otherwise, falls back to 4-season system.
    """
    if contrast is not None:
        return recommend_12_season(undertone, depth, contrast)
    
    # 4-Season fallback
    if undertone == "Cool":
        if depth in ["Light", "Medium"]:
            return "summer"
        else:
            return "winter"
    else:
        if depth in ["Light", "Medium"]:
            return "spring"
        else:
            return "autumn"

def recommend_shape_style(face_ratio):
    """
    Recommends neckline/style based on face width/height ratio.
    """
    if face_ratio > 0.9:
        shape = "Round/Square (Wide)"
        necklines = ["V-Neck", "Scoop Neck", "Sweetheart"]
    elif face_ratio < 0.75:
        shape = "Oblong/Rectangular (Long)"
        necklines = ["Crew Neck", "Boat Neck", "High Neck"]
    else:
        shape = "Oval (Balanced)"
        necklines = ["Any Neckline", "Off-Shoulder", "Square Neck"]
        
    return shape, necklines

def analyze_user_profile(skin_lab, hair_lab, eye_lab):
    """
    Complete user profile analysis for 12-season system.
    
    Returns:
        dict with undertone, depth, contrast, season, and season_details
    """
    from .color_analysis import analyze_undertone_lab, calculate_contrast
    
    undertone, depth = analyze_undertone_lab(skin_lab)
    contrast = calculate_contrast(skin_lab, hair_lab, eye_lab)
    season = recommend_12_season(undertone, depth, contrast)
    
    # Season display names
    season_names = {
        "deep_winter": "Deep Winter",
        "deep_autumn": "Deep Autumn",
        "light_summer": "Light Summer",
        "light_spring": "Light Spring",
        "soft_summer": "Soft Summer",
        "soft_autumn": "Soft Autumn",
        "clear_winter": "Clear Winter",
        "clear_spring": "Clear Spring",
        "cool_summer": "Cool Summer",
        "cool_winter": "Cool Winter",
        "warm_autumn": "Warm Autumn",
        "warm_spring": "Warm Spring"
    }
    
    return {
        "undertone": undertone,
        "depth": depth,
        "contrast": contrast,
        "season_key": season,
        "season_name": season_names.get(season, season.replace("_", " ").title())
    }

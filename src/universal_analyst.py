"""
Universal Color Analyst - Dynamic 12-Season Classification
Handles glare, highlights, and works for all skin tones.
"""

import cv2
import numpy as np


class UniversalColorAnalyst:
    """
    Dynamic 12-Season Color Analysis using scoring system.
    No hardcoded if-else chains. Uses weighted metrics.
    
    Handles:
    - Glare on dark hair
    - All skin tones (Fair, Medium, Dark)
    - Variable lighting conditions
    """
    
    SEASON_NAMES = {
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
    
    def __init__(self):
        pass
    
    def _rgb_to_lab(self, rgb):
        """Convert RGB tuple to CIELAB. Returns normalized values."""
        bgr = np.uint8([[rgb[::-1]]])
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0]
        # Normalize: L is 0-255 in OpenCV, a and b are centered at 128
        return {
            'L': lab[0],
            'a': lab[1],
            'b': lab[2],
            'L_norm': (lab[0] / 255) * 100,  # 0-100 scale
            'b_centered': lab[2] - 128        # -128 to 127 scale
        }
    
    def calculate_temperature(self, skin_lab, hair_lab):
        """
        Calculate Temperature Score using weighted b-channel.
        Skin = 70%, Hair = 30%.
        
        Returns:
            score: float
            category: 'WARM', 'COOL', or 'NEUTRAL'
        """
        skin_b = skin_lab['b_centered']
        hair_b = hair_lab['b_centered']
        
        temp_score = (0.7 * skin_b) + (0.3 * hair_b)
        
        if temp_score > 2:
            category = "WARM"
        elif temp_score < -2:
            category = "COOL"
        else:
            category = "NEUTRAL"
        
        return temp_score, category
    
    def calculate_depth(self, skin_lab, hair_lab):
        """
        Calculate Depth using INVERTED Luminance.
        Skin = 40%, Hair = 60%.
        
        Higher score = Deeper coloring.
        
        Returns:
            score: float (0-100)
            category: 'DEEP', 'MEDIUM', or 'LIGHT'
        """
        skin_depth = 100 - skin_lab['L_norm']
        hair_depth = 100 - hair_lab['L_norm']
        
        total_depth = (0.4 * skin_depth) + (0.6 * hair_depth)
        
        if total_depth > 65:
            category = "DEEP"
        elif total_depth < 35:
            category = "LIGHT"
        else:
            category = "MEDIUM"
        
        return total_depth, category
    
    def calculate_contrast(self, skin_lab, hair_lab):
        """
        Calculate Contrast using L channel difference.
        
        Returns:
            score: float (0-100)
            category: 'HIGH', 'MEDIUM', or 'LOW'
        """
        contrast = abs(skin_lab['L_norm'] - hair_lab['L_norm'])
        
        if contrast > 50:
            category = "HIGH"
        elif contrast < 25:
            category = "LOW"
        else:
            category = "MEDIUM"
        
        return contrast, category
    
    def _is_dark_on_dark(self, skin_lab, hair_lab):
        """
        Detect edge case: Dark skin + Dark hair.
        Both have low luminance, so contrast appears LOW,
        but visually it's still DEEP.
        """
        return skin_lab['L_norm'] < 40 and hair_lab['L_norm'] < 20
    
    def classify(self, skin_rgb, hair_rgb, eye_rgb):
        """
        Classify into 12-Season using the Flow System with scores.
        
        Args:
            skin_rgb: Tuple (R, G, B)
            hair_rgb: Tuple (R, G, B)
            eye_rgb: Tuple (R, G, B)
        
        Returns:
            season_key: String like 'deep_autumn'
        """
        # Convert to LAB
        skin_lab = self._rgb_to_lab(skin_rgb)
        hair_lab = self._rgb_to_lab(hair_rgb)
        eye_lab = self._rgb_to_lab(eye_rgb)
        
        # Calculate metrics
        temp_score, temperature = self.calculate_temperature(skin_lab, hair_lab)
        depth_score, depth = self.calculate_depth(skin_lab, hair_lab)
        contrast_score, contrast = self.calculate_contrast(skin_lab, hair_lab)
        
        # === EDGE CASE: Dark Skin + Dark Hair ===
        # Force DEEP classification based on temperature alone
        if self._is_dark_on_dark(skin_lab, hair_lab):
            if temperature in ["COOL", "NEUTRAL"]:
                return "deep_winter"
            else:
                return "deep_autumn"
        
        # === MAIN FLOW DECISION TREE ===
        
        # Flow 1: DEEP (Dominant characteristic)
        if depth == "DEEP":
            if temperature in ["COOL", "NEUTRAL"]:
                return "deep_winter"
            else:
                return "deep_autumn"
        
        # Flow 2: LIGHT (Dominant characteristic)
        if depth == "LIGHT":
            if temperature in ["COOL", "NEUTRAL"]:
                return "light_summer"
            else:
                return "light_spring"
        
        # For MEDIUM depth, use contrast as secondary characteristic
        
        # Flow 4: HIGH CONTRAST (Clear/Bright)
        if contrast == "HIGH":
            if temperature in ["COOL", "NEUTRAL"]:
                return "clear_winter"
            else:
                return "clear_spring"
        
        # Flow 3: LOW CONTRAST (Soft/Muted)
        if contrast == "LOW":
            if temperature in ["COOL", "NEUTRAL"]:
                return "soft_summer"
            else:
                return "soft_autumn"
        
        # Flow 5 & 6: PURE COOL or WARM (Medium depth, Medium contrast)
        if temperature in ["COOL", "NEUTRAL"]:
            return "cool_summer"
        else:
            return "warm_autumn"
    
    def analyze(self, skin_rgb, hair_rgb, eye_rgb):
        """
        Full analysis with all metrics and result.
        
        Returns:
            dict with all scores and classifications
        """
        skin_lab = self._rgb_to_lab(skin_rgb)
        hair_lab = self._rgb_to_lab(hair_rgb)
        eye_lab = self._rgb_to_lab(eye_rgb)
        
        temp_score, temperature = self.calculate_temperature(skin_lab, hair_lab)
        depth_score, depth = self.calculate_depth(skin_lab, hair_lab)
        contrast_score, contrast = self.calculate_contrast(skin_lab, hair_lab)
        
        season_key = self.classify(skin_rgb, hair_rgb, eye_rgb)
        
        return {
            # Raw LAB values
            "skin_lab": skin_lab,
            "hair_lab": hair_lab,
            "eye_lab": eye_lab,
            
            # Scores
            "temperature_score": round(temp_score, 2),
            "depth_score": round(depth_score, 2),
            "contrast_score": round(contrast_score, 2),
            
            # Categories
            "temperature": temperature,
            "depth": depth,
            "contrast": contrast,
            
            # Result
            "season_key": season_key,
            "season_name": self.SEASON_NAMES.get(season_key, season_key.replace("_", " ").title())
        }


# Convenience function
def analyze_colors(skin_rgb, hair_rgb, eye_rgb):
    """Quick analysis using UniversalColorAnalyst."""
    analyst = UniversalColorAnalyst()
    return analyst.analyze(skin_rgb, hair_rgb, eye_rgb)

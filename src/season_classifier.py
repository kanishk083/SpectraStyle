"""
Universal Color Analysis Engine - 12-Season System
Works for ALL skin tones without race-specific thresholds.
"""

import cv2
import numpy as np


class SeasonClassifier:
    """
    A dynamic 12-Season Color Analysis classifier.
    Accepts Skin, Hair, and Eye RGB values and determines the seasonal palette.
    Works universally for Fair, Medium, and Dark skin tones.
    """
    
    # Season display names mapping
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
        """
        Convert RGB tuple to CIELAB color space.
        Returns (L, a, b) where:
        - L: Lightness (0-100 in theory, 0-255 in OpenCV)
        - a: Green-Red axis
        - b: Blue-Yellow axis (Temperature indicator)
        """
        # RGB to BGR for OpenCV
        bgr = np.uint8([[rgb[::-1]]])
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0]
        
        # OpenCV LAB: L is 0-255 (scaled from 0-100)
        # a and b are 0-255 (centered at 128 instead of 0)
        return tuple(lab)
    
    def _calculate_temperature(self, skin_lab, hair_lab):
        """
        Calculate Temperature Score using weighted average of b-channel.
        Skin contributes 70%, Hair contributes 30%.
        
        Returns: 'WARM', 'COOL', or 'NEUTRAL'
        """
        skin_b = skin_lab[2]  # b-channel
        hair_b = hair_lab[2]
        
        # Center b at 0 (OpenCV uses 128 as center)
        skin_b_centered = skin_b - 128
        hair_b_centered = hair_b - 128
        
        # Weighted average: Skin 70%, Hair 30%
        temp_score = (0.7 * skin_b_centered) + (0.3 * hair_b_centered)
        
        if temp_score < -2:
            return "COOL"
        elif temp_score > 2:
            return "WARM"
        else:
            return "NEUTRAL"
    
    def _calculate_value_depth(self, skin_lab):
        """
        Calculate Value/Depth using inverted L channel.
        Value = 100 - (L * 100 / 255) to normalize to 0-100 scale.
        
        Returns: 'DEEP', 'MEDIUM', or 'LIGHT'
        """
        skin_l = skin_lab[0]
        
        # Normalize L from 0-255 to 0-100
        l_normalized = (skin_l / 255) * 100
        
        # Invert: higher value means darker skin
        depth_value = 100 - l_normalized
        
        if depth_value > 60:
            return "DEEP"
        elif depth_value < 40:
            return "LIGHT"
        else:
            return "MEDIUM"
    
    def _calculate_contrast(self, skin_lab, hair_lab):
        """
        Calculate Contrast Ratio using absolute difference of L channels.
        
        Returns: 'HIGH', 'MEDIUM', or 'LOW'
        """
        skin_l = skin_lab[0]
        hair_l = hair_lab[0]
        
        # Normalize to 0-100 scale
        skin_l_norm = (skin_l / 255) * 100
        hair_l_norm = (hair_l / 255) * 100
        
        contrast = abs(skin_l_norm - hair_l_norm)
        
        if contrast > 40:
            return "HIGH"
        elif contrast < 20:
            return "LOW"
        else:
            return "MEDIUM"
    
    def _is_dark_skin_dark_hair(self, skin_lab, hair_lab):
        """
        Special edge case detection for Dark Skin + Black Hair.
        When both are dark, mathematical contrast is LOW but visual appearance is DEEP.
        """
        skin_l_norm = (skin_lab[0] / 255) * 100
        hair_l_norm = (hair_lab[0] / 255) * 100
        
        # Dark skin (L < 40) AND Black hair (L < 20)
        return skin_l_norm < 40 and hair_l_norm < 20
    
    def predict(self, skin_rgb, hair_rgb, eye_rgb):
        """
        Predict the 12-Season palette based on Skin, Hair, and Eye colors.
        
        Args:
            skin_rgb: Tuple (R, G, B) of skin color
            hair_rgb: Tuple (R, G, B) of hair color
            eye_rgb: Tuple (R, G, B) of eye color
        
        Returns:
            season_key: String like 'deep_autumn', 'light_spring', etc.
        """
        # Convert to LAB
        skin_lab = self._rgb_to_lab(skin_rgb)
        hair_lab = self._rgb_to_lab(hair_rgb)
        eye_lab = self._rgb_to_lab(eye_rgb)
        
        # Calculate the 3 key metrics
        temperature = self._calculate_temperature(skin_lab, hair_lab)
        depth = self._calculate_value_depth(skin_lab)
        contrast = self._calculate_contrast(skin_lab, hair_lab)
        
        # === EDGE CASE: Dark Skin + Dark Hair ===
        # Override contrast to DEEP based purely on temperature
        if self._is_dark_skin_dark_hair(skin_lab, hair_lab):
            if temperature == "COOL" or temperature == "NEUTRAL":
                return "deep_winter"
            else:
                return "deep_autumn"
        
        # === MAIN DECISION TREE ===
        
        # Flow 1: DEEP colors
        if depth == "DEEP":
            if temperature == "COOL" or temperature == "NEUTRAL":
                return "deep_winter"
            else:
                return "deep_autumn"
        
        # Flow 2: LIGHT colors
        if depth == "LIGHT":
            if temperature == "COOL" or temperature == "NEUTRAL":
                return "light_summer"
            else:
                return "light_spring"
        
        # For MEDIUM depth, use contrast
        
        # Flow 4: HIGH CONTRAST (Clear/Bright)
        if contrast == "HIGH":
            if temperature == "COOL" or temperature == "NEUTRAL":
                return "clear_winter"
            else:
                return "clear_spring"
        
        # Flow 3: LOW CONTRAST (Soft/Muted)
        if contrast == "LOW":
            if temperature == "COOL" or temperature == "NEUTRAL":
                return "soft_summer"
            else:
                return "soft_autumn"
        
        # Flow 5 & 6: MEDIUM everything - Pure Cool/Warm
        if temperature == "COOL" or temperature == "NEUTRAL":
            return "cool_summer"
        else:
            return "warm_autumn"
    
    def predict_with_details(self, skin_rgb, hair_rgb, eye_rgb):
        """
        Predict with full analysis details.
        
        Returns:
            dict with all metrics and result
        """
        skin_lab = self._rgb_to_lab(skin_rgb)
        hair_lab = self._rgb_to_lab(hair_rgb)
        eye_lab = self._rgb_to_lab(eye_rgb)
        
        temperature = self._calculate_temperature(skin_lab, hair_lab)
        depth = self._calculate_value_depth(skin_lab)
        contrast = self._calculate_contrast(skin_lab, hair_lab)
        
        season_key = self.predict(skin_rgb, hair_rgb, eye_rgb)
        
        return {
            "skin_lab": skin_lab,
            "hair_lab": hair_lab,
            "eye_lab": eye_lab,
            "temperature": temperature,
            "depth": depth,
            "contrast": contrast,
            "season_key": season_key,
            "season_name": self.SEASON_NAMES.get(season_key, season_key.replace("_", " ").title())
        }


# Convenience function for direct use
def classify_season(skin_rgb, hair_rgb, eye_rgb):
    """
    Classify the 12-season palette from RGB colors.
    
    Args:
        skin_rgb: Tuple (R, G, B)
        hair_rgb: Tuple (R, G, B)
        eye_rgb: Tuple (R, G, B)
    
    Returns:
        season_key: String like 'deep_autumn'
    """
    classifier = SeasonClassifier()
    return classifier.predict(skin_rgb, hair_rgb, eye_rgb)

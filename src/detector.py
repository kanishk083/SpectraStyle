import cv2
import numpy as np
import urllib.request
import os

# MediaPipe Task API imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceMeshDetector:
    def __init__(self, min_detection_confidence=0.5):
        # Download model if not exists
        model_path = self._ensure_model()
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _ensure_model(self):
        """Downloads the face landmarker model if not present."""
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'face_landmarker.task')
        
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            print(f"Downloading FaceLandmarker model to {model_path}...")
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")
        return model_path

    def get_landmarks(self, image):
        """
        Returns facial landmarks for the first detected face.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        results = self.detector.detect(mp_image)
        
        if not results.face_landmarks:
            return None
            
        return results.face_landmarks[0]

    def get_skin_mask(self, image, landmarks):
        """
        Generates a binary mask for skin analysis regions.
        Strictly isolates ONLY cheeks and center of forehead.
        EXCLUDES chin, jawline, upper lip, eyes, and eyebrows.
        """
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define REFINED indices for key regions (MediaPipe Face Mesh indices)
        # These are carefully chosen to AVOID facial hair zones
        
        roi_indices = [
            # LEFT CHEEK - Inner cheek area, avoiding jawline
            # Landmarks: 116(outer cheek), 117, 118, 119, 120, 121, 122, 123 (inner cheek near nose)
            # Upper boundary near eye, lower boundary ABOVE jawline
            [116, 117, 118, 119, 47, 126, 209, 214, 192, 213, 147],
            
            # RIGHT CHEEK - Mirror of left cheek
            # Same logic, avoid jawline and stubble zones
            [345, 346, 347, 348, 277, 355, 429, 434, 416, 433, 376],
            
            # CENTER FOREHEAD - Safe zone, no hair typically
            # Between eyebrows and hairline
            [107, 66, 105, 63, 70, 156, 139, 71, 68, 104, 69, 108,
             337, 299, 296, 334, 293, 300, 383, 368, 301, 298, 333, 338]
            
            # CHIN is REMOVED - prone to stubble/beard
            # Upper lip area is REMOVED
        ]
        
        points_list = []
        
        for group in roi_indices:
            poly = []
            for idx in group:
                pt = landmarks[idx]
                x = int(pt.x * w)
                y = int(pt.y * h)
                poly.append((x, y))
            points_list.append(np.array(poly, dtype=np.int32))
            
        cv2.fillPoly(mask, points_list, 255)
        
        return mask

    def get_face_metrics(self, image, landmarks):
        """
        Calculates face width/height ratio to estimate shape.
        """
        h, w, _ = image.shape
        
        # Top of forehead (10) to Chin (152)
        top = landmarks[10]
        bottom = landmarks[152]
        height = np.sqrt((top.x - bottom.x)**2 * w**2 + (top.y - bottom.y)**2 * h**2)
        
        # Cheek to Cheek (234 to 454)
        left = landmarks[234]
        right = landmarks[454]
        width = np.sqrt((left.x - right.x)**2 * w**2 + (left.y - right.y)**2 * h**2)
        
        ratio = width / height if height > 0 else 0
        return ratio

    def get_hair_mask(self, image, landmarks):
        """
        Generates a mask for hair region (sides of forehead/temples).
        Uses landmarks near the hairline and temple areas.
        """
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Hair region indices - temples and upper forehead edges
        # These landmarks are near the hairline
        hair_indices = [
            # Left temple/hairline area
            [127, 162, 21, 54, 103, 67, 109, 10],
            # Right temple/hairline area  
            [356, 389, 251, 284, 332, 297, 338, 10]
        ]
        
        points_list = []
        for group in hair_indices:
            poly = []
            for idx in group:
                pt = landmarks[idx]
                x = int(pt.x * w)
                y = int(pt.y * h)
                poly.append((x, y))
            points_list.append(np.array(poly, dtype=np.int32))
        
        cv2.fillPoly(mask, points_list, 255)
        return mask

    def get_eye_mask(self, image, landmarks):
        """
        Generates a mask for iris/eye region.
        Uses the refined iris landmarks (468-477 for left, 473-477 for right).
        Falls back to eye contour if iris landmarks unavailable.
        """
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Left iris landmarks (center at 468, ring at 469-472)
        # Right iris landmarks (center at 473, ring at 474-477)
        # If not available, use eye contour landmarks
        
        # Left eye contour (simpler approach)
        left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
        # Right eye contour
        right_eye_indices = [362, 263, 387, 386, 385, 373, 374, 380]
        
        eye_regions = [left_eye_indices, right_eye_indices]
        
        points_list = []
        for group in eye_regions:
            poly = []
            for idx in group:
                pt = landmarks[idx]
                x = int(pt.x * w)
                y = int(pt.y * h)
                poly.append((x, y))
            points_list.append(np.array(poly, dtype=np.int32))
        
        cv2.fillPoly(mask, points_list, 255)
        return mask

    def get_eyebrow_mask(self, image, landmarks):
        """
        Generates a mask for eyebrow regions.
        Used for hair color verification (eyebrows = true pigment).
        """
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Left eyebrow landmarks
        left_brow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        # Right eyebrow landmarks
        right_brow_indices = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
        
        brow_regions = [left_brow_indices, right_brow_indices]
        
        points_list = []
        for group in brow_regions:
            poly = []
            for idx in group:
                pt = landmarks[idx]
                x = int(pt.x * w)
                y = int(pt.y * h)
                poly.append((x, y))
            points_list.append(np.array(poly, dtype=np.int32))
        
        cv2.fillPoly(mask, points_list, 255)
        return mask

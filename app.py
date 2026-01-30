import streamlit as st
import cv2
import numpy as np
import json
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.detector import FaceMeshDetector
from src.color_analysis import extract_skin_tone, extract_hair_color_robust, extract_eye_color
from src.recommender import recommend_shape_style
from src.universal_analyst import UniversalColorAnalyst
from src.utils import resize_image, rgb_to_hex, visualize_mask
from src.shopping_agent import get_agent_recommendations

# Set page config
st.set_page_config(page_title="Personal Stylist AI - 12 Seasons", layout="wide")

# Load palettes
with open('data/palettes.json', 'r') as f:
    palettes = json.load(f)

# Initialize Detector (Face Mesh)
detector = FaceMeshDetector()

st.title("🎨 Personal Stylist AI - 12 Season Analysis")
st.write("Upload your photo to discover your personalized seasonal color palette using advanced color science!")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("This AI uses the **12-Season Color Analysis System** (Flow System) with CIELAB color science to recommend your perfect wardrobe colors.")

# Main Interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Resize for display and faster processing
    image = resize_image(image, width=600)
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_container_width=True)
    
    with st.spinner('Analyzing Skin, Hair, and Eyes...'):
        landmarks = detector.get_landmarks(image)
        
        if landmarks:
            detection_successful = True
            
            # 1. Extract Skin
            skin_mask = detector.get_skin_mask(image, landmarks)
            skin_rgb, skin_lab = extract_skin_tone(image, skin_mask)
            skin_hex = rgb_to_hex(skin_rgb)
            
            # 2. Extract Hair (with eyebrow verification for glare handling)
            hair_mask = detector.get_hair_mask(image, landmarks)
            eyebrow_mask = detector.get_eyebrow_mask(image, landmarks)
            hair_rgb, hair_lab = extract_hair_color_robust(image, hair_mask, eyebrow_mask)
            hair_hex = rgb_to_hex(hair_rgb)
            
            # 3. Extract Eyes
            eye_mask = detector.get_eye_mask(image, landmarks)
            eye_rgb, eye_lab = extract_eye_color(image, eye_mask)
            eye_hex = rgb_to_hex(eye_rgb)
            
            # Visualize combined mask
            combined_mask = cv2.add(skin_mask, cv2.add(hair_mask, eye_mask))
            masked_vis = visualize_mask(image.copy(), combined_mask)
            with col_img2:
                st.image(cv2.cvtColor(masked_vis, cv2.COLOR_BGR2RGB), caption='Analysis Regions', use_container_width=True)
            
            # 4. Analyze Profile (12-Season) using UniversalColorAnalyst
            analyst = UniversalColorAnalyst()
            profile = analyst.analyze(skin_rgb, hair_rgb, eye_rgb)
            
            # 5. Get Palette
            season_key = profile['season_key']
            palette = palettes.get(season_key, palettes.get('autumn', {}))
            
            # 6. Face Shape Analysis
            ratio = detector.get_face_metrics(image, landmarks)
            face_shape, necklines = recommend_shape_style(ratio)
            
        else:
            detection_successful = False
            st.error("No face detected! Please upload a clear portrait.")

    if detection_successful:
        st.success("Analysis Complete!")
        
        st.write("---")
        
        # Feature Colors Section
        st.subheader("🔬 Detected Feature Colors")
        feat1, feat2, feat3 = st.columns(3)
        
        with feat1:
            st.color_picker("Skin Tone", skin_hex, disabled=True, key="skin_color")
            st.caption(f"Skin: {skin_hex}")
            
        with feat2:
            st.color_picker("Hair Color", hair_hex, disabled=True, key="hair_color")
            st.caption(f"Hair: {hair_hex}")
            
        with feat3:
            st.color_picker("Eye Color", eye_hex, disabled=True, key="eye_color")
            st.caption(f"Eyes: {eye_hex}")
        
        st.write("---")
        
        # Analysis Results
        st.subheader("📊 Color Analysis Results")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.metric("Temperature", profile['temperature'])
        with c2:
            st.metric("Depth", profile['depth'])
        with c3:
            st.metric("Contrast", profile['contrast'])
        with c4:
            st.metric("Season", profile['season_name'])
        
        st.write("---")
        
        # Season Result
        st.subheader(f"🌸 Your Season: {profile['season_name']}")
        st.write(palette.get('description', ''))
        st.write(f"**Strategy:** {palette.get('contrast_strategy', '')}")
        
        # Color Palette - Power Colors
        st.subheader(" Your Power Colors")
        colors = palette.get('power_colors', [])
        color_names = palette.get('power_color_names', [])
        if colors:
            cols = st.columns(len(colors))
            for i, color in enumerate(colors):
                with cols[i]:
                    st.color_picker(f"", color, disabled=True, key=f"power_{i}")
                    name = color_names[i] if i < len(color_names) else color
                    st.caption(name)
        
        # Neutrals
        st.subheader("🎨 Your Neutral Colors")
        neutrals = palette.get('neutrals', [])
        neutral_names = palette.get('neutral_names', [])
        if neutrals:
            cols = st.columns(len(neutrals))
            for i, color in enumerate(neutrals):
                with cols[i]:
                    st.color_picker(f"", color, disabled=True, key=f"neutral_{i}")
                    name = neutral_names[i] if i < len(neutral_names) else color
                    st.caption(name)
        
        # Colors to Avoid
        st.subheader("🚫 Colors to Avoid")
        avoid = palette.get('avoid', [])
        avoid_names = palette.get('avoid_names', [])
        if avoid:
            cols = st.columns(len(avoid))
            for i, color in enumerate(avoid):
                with cols[i]:
                    st.color_picker(f"", color, disabled=True, key=f"avoid_{i}")
                    name = avoid_names[i] if i < len(avoid_names) else color
                    st.caption(name)
        
        # Metals
        metals = palette.get('metals', [])
        if metals:
            st.write(f"**Best Metals:** {', '.join(metals)}")
        
        st.write("---")
        
        # Face Shape
        st.subheader("📐 Face Geometry")
        st.metric("Face Shape", face_shape)
        st.write("**Recommended Necklines:**")
        for neck in necklines:
            st.write(f"• {neck}")
        
        st.write("---")
        
        # === SHOPPING SECTION ===
        st.subheader("🛒 Shop Your Colors")
        st.write(f"Find products in your **{profile['season_name']}** palette!")
        
        # Get power colors for shopping
        power_color_names = palette.get('power_color_names', ['Olive', 'Rust', 'Navy'])
        
        col_shop1, col_shop2, col_shop3 = st.columns(3)
        
        with col_shop1:
            selected_color = st.selectbox("Select Color", power_color_names, key="shop_color")
        
        with col_shop2:
            item_types = ["Shirt", "T-Shirt", "Kurta", "Jacket", "Pants", "Jeans"]
            selected_item = st.selectbox("Item Type", item_types, key="shop_item")
        
        with col_shop3:
            budgets = ["Low", "Medium", "High"]
            selected_budget = st.selectbox("Budget", budgets, index=1, key="shop_budget")
        
        if st.button("🔍 Find Products", type="primary"):
            with st.spinner(f"Finding {selected_color} {selected_item} for you..."):
                result = get_agent_recommendations(
                    season=profile['season_name'],
                    color=selected_color,
                    item_type=selected_item.lower(),
                    budget=selected_budget.lower()
                )
                
                if "error" in result and result["error"]:
                    st.error(f"Search error: {result['error']}")
                else:
                    # Show AI recommendations
                    products = result.get('products', [])
                    if products:
                        st.success(f"✨ {len(products)} AI-curated picks for your {profile['season_name']} palette!")
                        
                        for product in products:
                            with st.container():
                                col_a, col_b = st.columns([1, 4])
                                with col_a:
                                    st.write("👔")
                                with col_b:
                                    st.markdown(f"**{product.get('title', 'Product')}**")
                                    st.caption(f"💰 {product.get('price', 'Check Store')} • 🏪 {product.get('store_name', 'Store')}")
                                    st.link_button(f"Shop on {product.get('store_name', 'Store')} →", product.get('link', '#'))
                            st.write("")
                    
                    # Always show direct store links
                    store_links = result.get('store_links', [])
                    if store_links:
                        st.write("---")
                        st.markdown("### 🏬 Shop Directly on Stores")
                        st.caption(f"Find all **{selected_color} {selected_item}** options:")
                        
                        cols = st.columns(len(store_links))
                        for i, store in enumerate(store_links):
                            with cols[i]:
                                st.markdown(f"**{store['store_name']}**")
                                st.caption(store.get('description', ''))
                                st.link_button(
                                    f"🛒 Shop Now", 
                                    store['search_url'],
                                    use_container_width=True
                                )

else:
    st.info("👆 Please upload an image to start your color analysis.")

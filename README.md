# SpectraStyle - AI Personal Stylist 

Check out the full workflow and case study on [Simform Engineering Blog](https://pypush.simform.com/).

**SpectraStyle** is an advanced AI-powered personal stylist application that uses computer vision and color science to determine your perfect seasonal color palette. It analyzes your skin tone, hair color, and eye color to classify you into one of the **12 Seasonal Color Analysis** categories and provides personalized shopping recommendations.

![SpectraStyle Analysis](https://via.placeholder.com/800x400?text=SpectraStyle+AI+Analysis)

##  Key Features

*   **12-Season Color Analysis:** Uses the "Flow System" to accurately classify users into 12 distinct seasons (e.g., Deep Autumn, Light Spring).
*   **Advanced Face Detection:** Powered by **Google MediaPipe** for precise landmark detection and segmentation of skin, hair, and eyes.
*   **Glare-Resistant Hair Analysis:** Features a proprietary algorithm with **K-Means Clustering** and "Darkness Priority" to correctly identify hair color even under studio lighting or glare.
*   **Universal Skin Tone Support:** Validated across diverse skin tones using **CIELAB color space** for accurate temperature and depth scoring.
*   **AI Shopping Agent:** Integrated **LangChain** agent with **Groq (Llama 3)** and **Tavily Search** to find real clothing items from Indian e-commerce stores (Myntra, Amazon, Ajio) that match your season.
*   **Face Shape & Style Guide:** Analyzes facial geometry to recommend the best necklines and accessory styles.

##  Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **Computer Vision:** [OpenCV](https://opencv.org/), [MediaPipe](https://developers.google.com/mediapipe)
*   **AI/LLM:** [LangChain](https://langchain.com/), [Groq API](https://groq.com/) (Llama 3.3 70B)
*   **Data Science:** [Scikit-learn](https://scikit-learn.org/) (K-Means), NumPy, Pandas
*   **Search:** [Tavily API](https://tavily.com/)

##  Installation & Setup

Follow these steps to set up the project locally.

### Prerequisites

*   Python 3.10+
*   Git

### 1. Clone the Repository

```bash
git clone https://github.com/kanishk083/SpectraStyle.git
cd SpectraStyle
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add your API keys:

```ini
# Get your keys from:
# https://console.groq.com/keys
# https://tavily.com/

GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 5. Run the Application

```bash
streamlit run app.py
```

## 📂 Project Structure

```
SpectraStyle/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── .env                    # API keys (not committed)
├── data/
│   └── palettes.json       # Color palette database (12 seasons)
└── src/
    ├── detector.py         # Face detection & segmentation (MediaPipe)
    ├── color_analysis.py   # Skin/Hair/Eye color extraction logic
    ├── universal_analyst.py# Core logic for 12-season classification
    ├── shopping_agent.py   # AI Shopping Agent (LangChain + Groq)
    ├── recommender.py      # Face shape & style recommendations
    └── utils.py            # Helper functions
```

##  Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📄 License

This project is licensed under the MIT License.

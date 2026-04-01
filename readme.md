# Floor Plan Semantic Segmentation & Annotation Pipeline

An end-to-end machine learning pipeline that ingests raw, unstructured 2D floor plan images and outputs a semantically linked, machine-readable JSON database. 

This system utilizes a two-stream architecture:
1. **Geometric Vision Layer:** A fine-tuned YOLO11 segmentation model to extract spatial polygons for walls, rooms, doors, and windows.
2. **Semantic Text Layer:** An Optical Character Recognition (OCR) engine (EasyOCR) combined with Fuzzy String Matching and spatial heuristics (`shapely`) to translate Finnish architectural abbreviations and link text annotations to their corresponding physical geometries.

## 📂 Project Structure
    floorplan/
    ├── data/                   # Auto-generated during Step 2
    ├── models/                 # Auto-generated during Step 4
    ├── scripts/
    │   ├── build_yolo_dataset.py
    │   ├── download_data.sh
    │   ├── predict.py
    │   ├── train.py
    │   ├── train_resume.py
    │   └── visualize_json.py
    ├── requirements.txt
    └── README.md

*(Note: All scripts are designed to be executed from inside the `scripts/` directory).*

---

### Step 1: Environment Setup
Clone the repository and install the required dependencies. It is highly recommended to use a virtual environment or Conda.

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt


### Step 2: Data Acquisition (Kaggle API)
The raw CubiCasa5k dataset is downloaded via the Kaggle API. 
1. Ensure you have an active Kaggle account.
2. Go to your Kaggle Account Settings and click **"Create New API Token"** to download `kaggle.json`.
3. Place `kaggle.json` in the `~/.kaggle/` directory on your machine (`mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/` and ensure permissions are set with `chmod 600 ~/.kaggle/kaggle.json`).
4. Navigate to the scripts folder and run the downloader:

    cd scripts
    bash download_data.sh

**Output:** This will create a `../data/cubicasa5k/` directory containing the raw high-quality, architectural, and colorful floor plans alongside their ground-truth SVG files.

### Step 3: Data Preprocessing & YOLO Formatting
The raw dataset contains over 80 noisy SVG categories. This script acts as a parser, extracting strictly the required polygons (walls, rooms, doors, windows, stairs) and mathematically normalizing them into the YOLO `.txt` format.

    python build_yolo_dataset.py

**Output:** Creates `../data/yolo_dataset/` containing properly split `train` and `val` folders with normalized images and coordinate text files, along with `dataset.yaml`.

### Step 4: Training the Geometric Vision Model
Train the YOLO11 Medium Segmentation model on the newly built dataset.

    python train.py

*(Optional: If training is interrupted or requires more epochs for high-density room detection, use `python train_resume.py`).*

**Output:** Creates the `../models/floorplan_seg_v2/` directory, saving training metrics (`results.csv`, `results.png`) and the final tuned weights (`best.pt`).

### Step 5: End-to-End Inference & Semantic Linking
The core engine. This script takes a raw floor plan image, extracts the geometries via YOLO, extracts text via EasyOCR, and uses `shapely` point-in-polygon heuristics and Levenshtein distance (fuzzy matching) to output a structured graph.

    python predict.py

**Output:** Generates `final_floorplan_data.json` in the current directory, containing the fully linked hierarchy of rooms, walls, and their respective dimensions/semantic labels.

### Step 6: Visualization (Sanity Check)
To physically verify the accuracy of the JSON output, run the visualizer to overlay the generated polygons and semantic text directly onto the original source image.

    python visualize_json.py

**Output:** Generates `final_system_output.jpg`, displaying color-coded bounding masks (e.g., green for rooms, blue for walls) and text overlays based exclusively on the JSON data.

### RUNNING/INFERENCE ONLY:
Given an image path, say image.png (in the floorplan directory), a weight file path, say "weights.pt" (sample weight files are available in floorplan/models/pretrained/epoch50.pt), simply run: python predict.py by making the following changes (on lines 14, 15):
IMAGE_PATH = "../data/cubicasa5k/cubicasa5k/cubicasa5k/high_quality/17/F1_scaled.png" (change this to your image file path)
YOLO_WEIGHTS = "runs/models/floorplan_seg_v3/weights/best.pt" (change this to your weight file path)
The output would then be a JSON file, as well as a visualization of the JSON as a png image.

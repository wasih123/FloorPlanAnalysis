from ultralytics import YOLO
import os

# 1. Load the pre-trained YOLO11 Medium Segmentation model
# This automatically downloads the base weights so we don't start from scratch
print("Loading YOLO11m-seg model...")
model = YOLO("yolo11m-seg.pt") 

# 2. Define the path to our dataset configuration
yaml_path = os.path.abspath("../data/yolo_dataset/dataset.yaml")

# 3. Start Training
print(f"Starting training using data configuration: {yaml_path}")
results = model.train(
    data=yaml_path,
    epochs=50,              # 50 is a good starting point to see if the loss goes down
    imgsz=800,              # Floor plans are detailed. 800px resolution helps see thin walls
    batch=16,               # Number of images processed at once (A6000 can easily handle 16+)
    device=3,               # Tells it to use your primary GPU
    project="../models",    # Where to save the results
    name="floorplan_seg_v2",# Name of this specific training run
    patience=10             # Stops early if the model stops improving for 10 epochs
)

print("Training initiated! Weights will be saved in ../models/floorplan_seg_v2/weights/")
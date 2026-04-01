from ultralytics import YOLO
import os

# 1. Load the golden weights from your successful 50-epoch run
# This means the AI starts with all its previous knowledge intact
previous_weights = os.path.abspath("runs/models/floorplan_seg_v2/weights/best.pt")
print(f"Loading base knowledge from: {previous_weights}")

model = YOLO(previous_weights)

yaml_path = os.path.abspath("../data/yolo_dataset/dataset.yaml")

# 2. Start the extended training phase
print(f"Starting extended training using data configuration: {yaml_path}")
results = model.train(
    data=yaml_path,
    epochs=100,              # Giving it 100 more rounds to find those smaller rooms
    imgsz=800,              
    batch=16,               
    device=3,                # Safely isolated on GPU 3!
    project="../models",    
    name="floorplan_seg_v3", # We create a v3 folder to keep your v2 logs safe
    patience=15              # Will stop early if it stops learning
)

print("Extended training finished! New weights are in ../models/floorplan_seg_v3/weights/")
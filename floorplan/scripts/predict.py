import os
import cv2
import json
import easyocr
import difflib
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

# ==========================================
# 1. CONFIGURATION & DICTIONARIES
# ==========================================
# Pointing to our single test image
IMAGE_PATH = "../data/cubicasa5k/cubicasa5k/cubicasa5k/high_quality/17/F1_scaled.png"
YOLO_WEIGHTS = "runs/models/floorplan_seg_v3/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.10

# The strict ground-truth dictionary (No typos allowed here!)
VALID_ROOM_CODES = {
    "MH": "bedroom", 
    "OH": "living_room", 
    "K": "kitchen", 
    "KH": "bathroom", 
    "WC": "bathroom",
    "VAR": "storage", 
    "ET": "hallway", 
    "TK": "vestibule",
    "S": "sauna",
    "VH": "closet",
    "PARV": "balcony",
    "KHH": "utility_room",
    "AT": "garage",
    "TEKN": "technical_room"
}

# YOLO Class Definitions 
CLASS_NAMES = {0: 'wall', 1: 'room', 2: 'door', 3: 'window', 4: 'staircase'}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_centroid(bbox):
    """Calculates the center point (x, y) of an EasyOCR bounding box."""
    pts = np.array(bbox)
    x_center = np.mean(pts[:, 0])
    y_center = np.mean(pts[:, 1])
    return Point(x_center, y_center)

def clean_polygon_for_json(polygon_array):
    """Converts numpy float32 arrays into standard Python floats for JSON."""
    return [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in polygon_array]

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def generate_floorplan_json(image_path):
    print(f"\nProcessing: {image_path}")
    
    final_output = {
        "metadata": {"source_image": os.path.basename(image_path)},
        "rooms": [],
        "walls": [],
        "doors": [],
        "windows": [],
        "staircases": []
    }

    # --- A. RUN VISION LAYER (YOLO11) ---
    print("Running YOLO11 Segmentation...")
    yolo_model = YOLO(YOLO_WEIGHTS)
    yolo_results = yolo_model(image_path, verbose=False)[0]
    
    room_geometries = []
    wall_geometries = []
    
    if yolo_results.masks is not None:
        for i, mask in enumerate(yolo_results.masks.xy):
            class_id = int(yolo_results.boxes.cls[i].item())
            class_name = CLASS_NAMES[class_id]
            
            if len(mask) < 3: continue # Skip invalid polygons
            
            poly_coords = clean_polygon_for_json(mask)
            shapely_poly = Polygon(mask)
            
            geo_object = {
                "id": i,
                "type": class_name,
                "polygon": poly_coords
            }
            
            if class_name == 'room':
                geo_object['label'] = "unknown" 
                final_output["rooms"].append(geo_object)
                room_geometries.append({"id": i, "poly": shapely_poly})
            elif class_name == 'wall':
                geo_object['dimension'] = None  
                final_output["walls"].append(geo_object)
                wall_geometries.append({"id": i, "poly": shapely_poly})
            elif class_name == 'door': final_output["doors"].append(geo_object)
            elif class_name == 'window': final_output["windows"].append(geo_object)
            elif class_name == 'staircase': final_output["staircases"].append(geo_object)

    # --- B. RUN TEXT LAYER (EasyOCR) ---
    print("Running EasyOCR Text Extraction...")
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    ocr_results = reader.readtext(image_path)
    
    # --- C. SEMANTIC LINKING ENGINE (With Fuzzy Logic) ---
    print("Executing Spatial & Fuzzy Heuristics...")
    for (bbox, text, prob) in ocr_results:
        if prob < CONFIDENCE_THRESHOLD: continue
        
        centroid = get_centroid(bbox)
        
        # Heuristic 1: Is it a dimension? (Contains numbers)
        if any(char.isdigit() for char in text):
            text_clean = text.strip()
            if not wall_geometries: continue
            
            min_dist = float('inf')
            closest_wall_id = None
            
            for wall in wall_geometries:
                dist = wall["poly"].distance(centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_wall_id = wall["id"]
                    
            if closest_wall_id is not None:
                for w in final_output["walls"]:
                    if w["id"] == closest_wall_id:
                        w["dimension"] = text_clean
                        break
                        
        # Heuristic 2: Is it a Room Label?
        else:
            # Strip out common OCR garbage (punctuation and spaces)
            clean_text = text.upper().strip(".:;,-'\" ")
            
            # Fuzzy match against our valid keys. 
            # cutoff=0.6 means "V.A.R" will safely match "VAR"
            possible_matches = difflib.get_close_matches(
                clean_text, 
                VALID_ROOM_CODES.keys(), 
                n=1, 
                cutoff=0.6
            )
            
            if not possible_matches: 
                continue # Ignore text that is completely unrecognizable
                
            best_match = possible_matches[0]
            matched_label = VALID_ROOM_CODES[best_match]
            
            # Spatial Check: Which room does this text belong to?
            for room in room_geometries:
                if room["poly"].contains(centroid):
                    for r in final_output["rooms"]:
                        if r["id"] == room["id"]:
                            r["label"] = matched_label
                            break

    # --- D. EXPORT ---
    output_json_path = "final_floorplan_data.json"
    with open(output_json_path, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"SUCCESS! Fully linked JSON saved to: {output_json_path}\n")

if __name__ == "__main__":
    generate_floorplan_json(IMAGE_PATH)
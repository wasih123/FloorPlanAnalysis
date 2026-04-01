import cv2
import json
import numpy as np

# 1. Load the raw image and your generated JSON
image_path = "../data/cubicasa5k/cubicasa5k/cubicasa5k/high_quality/17/F1_scaled.png"
json_path = "final_floorplan_data.json"

image = cv2.imread(image_path)
with open(json_path, 'r') as f:
    data = json.load(f)

# 2. Draw Rooms (Green) & Labels
for room in data.get("rooms", []):
    pts = np.array(room["polygon"], np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    if room["label"] != "unknown":
        # Put the semantic label near the first point of the polygon
        x, y = pts[0][0]
        cv2.putText(image, room["label"].upper(), (int(x), int(y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 3. Draw Walls (Blue) & Dimensions
for wall in data.get("walls", []):
    pts = np.array(wall["polygon"], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], color=(255, 0, 0)) # Fill walls to make them pop
    
    if wall.get("dimension"):
        x, y = pts[0][0]
        cv2.putText(image, wall["dimension"], (int(x), int(y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Save the final masterpiece
cv2.imwrite("final_system_output.jpg", image)
print("Saved final_system_output.jpg! Open it in VS Code.")
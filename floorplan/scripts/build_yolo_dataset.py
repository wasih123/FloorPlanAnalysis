import os
import cv2
import shutil
import xml.etree.ElementTree as ET

# Configuration - Pointing to the root where the .txt files and style folders live
BASE_DIR = "../data/cubicasa5k/cubicasa5k/cubicasa5k/"
YOLO_DIR = "../data/yolo_dataset/"

# The specific classes required by the assignment
CLASSES = {
    'wall': 0,
    'room': 1,
    'door': 2,
    'window': 3,
    'stair': 4
}

def setup_directories():
    # Clear out the old prototype data if it exists
    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)
        
    for split in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, 'labels', split), exist_ok=True)

def parse_and_convert(relative_path, split):
    # relative_path looks like "/high_quality_architectural/6044/"
    # We strip the leading slash so os.path.join doesn't get confused
    clean_path = relative_path.strip(" /\n")
    
    img_path = os.path.join(BASE_DIR, clean_path, 'F1_scaled.png')
    svg_path = os.path.join(BASE_DIR, clean_path, 'model.svg')
    
    if not os.path.exists(img_path) or not os.path.exists(svg_path):
        return False
        
    img = cv2.imread(img_path)
    if img is None: return False
    h, w = img.shape[:2]
    
    tree = ET.parse(svg_path)
    root = tree.getroot()
    yolo_labels = []
    
    for group in root.iter():
        tag = group.tag.split('}')[-1]
        if tag == 'g':
            group_id = group.attrib.get('id', '').lower()
            group_class = group.attrib.get('class', '').lower()
            
            class_id = -1
            for class_name, c_id in CLASSES.items():
                if class_name in group_id or class_name in group_class:
                    class_id = c_id
                    break
                    
            if class_id == -1: continue 
            
            for elem in group.iter():
                elem_tag = elem.tag.split('}')[-1]
                if elem_tag == 'polygon':
                    pts_str = elem.attrib.get('points', '')
                    if not pts_str: continue
                    
                    normalized_pts = []
                    for pt in pts_str.strip().split():
                        if ',' in pt:
                            try:
                                x, y = pt.split(',')
                                nx, ny = max(0, min(1, float(x)/w)), max(0, min(1, float(y)/h))
                                normalized_pts.extend([f"{nx:.6f}", f"{ny:.6f}"])
                            except ValueError:
                                continue
                                
                    if len(normalized_pts) >= 6: 
                        yolo_labels.append(f"{class_id} " + " ".join(normalized_pts))
                        
    if not yolo_labels: return False
    
    # Extract just the ID number to use as the filename (e.g., "6044")
    sample_id = clean_path.split('/')[-1]
    
    # Save Image
    new_img_path = os.path.join(YOLO_DIR, 'images', split, f"{sample_id}.png")
    shutil.copy(img_path, new_img_path)
    
    # Save Label TXT
    label_path = os.path.join(YOLO_DIR, 'labels', split, f"{sample_id}.txt")
    with open(label_path, 'w') as f:
        f.write("\n".join(yolo_labels))
        
    return True

def process_split(split_name, txt_filename):
    txt_path = os.path.join(BASE_DIR, txt_filename)
    if not os.path.exists(txt_path):
        print(f"Error: Could not find {txt_path}")
        return
        
    with open(txt_path, 'r') as f:
        paths = f.readlines()
        
    print(f"\nProcessing {len(paths)} {split_name} samples...")
    success_count = 0
    
    for i, path in enumerate(paths):
        if parse_and_convert(path, split_name):
            success_count += 1
            
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(paths)}...")
            
    print(f"-> Successfully converted {success_count} {split_name} samples.")

def create_yaml():
    yaml_content = f"""path: {os.path.abspath(YOLO_DIR)}
train: images/train
val: images/val

names:
  0: wall
  1: room
  2: door
  3: window
  4: staircase
"""
    with open(os.path.join(YOLO_DIR, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    print("Setting up fresh YOLO directories...")
    setup_directories()
    
    # Read exactly what the dataset creators want us to use
    process_split('train', 'train.txt')
    process_split('val', 'val.txt')
    
    create_yaml()
    print("\nDone! Full production YOLO dataset successfully built.")
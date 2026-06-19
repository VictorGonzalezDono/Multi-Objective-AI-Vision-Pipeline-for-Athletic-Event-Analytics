#!/usr/bin/env python3
"""
Multi-Objective AI Vision Pipeline for Athletic Event Analytics
Author: Víctor Manuel González Monroy
Description: Production-ready script to batch process race imagery,
             extracting bib numbers, marketing sentiments, and wearable tech penetration.
"""

import os
import glob
import argparse
import cv2
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO
import easyocr
from transformers import CLIPProcessor, CLIPModel

class SportsVisionPipeline:
    def __init__(self):
        print("[*] Initializing AI Models (YOLOv8, EasyOCR, CLIP)...")
        # Initialize Core Models
        self.yolo_model = YOLO('yolov8n.pt')
        self.ocr_reader = easyocr.Reader(['es', 'en'], gpu=torch.cuda.is_available())
        
        # Initialize Multimodal Zero-Shot Models
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Zero-Shot Target Prompts
        self.prompts_wearables = [
            "a runner wearing sports sunglasses", 
            "a runner wearing headphones", 
            "a runner wearing a smartwatch", 
            "a runner with no gadgets"
        ]
        self.prompts_emotions = [
            "a runner smiling or making a happy gesture like a heart", 
            "a runner with a neutral focused expression", 
            "a runner looking exhausted"
        ]

    def process_image(self, img_path):
        """Processes a single frame and extracts multi-objective insights."""
        img = cv2.imread(img_path)
        if img is None:
            return []
        
        h, w, _ = img.shape
        filename = os.path.basename(img_path)
        frame_results = []
        
        # Step 1: Detect runners via YOLOv8
        detections = self.yolo_model(img, verbose=False)[0]
        
        for box in detections.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Filter class 'person' (0) with high confidence thresholds
            if cls == 0 and conf > 0.45:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Spatial Heuristics: Target the runner's upper torso for optimal OCR
                height = y2 - y1
                torso_y1 = max(0, y1 + int(height * 0.15))
                torso_y2 = min(h, y1 + int(height * 0.65))
                
                cuerpo_crop = img[y1:y2, x1:x2]
                torso_crop = img[torso_y1:torso_y2, x1:x2]
                
                if cuerpo_crop.size == 0 or torso_crop.size == 0:
                    continue
                
                # Step 2: Optical Character Recognition (Bibs & Clubs)
                ocr_results = self.ocr_reader.readtext(torso_crop)
                bib_detected = "No detectado"
                club_detected = "Genérico / Independiente"
                
                for (_, text, ocr_conf) in ocr_results:
                    text_clean = text.strip().upper()
                    if text_clean.isdigit() and len(text_clean) >= 2 and bib_detected == "No detectado":
                        bib_detected = text_clean
                    elif any(word in text_clean for word in ["JABALI", "JABALÍES", "CALICO", "CALICOS", "MUEVETE", "MEXICO"]):
                        club_detected = text_clean
                
                # Step 3: Multimodal Classification via CLIP
                cuerpo_rgb = Image.fromarray(cv2.cvtColor(cuerpo_crop, cv2.COLOR_BGR2RGB))
                
                # Extract Wearables
                inputs_w = self.clip_processor(text=self.prompts_wearables, images=cuerpo_rgb, return_tensors="pt", padding=True)
                outputs_w = self.clip_model(**inputs_w)
                idx_w = torch.argmax(outputs_w.logits_per_image.softmax(dim=1)).item()
                wearable_final = self.prompts_wearables[idx_w].replace("a runner wearing ", "").replace("a runner with ", "").capitalize()
                
                # Extract Emotional Layer
                inputs_e = self.clip_processor(text=self.prompts_emotions, images=cuerpo_rgb, return_tensors="pt", padding=True)
                outputs_e = self.clip_model(**inputs_e)
                idx_e = torch.argmax(outputs_e.logits_per_image.softmax(dim=1)).item()
                emotion_final = "Alta Energía / Positiva" if idx_e == 0 else ("Concentrado" if idx_e == 1 else "Agotado")
                
                # Strict Data Quality Filter
                if bib_detected == "No detectado" and conf < 0.60:
                    continue
                
                frame_results.append({
                    "Archivo": filename,
                    "Bib_Corredor": bib_detected,
                    "Club / Jersey": club_detected,
                    "Gadgets_Detectados": wearable_final,
                    "Nivel_Emocion": emotion_final
                })
                
        return frame_results

    def run_batch(self, input_dir, output_csv):
        """Executes the pipeline over a directory of images and exports a clean CSV."""
        extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
            
        print(f"[*] Found {len(image_paths)} images in '{input_dir}' ready for batch processing.")
        
        all_results = []
        for idx, path in enumerate(image_paths):
            try:
                results = self.process_image(path)
                all_results.extend(results)
                if (idx + 1) % 10 == 0 or (idx + 1) == len(image_paths):
                    print(f"[>] Progress: {idx + 1}/{len(image_paths)} images processed.")
            except Exception as e:
                print(f"[!] Error processing {path}: {str(e)}")
                
        # Exporting Relational Data Structure
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"[✓] Execution finished. Consolidated data saved to '{output_csv}' ({len(df)} rows generated).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Production Sports Vision Pipeline")
    parser.add_argument('--input', type=str, default='data/sample_images/', help='Path to input images directory')
    parser.add_argument('--output', type=str, default='data/analytical_report.csv', help='Path to output CSV file')
    args = parser.parse_args()
    
    pipeline = SportsVisionPipeline()
    pipeline.run_batch(args.input, args.output)

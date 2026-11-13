"""
Fix CheXpert extraction to use unique filename patientXXXXX_studyY_viewZ.png
and reach full 10,000 images cleanly.
"""
import concurrent.futures
import io
import os
from pathlib import Path
import time
import zipfile
import pandas as pd
from PIL import Image

def process_image(img_bytes: bytes, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(img_bytes)) as im:
        rgb_im = im.convert("RGB")
        resized = rgb_im.resize((224, 224), Image.Resampling.BILINEAR)
        resized.save(dest_path, format="PNG")

def extract_and_resize_chexpert(target_count: int = 10000):
    print("=== Extracting and Resizing CheXpert (Unique Filenames) ===")
    t0 = time.time()
    dest_dir = Path("data/processed/chexpert/images")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    extracted = set(p.name for p in dest_dir.glob("*.png"))
    print(f"Already extracted CheXpert images: {len(extracted)}")

    csv_path = Path("data/processed/chexpert/train.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
        
    df_chex = pd.read_csv(csv_path)
    
    # Build unique mappings
    targets_map = {}
    for _, row in df_chex.iterrows():
        p = str(row["Path"])
        parts = p.split("/")
        # CheXpert-v1.0-small / train / patientXXXXX / studyY / viewZ.jpg
        if len(parts) >= 5:
            patient = parts[2]
            study = parts[3]
            view = parts[4].replace(".jpg", ".png")
            dest_filename = f"{patient}_{study}_{view}"
        else:
            dest_filename = Path(p).name.replace(".jpg", ".png")
            
        clean_path = p.replace("CheXpert-v1.0-small/", "")
        if dest_filename not in extracted:
            targets_map[clean_path] = dest_filename
            
        if len(extracted) + len(targets_map) >= target_count:
            break
            
    print(f"Need to extract {len(targets_map)} unique CheXpert images from archive.zip...")
    
    with zipfile.ZipFile("archive.zip", "r") as zf:
        infolist = zf.infolist()
        info_dict = {info.filename: info for info in infolist}
        
        valid_items = []
        for clean_path, dest_fname in targets_map.items():
            if clean_path in info_dict:
                valid_items.append((info_dict[clean_path], dest_fname))
                
        print(f"Matched {len(valid_items)} files inside archive.zip. Starting parallel extraction...")
        
        def save_item(item):
            info, dest_fname = item
            b = zf.read(info)
            process_image(b, dest_dir / dest_fname)
            return dest_fname
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for count, dest_fname in enumerate(ex.map(save_item, valid_items), 1):
                extracted.add(dest_fname)
                if count % 2000 == 0 or count == len(valid_items):
                    print(f"Progress CheXpert: {count}/{len(valid_items)} (Total on disk: {len(extracted)})")
                    
    print(f"CheXpert Extraction finished: {len(extracted)} images in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    extract_and_resize_chexpert(target_count=10000)


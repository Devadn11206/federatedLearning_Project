"""
Script to stream-extract and resize images directly from the 5 zip archives
into data/processed/{nih_cxr14,chexpert,vindr_cxr}/images/ as 224x224 RGB PNGs.
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

def extract_and_resize_nih(target_count: int = 10000):
    print("=== Extracting and Resizing NIH ChestX-ray14 ===")
    t0 = time.time()
    dest_dir = Path("data/processed/nih_cxr14/images")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Priority 1: Bbox images
    bbox_csv = Path("data/processed/nih_cxr14/BBox_List_2017.csv")
    bbox_imgs = set()
    if bbox_csv.exists():
        df_bbox = pd.read_csv(bbox_csv)
        bbox_imgs = set(df_bbox["Image Index"].unique())
        print(f"BBox list has {len(bbox_imgs)} unique images.")
        
    extracted = set(p.name for p in dest_dir.glob("*.png"))
    print(f"Already extracted NIH images: {len(extracted)}")
    
    zip_files = ["images_001.zip", "images_002.zip", "images_003.zip"]
    
    # First pass: extract all bbox images from zips
    for zname in zip_files:
        if not os.path.exists(zname):
            continue
        print(f"Scanning {zname} for BBox targets...")
        with zipfile.ZipFile(zname, "r") as zf:
            infolist = zf.infolist()
            targets = [info for info in infolist if Path(info.filename).name in bbox_imgs and Path(info.filename).name not in extracted]
            print(f"Found {len(targets)} BBox images in {zname}")
            
            def save_item(info):
                b = zf.read(info)
                fname = Path(info.filename).name
                process_image(b, dest_dir / fname)
                return fname
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                for fname in ex.map(save_item, targets):
                    extracted.add(fname)
                    
    print(f"Total after BBox extraction: {len(extracted)}")
    
    # Second pass: fill up to target_count
    for zname in zip_files:
        if len(extracted) >= target_count:
            break
        if not os.path.exists(zname):
            continue
        print(f"Filling images from {zname}...")
        with zipfile.ZipFile(zname, "r") as zf:
            infolist = zf.infolist()
            targets = []
            for info in infolist:
                if len(extracted) + len(targets) >= target_count:
                    break
                fname = Path(info.filename).name
                if fname.endswith(".png") and fname not in extracted:
                    targets.append(info)
                    
            print(f"Extracting {len(targets)} images from {zname}...")
            
            def save_item(info):
                b = zf.read(info)
                fname = Path(info.filename).name
                process_image(b, dest_dir / fname)
                return fname
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                for fname in ex.map(save_item, targets):
                    extracted.add(fname)
                    
    print(f"NIH Extraction finished: {len(extracted)} images in {time.time() - t0:.2f}s")


def extract_and_resize_chexpert(target_count: int = 10000):
    print("=== Extracting and Resizing CheXpert ===")
    t0 = time.time()
    dest_dir = Path("data/processed/chexpert/images")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    extracted = set(p.name for p in dest_dir.glob("*.png"))
    print(f"Already extracted CheXpert images: {len(extracted)}")
    if len(extracted) >= target_count:
        print("CheXpert target count already met.")
        return

    csv_path = Path("data/processed/chexpert/train.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
        
    df_chex = pd.read_csv(csv_path)
    # Filter frontal views first if possible, or take sequentially
    frontal_df = df_chex[df_chex["Frontal/Lateral"] == "Frontal"]
    if len(frontal_df) < target_count:
        sub_df = df_chex.iloc[:target_count]
    else:
        sub_df = frontal_df.iloc[:target_count]
        
    # Map required paths to target filenames
    targets_map = {}
    for _, row in sub_df.iterrows():
        p = str(row["Path"])
        clean_path = p.replace("CheXpert-v1.0-small/", "")
        dest_filename = f"{Path(p).parent.name}_{Path(p).name}".replace(".jpg", ".png")
        if dest_filename not in extracted:
            targets_map[clean_path] = dest_filename
            
    print(f"Need to extract {len(targets_map)} CheXpert images from archive.zip...")
    
    with zipfile.ZipFile("archive.zip", "r") as zf:
        infolist = zf.infolist()
        info_dict = {info.filename: info for info in infolist}
        
        valid_items = []
        for clean_path, dest_fname in targets_map.items():
            if clean_path in info_dict:
                valid_items.append((info_dict[clean_path], dest_fname))
            if len(extracted) + len(valid_items) >= target_count:
                break
                
        print(f"Matched {len(valid_items)} files inside archive.zip. Starting extraction...")
        
        def save_item(item):
            info, dest_fname = item
            b = zf.read(info)
            process_image(b, dest_dir / dest_fname)
            return dest_fname
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for dest_fname in ex.map(save_item, valid_items):
                extracted.add(dest_fname)
                if len(extracted) % 1000 == 0:
                    print(f"Progress CheXpert: {len(extracted)}/{target_count}")
                    
    print(f"CheXpert Extraction finished: {len(extracted)} images in {time.time() - t0:.2f}s")


def extract_and_resize_vindr(target_count: int = 1500):
    print("=== Extracting and Resizing VinDr-CXR ===")
    t0 = time.time()
    dest_dir = Path("data/processed/vindr_cxr/images")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    extracted = set(p.name for p in dest_dir.glob("*.png"))
    print(f"Already extracted VinDr images: {len(extracted)}")
    if len(extracted) >= target_count:
        print("VinDr target count already met.")
        return

    with zipfile.ZipFile("archive (1).zip", "r") as zf:
        infolist = zf.infolist()
        targets = []
        for info in infolist:
            if len(extracted) + len(targets) >= target_count:
                break
            fname = Path(info.filename).name
            if info.filename.endswith(".png") and fname not in extracted:
                targets.append((info, fname))
                
        print(f"Extracting {len(targets)} VinDr images...")
        
        def save_item(item):
            info, fname = item
            b = zf.read(info)
            process_image(b, dest_dir / fname)
            return fname
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for fname in ex.map(save_item, targets):
                extracted.add(fname)
                if len(extracted) % 500 == 0:
                    print(f"Progress VinDr: {len(extracted)}/{target_count}")
                    
    print(f"VinDr Extraction finished: {len(extracted)} images in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    extract_and_resize_nih(target_count=10000)
    extract_and_resize_chexpert(target_count=10000)
    extract_and_resize_vindr(target_count=1500)
    print("ALL REAL CLINICAL DATA EXTRACTED AND RESIZED SUCCESSFULLY!")


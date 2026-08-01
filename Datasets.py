import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import random
import numpy as np
import os

from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve

# Explainability & Segmentation Libraries
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import shap
import kagglehub

warnings.filterwarnings('ignore', category=UserWarning)


# ==========================================
# 0. Virtual Data Crawler & Path Parser
# ==========================================
def get_raw_dataset_paths():
    """Checks for Kaggle environment and dynamically handles unmounted datasets"""
    raw_paths = []
    datasets = [
        "andrewmvd/lung-and-colon-cancer-histopathological-images",
        "ambarish/breakhis",
        "mehradaria/leukemia",
        "prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed",
        "ashenafifasilkebede/dataset", # Oral
        "nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone",
        "masoudnickparvar/brain-tumor-mri-dataset" # Brain
    ]

    kaggle_input = '/kaggle/input'
    if os.path.exists(kaggle_input) and len(os.listdir(kaggle_input)) > 0:
        print(f"[Data Pipeline] Kaggle environment detected. Scanning mounted datasets in {kaggle_input}...")
        mounted_dirs = [os.path.join(kaggle_input, d) for d in os.listdir(kaggle_input)]
        raw_paths.extend(mounted_dirs)
        
        brain_mounted = any('brain' in d.lower() or 'mri' in d.lower() or 'masoudnickparvar' in d.lower() for d in mounted_dirs)
        
        if not brain_mounted:
            print("[Data Pipeline] Brain Cancer MRI dataset not found in /kaggle/input mounts!")
            print("[Data Pipeline] Attempting dynamic download via KaggleHub...")
            try:
                raw_paths.append(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"))
            except Exception as e:
                print(f"[Warning] Failed to download Brain MRI dataset dynamically: {e}")
    else:
        print("[Data Pipeline] Local environment detected. Downloading all via KaggleHub...")
        for ds in datasets:
            try:
                raw_paths.append(kagglehub.dataset_download(ds))
            except Exception as e:
                print(f"[Warning] Failed to download {ds}: {e}")

    if not raw_paths:
        raise RuntimeError("No datasets were downloaded and no local Kaggle input was found. Please check your internet/Kaggle credentials.")
    return raw_paths


def resolve_type_and_stage(path):
    """
    Bulletproof deterministic path mapping based on absolute file paths.
    Using an elif ladder and dataset slugs guarantees no cross-contamination.
    """
    path_lower = path.lower().replace('\\', '/')
    parts = path_lower.split('/')
    if len(parts) < 2: return None, None
    parent = parts[-2]

    # 1. Breast (BreaKHis_v1)
    if 'breakhis' in path_lower or 'breast' in path_lower or 'ambarish' in path_lower:
        if 'benign' in path_lower: return 'Breast', 'Benign'
        if 'malignant' in path_lower: return 'Breast', 'Malignant'

    # 2. Kidney (CT-Kidney) 
    elif 'kidney' in path_lower or 'ct-kidney' in path_lower or 'nazmul' in path_lower:
        if 'tumor' in parent: return 'Kidney', 'Tumor'
        if 'cyst' in parent: return 'Kidney', 'Cyst'
        if 'stone' in parent: return 'Kidney', 'Stone'
        if 'normal' in parent: return 'Kidney', 'Normal'

    # 3. Brain (Brain Tumor MRI)
    elif 'brain' in path_lower or 'mri' in path_lower or 'masoudnickparvar' in path_lower or 'glioma' in path_lower or 'meningioma' in path_lower:
        if 'glioma' in path_lower: return 'Brain', 'Glioma'
        if 'meningioma' in path_lower: return 'Brain', 'Meningioma'
        if 'pituitary' in path_lower: return 'Brain', 'Pituitary_Tumor'
        if 'notumor' in path_lower or 'no_tumor' in path_lower: return 'Brain', 'Normal'

    # 4. Lung and Colon (LC25000)
    elif 'lung' in path_lower or 'colon' in path_lower or 'lc25000' in path_lower or 'andrewmvd' in path_lower:
        if 'colon_aca' in path_lower: return 'Colon', 'Adenocarcinoma'
        if 'colon_n' in path_lower: return 'Colon', 'Benign'
        if 'lung_aca' in path_lower: return 'Lung', 'Adenocarcinoma'
        if 'lung_scc' in path_lower: return 'Lung', 'Squamous_Cell'
        if 'lung_n' in path_lower: return 'Lung', 'Benign'

    # 5. Leukemia (ALL)
    elif 'leukemia' in path_lower or 'c-nmc' in path_lower or 'lymphoblastic' in path_lower or 'mehradaria' in path_lower:
        if 'benign' in path_lower or 'hem' in path_lower: return 'Leukemia', 'Benign'
        if 'early' in path_lower: return 'Leukemia', 'Malignant_Early'
        if 'pre' in path_lower and 'early' not in path_lower: return 'Leukemia', 'Malignant_Pre'
        if 'pro' in path_lower: return 'Leukemia', 'Malignant_Pro'
        if 'all' in path_lower: return 'Leukemia', 'Malignant_Lymphoblasts'

    # 6. Cervical (Sipakmed - nested .bmp folders)
    elif 'sipakmed' in path_lower or 'cervical' in path_lower or 'prahladmehandiratta' in path_lower:
        if 'dyskeratotic' in path_lower: return 'Cervical', 'Abnormal_Dyskeratotic'
        if 'koilocytotic' in path_lower: return 'Cervical', 'Abnormal_Koilocytotic'
        if 'metaplastic' in path_lower: return 'Cervical', 'Abnormal_Metaplastic'
        if 'parabasal' in path_lower: return 'Cervical', 'Abnormal_Parabasal'
        if 'superficial' in path_lower or 'normal' in path_lower or 'intermediate' in path_lower: return 'Cervical', 'Normal_Superficial'

    # 7. Oral 
    elif 'oscc' in path_lower or 'oral' in path_lower or 'histopathological' in path_lower or 'ashenafi' in path_lower:
        if 'oscc' in path_lower: return 'Oral', 'Malignant_OSCC'
        if 'normal' in path_lower: return 'Oral', 'Benign_Normal'

    return None, None


def extract_patient_id(path, cancer_type):
    """
    Best-effort patient/slide grouping key so that the same physical patient
    cannot appear in both train and val/test.
    """
    p = path.replace('\\', '/')
    fname = os.path.basename(p)
    stem = os.path.splitext(fname)[0]

    if cancer_type == 'Breast':
        segs = stem.split('-')
        if len(segs) >= 3:
            return f"Breast::{segs[1]}-{segs[2]}"

    if cancer_type == 'Leukemia':
        parent = os.path.basename(os.path.dirname(p))
        return f"Leukemia::{parent}::{stem.split('_')[0]}"

    if cancer_type in ('Lung', 'Colon'):
        base = ''.join(ch for ch in stem if not ch.isdigit())
        return f"{cancer_type}::{base}::{stem[:12]}"

    if cancer_type == 'Oral':
        return f"Oral::{stem.split('_')[0]}"

    if cancer_type == 'Cervical':
        return f"Cervical::{stem.split('_')[0]}"

    if cancer_type == 'Kidney':
        return f"Kidney::{stem.split('_')[0]}"

    if cancer_type == 'Brain':
        return f"Brain::{stem.split('_')[0]}"

    return f"{cancer_type}::{stem}"

# ==========================================
# 3. Virtual Hierarchical Dataset Class
# ==========================================
class HierarchicalCancerDataset(Dataset):
    def __init__(self, data_dirs, config, is_train=True):
        self.config = config
        self.is_train = is_train
        self.samples = []
        
        types_set = set()
        stages_set = set()
        
        print("[Dataset] Building virtual memory map of filesystem...")
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        
        for data_dir in data_dirs:
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if not file.lower().endswith(valid_exts):
                        continue
                        
                    path = os.path.join(root, file)
                    t, s = resolve_type_and_stage(path)
                    
                    if t and s:
                        types_set.add(t)
                        full_stage_name = f"{t}_{s}"
                        stages_set.add(full_stage_name)
                        
                        self.samples.append({
                            'path': path,
                            'type_name': t,
                            'stage_name': full_stage_name
                        })

        if len(self.samples) == 0:
            raise RuntimeError("No valid images successfully mapped. Verify directory paths.")

        self.types = sorted(list(types_set))
        self.stages = sorted(list(stages_set))
        
        self.type_to_idx = {t: i for i, t in enumerate(self.types)}
        self.stage_to_idx = {s: i for i, s in enumerate(self.stages)}
        
        config.num_types = len(self.types)
        config.num_stages = len(self.stages)
        
        print(f"[Dataset] Success! Mapped {len(self.samples)} real images natively.")
        print(f" -> Level 1 Types ({config.num_types}): {self.types}")
        print(f" -> Level 2 Stages ({config.num_stages}): {self.stages}")
        
        # Training Data Augmentations for Curve Alignment
        if self.is_train:
            self.base_transform = transforms.Compose([
                transforms.Resize((config.img_size, config.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor()
            ])
        else:
            self.base_transform = transforms.Compose([
                transforms.Resize((config.img_size, config.img_size)),
                transforms.ToTensor()
            ])
        
    def stain_normalization(self, img_tensor):
        target_means = torch.tensor([0.7, 0.5, 0.6]).view(3, 1, 1)
        target_stds = torch.tensor([0.1, 0.1, 0.15]).view(3, 1, 1)
        img_mean = img_tensor.mean(dim=(1,2), keepdim=True)
        img_std = img_tensor.std(dim=(1,2), keepdim=True) + 1e-6
        normalized = (img_tensor - img_mean) / img_std
        return torch.clamp(normalized * target_stds + target_means, 0, 1)

    def morphological_edge_enhancement(self, img_tensor):
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        img_padded = F.pad(img_tensor.unsqueeze(0), (1,1,1,1), mode='reflect')
        edges = F.conv2d(img_padded, laplacian, groups=3).squeeze(0)
        return torch.clamp(img_tensor + 0.3 * edges, 0, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_img = Image.open(item['path']).convert('RGB')
        
        img_tensor = self.base_transform(raw_img)
        img_tensor = self.stain_normalization(img_tensor)
        img_tensor = self.morphological_edge_enhancement(img_tensor)
        
        type_idx = self.type_to_idx[item['type_name']]
        stage_idx = self.stage_to_idx[item['stage_name']]
        
        return img_tensor, torch.tensor(type_idx, dtype=torch.long), torch.tensor(stage_idx, dtype=torch.long)
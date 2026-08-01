
import torch
import os


# ==========================================
# 1. Configuration Class
# ==========================================
class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Directories
        base_working_dir = ''
        self.ckpt_dir = os.path.join(base_working_dir, 'models')
        self.output_dir = os.path.join(base_working_dir, 'outputs')
        self.checkpoint_path = os.path.join(self.ckpt_dir, 'CanceRX_ckpt.pth')

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Hyperparameters
        self.img_size = 224
        self.in_chans = 3
        self.num_types = None
        self.num_stages = None
        self.dim = 512
        self.depth = 4
        self.num_heads = 8
        self.num_experts = 4
        self.patch_size = 16

        self.epochs = 150
        self.batch_size = 64
        self.lr = 3e-4
        self.weight_decay = 1e-4

        # Split control (70% train, 15% validation, 15% test)
        self.train_fraction = 0.70
        self.val_fraction = 0.15
        self.test_fraction = 0.15
        self.split_seed = 42
        self.patient_aware_split = True

        # Complexity / benchmarking
        self.complexity_input_shapes = [(1, 3, 224, 224)]
        self.bench_batch_sizes = [1, 8, 32]
        self.bench_warmup_iters = 20
        self.bench_measure_iters = 100

        # XAI budgets
        self.lime_num_samples = 1000
        self.shap_nsamples = 200
        self.shap_segments = 32
        self.ig_steps = 64
        self.scorecam_topk = 32
        self.occlusion_patch = 32
        self.occlusion_stride = 8
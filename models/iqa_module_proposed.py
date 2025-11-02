import time
import torch
import torchvision
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

model_map = {
    # ViT-Base / -Large / -Huge (Default input size: 224x224 or 384x384 depending on the variant)
    "vitbase16":    (torchvision.models.vit_b_16,    torchvision.models.ViT_B_16_Weights.DEFAULT,                  768),  # 224x224
    "vitbase16_2":  (torchvision.models.vit_b_16,    torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,  768),  # 384x384
    "vitbase32":    (torchvision.models.vit_b_32,    torchvision.models.ViT_B_32_Weights.DEFAULT,                  768),  # 224x224
    "vitlarge16":   (torchvision.models.vit_l_16,    torchvision.models.ViT_L_16_Weights.DEFAULT,                 1024), # 224x224
    "vitlarge16_2": (torchvision.models.vit_l_16,    torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1, 1024), # 384x384
    "vitlarge32":   (torchvision.models.vit_l_32,    torchvision.models.ViT_L_32_Weights.DEFAULT,                 1024), # 224x224
    "vithuge14":    (torchvision.models.vit_h_14,    torchvision.models.ViT_H_14_Weights.DEFAULT,                 1280), # 518x518

    # Swin-T / -S / -B (Default input size: 224x224)
    "swin_t":       (torchvision.models.swin_t,      torchvision.models.Swin_T_Weights.DEFAULT,                   768),  # 224x224
    "swin_s":       (torchvision.models.swin_s,      torchvision.models.Swin_S_Weights.DEFAULT,                   768),  # 224x224
    "swin_b":       (torchvision.models.swin_b,      torchvision.models.Swin_B_Weights.DEFAULT,                  1024),  # 224x224

    # Swin-V2 T / S / B (Default input size: 256x256)
    "swin_v2_t":    (torchvision.models.swin_v2_t,   torchvision.models.Swin_V2_T_Weights.DEFAULT,                768),  # 256x256
    "swin_v2_s":    (torchvision.models.swin_v2_s,   torchvision.models.Swin_V2_S_Weights.DEFAULT,                768),  # 256x256
    "swin_v2_b":    (torchvision.models.swin_v2_b,   torchvision.models.Swin_V2_B_Weights.DEFAULT,               1024),  # 256x256

    # EfficientNet (Progressively increasing sizes from B0 to B4)
    "efficientnet_b0": (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.DEFAULT, 1280),  # 224x224
    "efficientnet_b1": (torchvision.models.efficientnet_b1, torchvision.models.EfficientNet_B1_Weights.DEFAULT, 1280),  # 240x240
    "efficientnet_b2": (torchvision.models.efficientnet_b2, torchvision.models.EfficientNet_B2_Weights.DEFAULT, 1408),  # 260x260
    "efficientnet_b3": (torchvision.models.efficientnet_b3, torchvision.models.EfficientNet_B3_Weights.DEFAULT, 1536),  # 300x300
    "efficientnet_b4": (torchvision.models.efficientnet_b4, torchvision.models.EfficientNet_B4_Weights.DEFAULT, 1792),  # 380x380

    # ResNet v1 (Default input size: 224x224)
    "resnet18":     (torchvision.models.resnet18,    torchvision.models.ResNet18_Weights.IMAGENET1K_V1,           512),  # 224x224
    "resnet34":     (torchvision.models.resnet34,    torchvision.models.ResNet34_Weights.IMAGENET1K_V1,           512),  # 224x224
    "resnet50":     (torchvision.models.resnet50,    torchvision.models.ResNet50_Weights.IMAGENET1K_V1,          2048),  # 224x224
    "resnet101":    (torchvision.models.resnet101,   torchvision.models.ResNet101_Weights.IMAGENET1K_V1,         2048),  # 224x224
    "resnet152":    (torchvision.models.resnet152,   torchvision.models.ResNet152_Weights.IMAGENET1K_V1,         2048),  # 224x224

    # ResNet v2 (Default input size: 224x224)
    "resnet50_v2":  (torchvision.models.resnet50,    torchvision.models.ResNet50_Weights.IMAGENET1K_V2,          2048),  # 224x224
    "resnet101_v2": (torchvision.models.resnet101,   torchvision.models.ResNet101_Weights.IMAGENET1K_V2,         2048),  # 224x224
    "resnet152_v2": (torchvision.models.resnet152,   torchvision.models.ResNet152_Weights.IMAGENET1K_V2,         2048),  # 224x224

    # MobileNet (Default input size: 224x224)
    "mobilenet_v2":       (torchvision.models.mobilenet_v2, torchvision.models.MobileNet_V2_Weights.DEFAULT,           1280),  # 224x224
    "mobilenet_v3_small": (torchvision.models.mobilenet_v3_small, torchvision.models.MobileNet_V3_Small_Weights.DEFAULT, 1024),  # 224x224
    "mobilenet_v3_large": (torchvision.models.mobilenet_v3_large, torchvision.models.MobileNet_V3_Large_Weights.DEFAULT, 1280),  # 224x224

    # ConvNeXt (Default input size: 224x224)
    "convnext_t":   (torchvision.models.convnext_tiny,  torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT,         768),  # 224x224
    "convnext_s":   (torchvision.models.convnext_small, torchvision.models.ConvNeXt_Small_Weights.DEFAULT,       768),  # 224x224
    "convnext_b":   (torchvision.models.convnext_base,  torchvision.models.ConvNeXt_Base_Weights.DEFAULT,       1024),  # 224x224
    "convnext_l":   (torchvision.models.convnext_large, torchvision.models.ConvNeXt_Large_Weights.DEFAULT,      1536),  # 224x224

    # RegNet (Default input size: 224x224)
    "regnet_y_8gf": (torchvision.models.regnet_y_8gf, torchvision.models.RegNet_Y_8GF_Weights.DEFAULT,           2016),  # 224x224
}


class FuseBackbones(torch.nn.Module):

    """
    This class fuses multiple pretrained deep learning backbones 
    (CNNs or Transformers) into a single feature extractor.

    The module freezes the parameters of each backbone, extracts their features,
    and concatenates them into a single vector. An optional dimensionality 
    reduction step (via AdaptiveAvgPool1d) can be applied to obtain a fixed-size vector.
    """

    def __init__(self, model_list, vector_size=None):

        """
        Args:
            model_list (list[str]): List of model names (keys in `model_map`) 
                                    to be fused together.
            vector_size (int, optional): If set, reduces the concatenated feature 
                                         vector to this size using adaptive pooling.
        """

        super().__init__()
        
        self.backbones = torch.nn.ModuleList()
        self.output_dims = []
        self.istransformer = []

        # Load each backbone
        for model_name in model_list:
            if model_name in model_map.keys():
                backbone, out_dim, is_t = self.get_backbone(model_name)
                self.backbones.append(backbone)
                self.output_dims.append(out_dim)
                self.istransformer.append(is_t)
            else:
                raise ValueError(f"Model '{model_name}' not found in model_map.")
        
        # Total output dimension after concatenating all backbones
        self.total_output_dim = sum(self.output_dims)

        # Freeze all backbone parameters
        for param in self.backbones.parameters():
            param.requires_grad = False

        self.vector_size = vector_size
        
        # Dimensionality reduction if requested
        if self.vector_size is not None and self.vector_size < self.total_output_dim:
            # Instead of a Linear layer, use adaptive average pooling
            self.reduce_dim = torch.nn.AdaptiveAvgPool1d(self.vector_size)
        else:
            self.reduce_dim = None
        
        self.bn1 = torch.nn.BatchNorm1d(self.total_output_dim)
    
    def get_backbone(self, model_name):

        """
        Given a model name, returns its feature extraction backbone, output dimension, 
        and whether it is a transformer.

        Args:
            model_name (str): Key from `model_map`.

        Returns:
            backbone (nn.Module): Feature extraction layers.
            out_dim (int): Dimensionality of extracted features.
            is_transformer (bool): True if model is transformer-based.
        """

        model_fn, weights_enum, out_dim = model_map[model_name]
        model = model_fn(weights=weights_enum)

        if 'resnet' in model_name:
            # Remove the final fully connected layer (`fc`)
            modules = list(model.children())[:-1]  # Remove avgpool and fc
            backbone = torch.nn.Sequential(*modules)
            return backbone, out_dim, False

        elif 'efficientnet' in model_name or 'mobilenet' in model_name:
            # EfficientNet and MobileNet use 'features'
            return model.features, out_dim, False

        elif 'convnext' in model_name:
            # ConvNeXt uses a 'features' block            
            return model.features, out_dim, False

        elif 'vit' in model_name or 'swin' in model_name:
            # ViT and Swin require forwarding up to the transformer encoder
            return model.features, out_dim, True
        
        elif 'regnet' in model_name:
            # RegNet models use a similar structure with `trunk_output` and `avgpool`
            backbone = torch.nn.Sequential(
                model.trunk_output,
                model.avgpool  # Global average pooling
            )
            return backbone, out_dim, False

        else:
            raise NotImplementedError(f"Backbone extraction not implemented for: {model_name}")
        
    def forward(self, x):

        """
        Forward pass through all backbones, feature fusion, normalization,
        and optional dimensionality reduction.

        Args:
            x (Tensor): Input batch of images of shape (B, C, H, W).

        Returns:
            fused (Tensor): Final feature vector of shape (B, D) or (B, vector_size).
        """

        features = []

        # Extract features from each backbone
        for backbone, out_dim, is_t in zip(self.backbones, self.output_dims, self.istransformer):
            out = self._extract_features(backbone, x)
            
            # Ensure shape is (B, D, 1, 1)
            if out.ndim == 2:  # e.g., ViT outputs (B, D)
                out = out.unsqueeze(-1).unsqueeze(-1)
            elif out.ndim == 3:  # (B, D, T) → treat T as spatial? handle case by case
                out = out.unsqueeze(-1)
            elif out.ndim == 4:  # (B, D, H, W)
                if is_t:
                    out = out.permute(0, 3, 1, 2)
                out = torch.nn.AdaptiveAvgPool2d(1)(out)
            elif out.ndim == 5:
                out = torch.nn.AdaptiveAvgPool3d(1)(out)
            
            features.append(out)

        # Concatenate all backbones on channel dim (dim=1)
        fused = torch.cat(features, dim=1)  # shape: (B, total_output_dim, 1, 1)
        fused = fused.view(fused.size(0), -1)
        fused = self.bn1(fused)

        if self.reduce_dim:
            # AdaptiveAvgPool1d expects input shape (B, C, L), so add a dummy dimension:
            fused = fused.unsqueeze(1)  # (B, 1, total_output_dim)
            fused = self.reduce_dim(fused)  # (B, 1, vector_size)
            fused = fused.squeeze(1)  # (B, vector_size)

        return fused
    
    def _extract_features(self, backbone, x):

        """
        Helper to forward input through a backbone.
        Handles models with custom `forward_features` (e.g., ViT, Swin).

        Args:
            backbone (nn.Module): Backbone model.
            x (Tensor): Input images.

        Returns:
            Tensor: Extracted feature maps.
        """

        # Special handling for ViT/Swin if needed
        if hasattr(backbone, 'forward_features'):
            return backbone.forward_features(x)
        
        return backbone(x)


class IQAModel:

    """
    Image Quality Assessment (IQA) model that:
    1. Extracts deep features from images using fused backbones.
    2. Reduces dimensionality via PCA.
    3. Predicts MOS scores using an ensemble of LightGBM models with final calibration.
    4. Profiles runtime across pipeline stages.
    """

    def __init__(self, 
                 model_list,                   
                 batch_size, 
                 num_workers,
                 lgb_models, 
                 calibrators,
                 pca_model,
                 dl_device,
                 ):
        
        """
        Initializes the IQA model.

        Args:
            model_list (list): List of backbone model names.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of DataLoader workers.
            lgb_models (list): Trained LightGBM models forming the ensemble.
            calibrators (list or None): Calibration models applied to each LightGBM model.
            pca_model (sklearn PCA): Pre-fitted PCA model for dimensionality reduction.
            dl_device (torch.device): Device for deep learning (CPU or GPU).
        """
        
        self.device = dl_device
        self.lgb_models = lgb_models
        self.calibrators = calibrators
        self.pca_model = pca_model
        self.n_pca_comps = 128

        # DataLoader params
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Backbone fusion model
        self.backbone = FuseBackbones(model_list=model_list, vector_size=None)
        self.backbone.to(dl_device).eval().float()

        # Profiling storage
        self.timing = {
            "preprocessing": 0.0,
            "feature_extraction": 0.0,
            "pca": 0.0,
            "ensemble": 0.0,
            "other": 0.0,
            "total": 0.0
        }

    def apply_pca(self, df):    

        """
        Applies PCA to the feature vectors generated by the backbones.

        Args:
            df: DataFrame containing raw features + metadata.

        Returns:
            Transformed DataFrame with PCA components + metadata.
        """

        X_pca = self.pca_model.transform(df.drop(columns=['image_name', 'fold']))
        pca_columns = [f'PCA_{i}' for i in range(self.n_pca_comps)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
        df_pca['fold'] = df['fold']
        df_pca['image_name'] = df['image_name']

        return df_pca
    

    def predict(self, dataset, imgs_to_process):
        
        """
        Predicts the MOS for a given dataset of images.

        Args:
            dataset: Dataset providing images and metadata.
            imgs_to_process: Max number of images to process.

        Returns:
            DataFrame with PCA features, metadata, and MOS predictions.
        """

        self.num_images = min(imgs_to_process, len(dataset))

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        features, folds, names = [], [], []

        start_time = time.time()
        cont = 0
        for img, _, fold, name, prep_time in tqdm(dataloader, desc="Extracting features"):
            
            # Preprocessing already happens inside dataset+transform
            self.timing["preprocessing"] += prep_time

            # Deep feature extraction
            t0 = time.time()
            with torch.inference_mode():
                feats = self.backbone(img.to(self.device))
            # Synchronize CUDA to ensure accurate timing
            if self.device == 'cuda':
                torch.cuda.synchronize()
            self.timing["feature_extraction"] += (time.time() - t0)

            features.append(feats.cpu())
            folds.append(fold.cpu())
            names.extend(name)

            cont += 1
            if cont > imgs_to_process:
                break

        # Apply PCA dimensionality reduction       
        features_np = torch.cat(features).numpy()
        folds_np = torch.cat(folds).numpy()
        names_np = np.array(names).reshape(-1, 1)
        num_features = features_np.shape[1]
        feature_columns = [f"f_{i}" for i in range(num_features)]
        columns = ['image_name'] + feature_columns + ['fold']

        t1 = time.time()
        df = self.apply_pca(
            df=pd.DataFrame(
                data=np.hstack([names_np, features_np, folds_np.reshape(-1, 1)]),
                columns=columns
                )
            )
        self.timing["pca"] += (time.time() - t1)
        
        # Apply LightGBM ensemble with calibration.        
        df_pred = df.drop(columns=['image_name', 'fold'])
        t2 = time.time()
        preds_list = []
        for fold, model in enumerate(self.lgb_models):
            preds = model.predict(df_pred)
            if self.calibrators is not None:
                preds = self.calibrators[fold].predict(preds.reshape(-1, 1))
            preds_list.append(preds)
        
        # Soft voting: average across ensemble models
        preds_per_model = np.column_stack(preds_list)
        mos_preds = preds_per_model.mean(axis=1)
        self.timing["ensemble"] += (time.time() - t2)

        # Total time as sum of stages
        total_time = time.time() - start_time
        stage_time = (
            self.timing["preprocessing"] +
            self.timing["feature_extraction"] +
            self.timing["pca"] +
            self.timing["ensemble"]
        )
        self.timing["total"] = total_time

        # "other": e.g., dataLoader overhead, batching, CPU-GPU transfers, Python loops)
        self.timing["other"] = max(0.0, total_time - stage_time)

        # Attach MOS predictions to DataFrame
        df['mos'] = mos_preds

        return df

    def profile(self):

        """
        Returns profiling results as a DataFrame with stage, time, and percentage.

        Returns:
            Profiling summary of the pipeline in Dataframe format.
        """

        total_time = self.timing["total"]

        stages = {
            "preprocessing": self.timing["preprocessing"],
            "feature_extraction": self.timing["feature_extraction"],
            "pca": self.timing["pca"],
            "ensemble": self.timing["ensemble"],
            "other": self.timing["other"],
            "total": self.timing["total"],
            "per_image": self.timing["total"] / self.num_images
        }

        rows = []
        for stage, t in stages.items():
            
            if isinstance(t, torch.Tensor):
                t = t.item()
            #percentage = (t / self.timing["total"] * 100) if stage not in ["total", "per_image"] else (100.0 if stage == "total" else 100.0 / self.num_images)
            if stage == "total":
                percentage = 100.0
            elif stage == "per_image":
                percentage = 100.0 / self.num_images
            else:
                percentage = (t / self.timing["total"] * 100)
            rows.append([stage, t, percentage])

        df = pd.DataFrame(rows, columns=["Stage", "Time (s)", "Percentage (%)"])

        return df

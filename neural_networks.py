"""Neural network building blocks shared by PGCM and competitor models.

Key components:
- **Embedders** — ``ImageEmbedder``, ``BigImageEmbedder``, ``FeatureEmbedder``
  map raw images or feature vectors to fixed-size prototype-space embeddings.
- **Decoders** — ``PrototypeDecoder``, ``PrototypeDecoderMNIST``,
  ``FeatureDecoder`` reconstruct images or features from prototype embeddings.
- **Segmenters** — ``ResNetUNetSegmenter``, ``ResNetUNetSegmenterMNIST`` learn
  object masks from raw images (used when ``use_pretrained_segmenter=False``).
- **Concept & Task heads** — ``ProtoToConcepts``, ``TaskPredictor`` map
  prototype embeddings to concept probabilities and task predictions.
"""

import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

class SegmentationModel(torch.nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SlotAttentionEncoder(torch.nn.Module):
    def __init__(self, num_slots, slot_size):
        pass
    

class BigImageEmbedder(torch.nn.Module):
    def __init__(self, proto_size, single_image=False):
        super().__init__()
        self.embedder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3), # 512 -> 256
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 256 -> 128
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128 -> 64
            torch.nn.ReLU(inplace=True),

            # torch.nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3), # 128 -> 64
            # torch.nn.ReLU(inplace=True)
        )
        # Global average pooling to produce one embedding vector per image
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # outputs (B, 128, 1, 1)

        # Linear projection to the final proto_size (embedding dimension)
        self.fc = torch.nn.Linear(128, proto_size)

        self.single_image = single_image

    def forward(self, x):
        # x: (batch, 2, 3, 256, 256)
        if self.single_image:
            x = x.unsqueeze(1)  # (batch, 1, 3, 256, 256)
        batch_size, num_images, c, h, w = x.shape
        x = x.view(batch_size * num_images, c, h, w)
        emb = self.embedder(x)
        emb = self.pool(emb).squeeze(-1).squeeze(-1)  # (B, 128)
        emb = self.fc(emb)  # (B, proto_size)
        emb = emb.view(batch_size, num_images, -1)
        return emb if not self.single_image else emb.squeeze()


class ImageEmbedder(torch.nn.Module):
    def __init__(self, embedding_size, proto_size, single_image=False):
        super(ImageEmbedder, self).__init__()
        self.map_image_to_object_emb = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, 5),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(704, embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_size, proto_size),
        )
        self.proto_size = proto_size
        self.embedding_size = embedding_size

        self.single_image = single_image

    def forward(self, x):
        # x: (batch, 2, 3, 56, 28)
        if self.single_image:
            x = x.unsqueeze(1)  # (batch, 1, 3, 56, 28)
        batch_size, num_images, c, h, w = x.shape
        x = x.view(batch_size * num_images, c, h, w)
        emb = self.map_image_to_object_emb(x)
        emb = emb.view(batch_size, num_images, self.proto_size)
        return emb if not self.single_image else emb.squeeze()


class FeatureEmbedder(torch.nn.Module):
    def __init__(self, input_dim=None, proto_size=64, hidden_size=None):
        super().__init__()
        hidden_size = hidden_size or max(128, proto_size * 2)
        first_layer = torch.nn.LazyLinear(hidden_size) if input_dim is None else torch.nn.Linear(input_dim, hidden_size)
        self.network = torch.nn.Sequential(
            first_layer,
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, proto_size),
        )

    def forward(self, x):
        if x.dim() == 2:
            return self.network(x)
        if x.dim() == 3:
            batch_size, num_items, feature_dim = x.shape
            x = x.reshape(batch_size * num_items, feature_dim)
            x = self.network(x)
            return x.view(batch_size, num_items, -1)
        x = x.flatten(start_dim=1)
        return self.network(x)


class FeatureDecoder(torch.nn.Module):
    def __init__(self, proto_size=64, output_dim=None, hidden_size=None):
        super().__init__()
        if output_dim is None:
            raise ValueError("FeatureDecoder requires an explicit output_dim.")
        hidden_size = hidden_size or max(128, proto_size * 2)
        self.output_dim = output_dim
        self.network = torch.nn.Sequential(
            torch.nn.Linear(proto_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            return self.network(x)
        if x.dim() == 3:
            batch_size, num_items, proto_dim = x.shape
            x = x.reshape(batch_size * num_items, proto_dim)
            x = self.network(x)
            return x.view(batch_size, num_items, self.output_dim)
        x = x.flatten(start_dim=1)
        return self.network(x)
    

class ProtoToConcepts(torch.nn.Module):
    def __init__(self, proto_size, nb_concepts, hid=64):
        super(ProtoToConcepts, self).__init__()
        self.map_proto_to_concepts = torch.nn.Sequential(
            torch.nn.Linear(proto_size, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, nb_concepts),
        )

    def forward(self, x):
        return self.map_proto_to_concepts(x)

    
class TaskPredictor(torch.nn.Module):
    def __init__(self, nb_possibleobjects, nb_concepts, nb_tasks, hidden=64, residual=0):
        super(TaskPredictor, self).__init__()
        self.task_predictor = torch.nn.Sequential(
            torch.nn.Linear(nb_possibleobjects * nb_concepts + residual, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, nb_tasks),
        )

    def forward(self, x):
        return self.task_predictor(x)
    

class SimpleDecoder(torch.nn.Module):
    def __init__(self, resolution, hid_dim):
        super(SimpleDecoder, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv5 = torch.nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = (8, 8)
        self.resolution = resolution

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        return x
    


import torchvision.models as models


def build_segmenter(dataset_name, segmentation_method, n_class):
    if segmentation_method != 'mask':
        raise NotImplementedError(f"Segmenter builder only supports segmentation_method='mask', got {segmentation_method}.")

    if dataset_name in ('celebamask', 'clevrhans', 'cub'):
        return ResNetUNetSegmenter(n_class=n_class)

    return ResNetUNetSegmenterMNIST(n_class=n_class)

class ResNetUNetSegmenter(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()
        
        # 1. Encoder (Pre-trained ResNet18 or ResNet34 is usually sufficient)
        # We grab the layers to access intermediate features for skip connections
        base_model = models.resnet18(pretrained=True)
        self.base_layers = list(base_model.children())
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        
        # 2. Decoder (Upsampling + Concat + Convolution)
        # We reduce channels after concatenation to keep parameters low
        self.up4 = self._up_block(512, 256)
        self.up3 = self._up_block(256 + 256, 128) # +256 from skip connection
        self.up2 = self._up_block(128 + 128, 64)  # +128 from skip connection
        self.up1 = self._up_block(64 + 64, 64)    # +64 from skip connection
        
        # Final block to restore full resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_class, kernel_size=1) # Output: 1 Channel logits
        )

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Decoder with Skip Connections
        x = self.up4(x4)
        x = torch.cat([x, x3], dim=1) # Skip connection from layer 3
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1) # Skip connection from layer 2

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1) # Skip connection from layer 1
        
        x = self.up1(x)
        
        # Final output
        logits = self.final_up(x) # Shape: (B, 1, H, W)
        
        # Note: We output logits here. Sigmoid is applied in the loss function or main model.
        return logits
    

class ResNetUNetSegmenterMNIST(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()
        
        self.final_up = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_class, kernel_size=1) 
        )

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- Encoder ---
        # Input: (B, 3, 56, 28)
        x0 = self.layer0(x) # ~ (14, 7)
        x1 = self.layer1(x0) # ~ (14, 7)
        x2 = self.layer2(x1) # ~ (7, 4)
        x3 = self.layer3(x2) # ~ (4, 2) - Bottleneck

        # --- Decoder ---
        # 1. Upsample x3 to match x2 size
        x = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1) 

        # 2. Upsample to match x1 size
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1) 

        # 3. Upsample to match original input size (56, 28)
        # We skip 'up1' concatenation logic for simplicity or define another block if needed,
        # but usually we just upsample the final result to the target.
        x = F.interpolate(x, size=(56, 28), mode='bilinear', align_corners=True)
        x = self.up1(x)
        
        # Final output
        logits = self.final_up(x) # Shape: (B, n_class, 56, 28)
        
        return logits


class PrototypeDecoderMNIST(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()

        self.fc = nn.Linear(embedding_dim, 256 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0) * z.size(1), 256, 7, 7)
        x = self.decoder(x)
        x = x.view(z.size(0), z.size(1), 3, 56, 28)
        return x


def affine_grid_from_params(theta, out_h, out_w, device):
    # The renderer uses bounded translation and scale so prototype patches stay
    # inside the visible image region instead of drifting or collapsing.
    B = theta.size(0)
    if theta.size(-1) == 3:
        tx = torch.tanh(theta[:, 0])
        ty = torch.tanh(theta[:, 1])

        min_scale = 1.0
        max_scale = 4.0
        s_logit = theta[:, 2]
        s = min_scale + (max_scale - min_scale) * torch.sigmoid(s_logit)

        zero = torch.zeros_like(tx)
        mat = torch.stack([s, zero, tx, zero, s, ty], dim=1).view(B, 2, 3)
    else:
        raise ValueError("Expect theta (...,3)")
    grid = F.affine_grid(mat, torch.Size((B, 3, out_h, out_w)), align_corners=False)
    return grid


class PrototypeMemory(nn.Module):
    def __init__(self, n_prototypes, proto_dim):
        super().__init__()
        self.np = n_prototypes
        self.proto_dim = proto_dim
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, proto_dim) * 0.05)

    def forward(self):
        return self.prototypes


class PrototypeDecoder(nn.Module):
    """Decode a prototype vector into an RGB patch."""

    def __init__(self, proto_dim, out_channels=3, patch_size=64, base_channels=64):
        super().__init__()
        self.patch_size = patch_size

        n_upsamples = int(math.log2(patch_size // 8))
        assert 2 ** n_upsamples * 8 == patch_size, (
            f"patch_size {patch_size} must be 8 * 2^n (e.g., 32, 64, 128)."
        )

        self.fc = nn.Sequential(
            nn.Linear(proto_dim, base_channels * 8 * 8),
            nn.ReLU(inplace=True),
        )

        layers = []
        in_ch = base_channels
        for _ in range(n_upsamples):
            out_ch = max(in_ch // 2, 16)
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        layers += [nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)]
        self.decoder = nn.Sequential(*layers)

    def forward(self, p_vecs):
        B = p_vecs.size(0)
        n_proto = p_vecs.size(1)
        x = self.fc(p_vecs)
        x = x.view(B * n_proto, -1, 8, 8)
        x = self.decoder(x)
        x = x.view(B, n_proto, 3, self.patch_size, self.patch_size)
        return x


class TransformAndMaskPredictor(nn.Module):
    def __init__(self, emb_dim, patch_size=64, image_size=(128, 128)):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.loc_head = nn.Linear(emb_dim, 3)

        self.mask_head = nn.Sequential(
            nn.Linear(emb_dim, 16 * 16 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 16, 16)),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, e):
        e = e.view(-1, self.emb_dim)
        theta = self.loc_head(e)
        mask_patch = self.mask_head(e)
        return theta, mask_patch


class ProtoSceneModel(nn.Module):
    # This module is the prototype renderer used by the reconstruction experiments:
    # select a prototype, decode it to a patch, predict placement, then composite.
    def __init__(self, n_prototypes, proto_dim, emb_dim, image_size=(128, 128), patch_size=64,
                 use_gumbel=True, temperature=0.5):
        super().__init__()
        self.proto_mem = PrototypeMemory(n_prototypes, proto_dim)
        self.decoder = PrototypeDecoder(proto_dim, out_channels=3, patch_size=patch_size)
        self.transformer = TransformAndMaskPredictor(emb_dim, patch_size=patch_size, image_size=image_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.use_gumbel = use_gumbel
        self.tau = temperature
        self.np = n_prototypes

    def select_prototypes(self, embeddings):
        protos = self.proto_mem()
        emb_norm = F.normalize(embeddings, dim=-1)
        proto_norm = F.normalize(protos, dim=-1)
        logits = emb_norm @ proto_norm.t()
        logits = logits * 10.0

        if self.use_gumbel and self.training:
            dist = RelaxedOneHotCategorical(self.tau, logits=logits)
            soft_one_hot = dist.rsample()
            pvec = soft_one_hot @ protos
            hard_idx = soft_one_hot.argmax(dim=-1)
            return pvec, soft_one_hot, hard_idx

        idx = logits.argmax(dim=-1)
        pvec = protos[idx]
        one_hot = F.one_hot(idx, num_classes=self.np).float()
        return pvec, one_hot, idx

    def forward(self, embeddings):
        K = embeddings.size(0)
        device = embeddings.device
        pvecs, assign_weights, idx = self.select_prototypes(embeddings)

        decoded_patches = self.decoder(pvecs)
        theta, mask_patch = self.transformer(embeddings)

        canvas = torch.zeros(1, 3, self.image_size, self.image_size, device=device)
        alpha_canvas = torch.zeros(1, 1, self.image_size, self.image_size, device=device)

        for i in range(K):
            grid = affine_grid_from_params(theta[i:i+1], self.image_size, self.image_size, device)
            patch = decoded_patches[i:i+1]
            mask_p = mask_patch[i:i+1]
            patch_full = F.grid_sample(patch, grid, align_corners=False)
            mask_full = F.grid_sample(mask_p, grid, align_corners=False)
            canvas = canvas * (1 - mask_full) + patch_full * mask_full
            alpha_canvas = alpha_canvas + mask_full

        return canvas, {
            'decoded_patches': decoded_patches,
            'assign_weights': assign_weights,
            'theta': theta,
            'mask_patch': mask_patch,
            'alpha_canvas': alpha_canvas,
            'idx': idx,
        }

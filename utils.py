"""Shared evaluation metrics, visualization helpers, and training utilities.

Provides:
- ``compute_balanced_accuracy_loader`` — end-of-training balanced accuracy over
  a full DataLoader, supporting both binary and multiclass tasks.
- ``GlobalBalancedAccuracy`` — accumulator for online balanced accuracy during
  training epochs (avoids per-batch bias in imbalanced datasets).
- ``generate_plots`` — qualitative prototype visualizations saved to the run
  output folder.
- ``get_current_lam_entropy`` — entropy regularization schedule.
- ``set_seed`` — deterministic seeding for reproducibility.
"""

import torch
import numpy as np
import random
from scipy.optimize import linear_sum_assignment


def _as_batch_label_matrix(y):
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    return y.reshape(y.shape[0], -1).float()


def compute_balanced_accuracy_batch(y_true, y_pred_bin, eps=1e-10):
    """
    Single batch API for canonical macro balanced accuracy.
    """
    y_true_flat = _as_batch_label_matrix(y_true)
    y_pred_flat = _as_batch_label_matrix(y_pred_bin)

    if y_true_flat.shape != y_pred_flat.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true_flat.shape}, y_pred={y_pred_flat.shape}")

    tp = ((y_pred_flat == 1) & (y_true_flat == 1)).float().sum(dim=0)
    tn = ((y_pred_flat == 0) & (y_true_flat == 0)).float().sum(dim=0)
    fp = ((y_pred_flat == 1) & (y_true_flat == 0)).float().sum(dim=0)
    fn = ((y_pred_flat == 0) & (y_true_flat == 1)).float().sum(dim=0)
    positives = tp + fn
    negatives = tn + fp

    valid_mask = (positives > 0) & (negatives > 0)
    if not valid_mask.any():
        return torch.tensor(0.0, device=tp.device)

    tpr = tp[valid_mask] / (positives[valid_mask] + eps)
    tnr = tn[valid_mask] / (negatives[valid_mask] + eps)
    return (0.5 * (tpr + tnr)).mean()

class GlobalBalancedAccuracy:
    def __init__(self, device='cpu'):
        self.device = device
        self.reset()
        
    def reset(self):
        self.tp = None
        self.tn = None
        self.fp = None
        self.fn = None
        
    def to(self, device):
        self.device = device
        if self.tp is not None:
            self.tp = self.tp.to(device)
            self.tn = self.tn.to(device)
            self.fp = self.fp.to(device)
            self.fn = self.fn.to(device)
        return self
        
    def update(self, y_pred_bin, y_true):
        # Accumulate metrics across batch
        y_true_flat = _as_batch_label_matrix(y_true)
        y_pred_flat = _as_batch_label_matrix(y_pred_bin).to(y_true_flat.device)
        
        device = y_true_flat.device
        
        if y_true_flat.shape != y_pred_flat.shape:
            raise ValueError(f"Shape mismatch: y_true={y_true_flat.shape}, y_pred={y_pred_flat.shape}")

        if self.tp is None:  # First update initialize tensors to the correct size
             num_classes = y_true_flat.shape[1]
             self.tp = torch.zeros(num_classes, device=device)
             self.tn = torch.zeros(num_classes, device=device)
             self.fp = torch.zeros(num_classes, device=device)
             self.fn = torch.zeros(num_classes, device=device)
        
        # Ensure they are on the right device
        if self.tp.device != device:
             self.tp = self.tp.to(device)
             self.tn = self.tn.to(device)
             self.fp = self.fp.to(device)
             self.fn = self.fn.to(device)

        self.tp += ((y_pred_flat == 1) & (y_true_flat == 1)).float().sum(dim=0)
        self.tn += ((y_pred_flat == 0) & (y_true_flat == 0)).float().sum(dim=0)
        self.fp += ((y_pred_flat == 1) & (y_true_flat == 0)).float().sum(dim=0)
        self.fn += ((y_pred_flat == 0) & (y_true_flat == 1)).float().sum(dim=0)

    def __call__(self, y_pred_bin, y_true):
        self.update(y_pred_bin, y_true)
        return self.compute()

    def compute(self, eps=1e-10):
        positives = self.tp + self.fn
        negatives = self.tn + self.fp
        
        valid_mask = (positives > 0) & (negatives > 0)
        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device)
        
        tpr = self.tp[valid_mask] / (positives[valid_mask] + eps)
        tnr = self.tn[valid_mask] / (negatives[valid_mask] + eps)
        
        return (0.5 * (tpr + tnr)).mean()


def get_concept_balance(train_loader, num_concepts, num_objects, DEVICE):
    class_counts = torch.zeros(num_concepts * num_objects, device=DEVICE)
    total_samples = 0 ### NEW: Track total images
    
    for batch in train_loader:
        _, masks, concept_labels, _ = batch[:4]
        batch_size_curr = concept_labels.shape[0]
        
        total_samples += batch_size_curr ### NEW: Count samples

        if concept_labels.dim() == 2:
            if num_objects != 1:
                raise ValueError(f"Expected concept labels with an object dimension for num_objects={num_objects}, got shape {concept_labels.shape}.")
            class_counts[:num_concepts] += concept_labels.sum(dim=0).reshape(-1)
            continue

        # Existing counting logic for multi-object datasets
        for obj_idx in range(num_objects):
            for concept_idx in range(num_concepts):
                flat_idx = obj_idx * num_concepts + concept_idx
                class_counts[flat_idx] += concept_labels[:, obj_idx, concept_idx].sum().item()
    
    print(f"Class counts (Positives): {class_counts}")
    print(f"Total samples: {total_samples}")

    ### NEW: Calculate the weights
    # 1. Calculate Negatives
    # total_samples needs to be broadcastable or iterated, but since class_counts is a tensor:
    num_positives = class_counts
    num_negatives = total_samples - num_positives

    # 2. Calculate pos_weight (Negatives / Positives)
    # Add 1e-6 to avoid division by zero if a class has 0 positives
    pos_weights = num_negatives / (num_positives + 1e-6)

    pos_weights = pos_weights.view(num_objects, num_concepts)  # Reshape to (num_objects, num_concepts)
    
    print(f"Calculated Weights: {pos_weights}")
    pos_weights = pos_weights.to(DEVICE)

    return pos_weights, num_positives, num_negatives


def calculate_kl_loss(z_posterior_mean, z_posterior_logvar):
    # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # 1. Calculate the KL term for every element in the batch and embedding
    # We use .exp() on logvar to get sigma^2 safely
    kl_elementwise = -0.5 * (1 + z_posterior_logvar - z_posterior_mean.pow(2) - z_posterior_logvar.exp())

    # 2. Sum across the embedding dimension (dim=1) 
    # This gives you the KL "cost" for each individual image in the batch
    kl_per_sample = kl_elementwise.sum(dim=1)

    # 3. Take the average across the batch (dim=0)
    # This ensures your loss magnitude doesn't change if you change batch size
    kl_loss = kl_per_sample.mean()

    return kl_loss


def check_prototypes(prototypes, concept_names):
    """
    Check which prototypes (concept combinations) appear in the given prototype tensor.

    Args:
        prototypes (torch.Tensor): Shape (nb_proto, nb_concepts).
        concept_names (list[str]): List of concept names corresponding to each column.
    """
    nb_proto, nb_concepts = prototypes.shape
    concept_flags = prototypes > 0.5

    counts = {}

    for k in range(nb_proto):
        active_idx = concept_flags[k].nonzero(as_tuple=True)[0].tolist()
        if len(active_idx) > 0:
            key = tuple(active_idx)
            counts[key] = counts.get(key, 0) + 1

    # Print results, sorted
    strings = []
    for key, count in counts.items():
        concept_str = " + ".join(concept_names[i] for i in key)
        # print(f"Prototype: {concept_str} → found {count} times")
        strings.append(f"Prototype: {concept_str} → found {count} times")
    print("\n".join(sorted(strings)))


def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_balanced_accuracy_loader(model, loader, interventions_mask=None, standard_interventions_mask=None, c_indices=None):
    """
    Compute task and concept accuracy for a whole dataloader.
    If model.use_balanced_accuracy is True, this computes balanced accuracy using
    compute_balanced_accuracy_batch on concatenated predictions over the loader.
    Args:
        model: torch.nn.Module returning (y_pred, c_pred)
        loader: DataLoader yielding (x, c_true, y_true)
    Returns:
        task_acc (float), concept_acc (float)
    """
    all_y_true = []
    all_y_pred = []
    all_c_true = []
    all_c_pred = []

    total_y_correct = 0.0
    total_y = 0.0
    total_c_correct = 0.0
    total_c = 0.0
    
    for batch in loader:   # TODO: better way to deal with m
        batch = tuple(elem.to(model.device) if torch.is_tensor(elem) else elem for elem in batch)
        x, m, c, y = batch[:4]
        model_input = batch if len(batch) > 4 else (x, m, c, y)
        y_pred, c_pred, e, _ = model(model_input, interventions_mask=interventions_mask, standard_interventions_mask=standard_interventions_mask)
        c_pred = (c_pred > 0.5).float()
        # keep task and concept dimensions to compute per-label metrics

        model_dataset = getattr(model, "dataset_name", getattr(model, "dataset", None))
        if model_dataset == "cubEMB" and c.dim() == 2:
            c = c.unsqueeze(1)
        if c_pred.dim() == 3 and c.dim() == 2:
            c = c.unsqueeze(1)

        if c_indices is not None:
            idx_list = [int(idx.item()) if isinstance(idx, torch.Tensor) else int(idx) for idx in c_indices]
            c = c[:, :, idx_list]
            c_pred = c_pred[:, :, idx_list]
        
        is_multiclass_task = getattr(model, "is_cub", False) or getattr(model, "is_multiclass_task", False) or model_dataset in ("cub", "cubEMB", "cub_presegm")
        if is_multiclass_task:
            y_pred_classes = y_pred.argmax(dim=-1)
            y_true_classes = y.argmax(dim=-1) if y.dim() > 1 else y.long()
            total_y_correct += (y_pred_classes == y_true_classes).float().sum().item()
            total_y += y_true_classes.numel()
            all_y_true.append(y_true_classes.detach().cpu())
            all_y_pred.append(y_pred_classes.detach().cpu())
        else:
            y_pred_bin = (y_pred > 0.5).float()
            total_y_correct += (y_pred_bin == y).float().sum().item()
            total_y += y.numel()
            all_y_true.append(y.detach().cpu())
            all_y_pred.append(y_pred_bin.detach().cpu())

        total_c_correct += (c_pred == c).float().sum().item()
        total_c += c.numel()

        all_c_true.append(c.detach().cpu())
        all_c_pred.append(c_pred.detach().cpu())
    
    use_balanced = getattr(model, 'use_balanced_accuracy', False)

    # Task accuracy is always regular (never balanced).
    y_acc = total_y_correct / max(total_y, 1.0)

    if use_balanced:
        c_true_cat = torch.cat(all_c_true).to(model.device)
        c_pred_cat = torch.cat(all_c_pred).to(model.device)

        c_metric = GlobalBalancedAccuracy().to(model.device)
        c_metric.update(c_pred_cat, c_true_cat)
        c_acc = c_metric.compute().item()
    else:
        c_acc = total_c_correct / max(total_c, 1.0)

    return y_acc, c_acc

def compute_acc_unordered(model, loader, num_objects=2):
    """
    Compute task accuracy and concept accuracy for unordered object predictions.

    Args:
        model: torch.nn.Module returning (c_pred, y_pred)
        loader: DataLoader yielding (x, c_true, y_true)

    Returns:
        task_acc (float), concept_acc (float)
    """
    model.eval()
    device = next(model.parameters()).device

    total_task_correct = 0
    total_task_samples = 0
    total_concept_correct = 0
    total_concept_total = 0

    for x, c_true, y_true in loader:
        x, c_true, y_true = x.to(device), c_true.to(device), y_true.to(device)

        b = x.shape[0]

        # Forward pass
        y_pred, c_pred, e, _ = model((x, c_true, y_true))

        c_true = c_true.view(b, num_objects, -1)  # [B, nb_objects, nb_concepts]
        c_pred = c_pred.view(b, num_objects, -1)  # [B, nb_objects, nb_concepts]

        # ====== Task accuracy ======
        y_pred_cls = y_pred.argmax(dim=-1)
        y_true_cls = y_true.argmax(dim=-1)
        total_task_correct += (y_pred_cls == y_true_cls).sum().item()
        total_task_samples += y_true_cls.numel()

        # ====== Concept accuracy (unordered, Hungarian matching) ======
        B, nb_objects, nb_concepts = c_true.shape
        for b in range(B):
            true_b = c_true[b]        # [nb_objects, nb_concepts]
            pred_b = c_pred[b]        # [nb_objects, nb_concepts]

            # Compute cost matrix (L2 distance)
            cost = torch.cdist(true_b, pred_b, p=2).detach()
            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

            matched_true = true_b[row_ind]
            matched_pred = pred_b[col_ind]

            # Concept predictions and targets (class indices)
            pred_classes = matched_pred.argmax(dim=-1)
            true_classes = matched_true.argmax(dim=-1)

            total_concept_correct += (pred_classes == true_classes).sum().item()
            total_concept_total += nb_objects

    task_acc = total_task_correct / total_task_samples
    concept_acc = total_concept_correct / total_concept_total

    return task_acc, concept_acc


def map_concepts_of_prototype_to_actual_labels_celeba(concepts_of_prototype):
    """
    Map the boolean array of concepts of a prototype to their actual labels in CelebA.
    """
    all_concepts = '5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young'.split(', ')
    concept_labels = []
    for i, val in enumerate(concepts_of_prototype):
        if val:
            concept_labels.append(all_concepts[i])
    concept_labels_string = ", ".join(concept_labels)
    return concept_labels_string

def map_concepts_of_prototype_to_actual_labels_clevrhans(concepts_of_prototype):
    """
    Map the boolean array of concepts of a prototype to their actual labels in CLEVR.
    """
    CLEVR_HANS_SIZES = ['large', 'small']
    CLEVR_HANS_COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
    CLEVR_HANS_SHAPES = ['cube', 'sphere', 'cylinder']
    CLEVR_HANS_MATERIALS = ['rubber', 'metal']

    all_concepts = (
        [f'size_{s}' for s in CLEVR_HANS_SIZES] +
        [f'color_{c}' for c in CLEVR_HANS_COLORS] +
        [f'shape_{s}' for s in CLEVR_HANS_SHAPES] +
        [f'material_{m}' for m in CLEVR_HANS_MATERIALS]
    )  # 15 concepts total

    concept_labels = []
    for i, val in enumerate(concepts_of_prototype):
        if val:
            concept_labels.append(all_concepts[i])
    concept_labels_string = ", ".join(concept_labels)
    return concept_labels_string

def map_concepts_of_prototype_to_actual_labels(concepts_of_prototype):
    """
    Map the boolean array of concepts of a prototype to their actual labels.
    """
    concept_labels = []
    for i, val in enumerate(concepts_of_prototype):
        if val:
            if i < 10:
                concept_labels.append(str(i))
            elif i == 10:
                concept_labels.append('red')
            elif i == 11:
                concept_labels.append('blue')
            elif i == 12:
                concept_labels.append('green')
            else:
                concept_labels.append(f'unknown{i}')

    concept_labels_string = ", ".join(concept_labels)
    return concept_labels_string


def get_used_prototypes_indices(model, loader):
    """
    Get the indices of prototypes that are used by the model on the given data loader.
    """
    used_indices = set()
    for elem in loader:
        elem = tuple(item.to(model.device) if torch.is_tensor(item) else item for item in elem)
        _, _, _, info = model(elem)
        prototype_probs_per_object = info['prototype_probs_per_object']  # (b, nb_objects, nb_proto)
        b, nb_objects, nb_proto = prototype_probs_per_object.shape
        prototype_probs_per_object = prototype_probs_per_object.reshape(b * nb_objects, nb_proto)
        max_indices = torch.argmax(prototype_probs_per_object, dim=1)  # (b * nb_objects)
        used_indices.update(max_indices.cpu().numpy().tolist())
    return used_indices


def tokp_objects_for_prototype(model, val_loader, top_k=5):
    """
    For each used prototype in the model, find the top_k objects from the validation loader that most activate it.
    """
    best_probs = []
    best_objects = []
    for batch in val_loader:
        batch = tuple(item.to(model.device) if torch.is_tensor(item) else item for item in batch)
        _, _, _, info = model(batch)
        prototype_probs_per_object = info['prototype_probs_per_object'].detach().cpu()  # (b, nb_objects, nb_proto)
        objects = info['objects'].detach().cpu()  # (batch, nb_objects, nb_channels, H, W)

        # flatten batch and nb_objects dimensions into one dim
        b, nb_objects, nb_proto = prototype_probs_per_object.shape
        prototype_probs_per_object = prototype_probs_per_object.reshape(b * nb_objects, nb_proto)
        objects = objects.reshape(b * nb_objects, *objects.shape[2:])

        
        # keep the top_k probabilities and corresponding objects
        topk_probs, topk_indices = torch.topk(prototype_probs_per_object, top_k, dim=0)  # (top_k, nb_proto)
        best_probs.append(topk_probs)  # list of (top_k, nb_proto)
        best_objects.append(objects[topk_indices])  # list of (top_k, nb_proto, nb_channels, H, W)
    
    return best_probs, best_objects

def get_input_images_and_reconstructions(model, val_loader):
    """
    Get input images and their reconstructions from the model on the validation loader.
    """
    reconstructions = []
    input_images = []
    for batch in val_loader:
        batch = tuple(item.to(model.device) if torch.is_tensor(item) else item for item in batch)
        _, _, _, info = model(batch)
        recon_x = info['reconstruction']  # (batch, nb_channels, H, W)
        reconstructions.append(recon_x.detach())
        input_images.append((batch[4] if len(batch) > 4 else batch[0]).detach())
    return input_images, reconstructions

def get_objects_notraining(x, m):
    #### NO TRAINING SEGMENTATION

    # x: (B, C, H, W)
    # m: (B, nb_objects, H, W)

    segmentation = m.unsqueeze(2) * x.unsqueeze(1)  # (B, nb_masks, C, H, W)

    return segmentation

def reconstruction_util(map_proto_to_image, prototype_embeddings, z_sampled_expanded):
    # prototype_embeddings_batch = prototype_embeddings.unsqueeze(0).expand(z_sampled_expanded.shape[0], -1, -1)  # (nb_proto, proto_size)

    # prototype_embeddings_and_z = self.combine_z_and_proto_emb(torch.cat([prototype_embeddings_batch, z_sampled_expanded], dim=-1))  # (b, nb_proto, proto_size)
    prototypes_as_images = map_proto_to_image(prototype_embeddings.unsqueeze(0))  # (1, nb_proto, 3, H, W)
    return prototypes_as_images

def get_current_lam_entropy(current_epoch, max_epochs, lam_entropy_start, lam_entropy_end=0.0):
    warmup_epochs = 20
    total_epochs = max_epochs
    
    if current_epoch < warmup_epochs:
        return lam_entropy_start
    
    progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return lam_entropy_start - progress * (lam_entropy_start - lam_entropy_end)

def get_mnist_objects(x):
    #### MNIST

    # x: (B, C, H, W)
    x0 = x[:, :, (x.shape[2] // 2):]  # bottom half
    x1 = x[:, :, :(x.shape[2] // 2)]  # top half

    # padding them (adding 28x14 zeros to the top of x0 and the bottom of x1)
    x0 = torch.cat([torch.zeros_like(x0), x0], dim=2)
    x1 = torch.cat([x1, torch.zeros_like(x1)], dim=2)

    # stack them
    m = torch.stack([x0, x1], dim=1)  # (B, 2, C, H, W)
    # the masks are where the object is greater than 0
    masks = (m.sum(dim=2) > 0).float()  # (B, 2, H, W)
    # print(x0.shape, x1.shape, m.shape, masks.shape)
    
    return m, masks

def generate_plots(model, logger=None, output_folder=None):
    import os
    current_epoch = model.current_epoch
    if output_folder is not None:
        model.output_folder = output_folder
        current_epoch = -1
    if not os.path.exists(os.path.join(model.output_folder, f"epoch_{current_epoch}")):
        os.makedirs(os.path.join(model.output_folder, f"epoch_{current_epoch}"))
        os.makedirs(os.path.join(model.output_folder, f"epoch_{current_epoch}/prototypes"))
        os.makedirs(os.path.join(model.output_folder, f"epoch_{current_epoch}/closest_objects"))

    logger = logger if logger else model._logger

    # Check which prototypes appear in the given prototype tensor.
    check_prototypes(model.stored_activations, concept_names=model.concept_names)

    # for each prototype we want to check if it is used at least once and keep a track of the 'used' indices
    used_indices = get_used_prototypes_indices(model, model.val_loader)
    print(f"Used {len(used_indices)} prototypes out of {model.prototypes.weight.shape[0]}")
    print(f"Used prototypes: {used_indices}")

    if logger is not None:
        import wandb
        logger.experiment.log({
            "used_prototypes_count": wandb.Html(f"Used {len(used_indices)} prototypes out of {model.prototypes.weight.shape[0]}"),
            "used_prototypes": wandb.Html(str(used_indices))
        }, step=current_epoch)

    if getattr(model, "dataset_name", None) == "cubEMB":
        _plot_intervenability(model, logger=logger, current_e=current_epoch)
        return

    _plot_best_object_per_prototype(model, used_indices, logger=logger, current_e=current_epoch)
    _plot_concept_table_and_prototypes(model, used_indices, logger=logger, current_e=current_epoch)
    _plot_posterior_prototypes_examples(model, logger=logger, current_e=current_epoch)
    _plot_intervenability(model, logger=logger, current_e=current_epoch)

def _plot_best_object_per_prototype(model, used_indices, logger=None, top_k=5, current_e=None):
    import os
    current_epoch = current_e if current_e is not None else model.current_epoch
    
    best_probs, best_objects = tokp_objects_for_prototype(model, model.val_loader, top_k=top_k)

    best_probs = torch.cat(best_probs, dim=0)  # (top_k * nb_batches, nb_proto)
    best_objects = torch.cat(best_objects, dim=0)  # (top_k * nb_batches, nb_proto, nb_channels, H, W)

    topk_final_probs, topk_final_indices = torch.topk(best_probs, top_k, dim=0)  # (top_k, nb_proto)
    best_objects = best_objects[topk_final_indices, torch.arange(model.nb_proto).unsqueeze(0).expand(top_k, -1)]  # (top_k, nb_proto, nb_channels, H, W)

    # save fig for the first prototype, the best object
    for i in used_indices:
        # Create a single image by concatenating all top_k images for this prototype horizontally
        imgs = [best_objects[j, i].permute(1, 2, 0).detach().cpu().numpy() for j in range(top_k)]
        # Ensure all images are the same shape
        imgs = [img if img.shape[2] == 3 else img[:, :, :3] for img in imgs]  # handle possible alpha channel
        import numpy as np
        concat_img = np.concatenate(imgs, axis=1)  # concatenate along width

        if concat_img.shape[2] == 1:
            concat_img = np.repeat(concat_img, 3, axis=2)  # convert to 3 channels for logging
        # normalize to 0-1

        concat_img = np.clip(concat_img, 0, 1)

        import matplotlib.pyplot as plt
        plt.imshow(concat_img)
        plt.axis('off')
        concepts_of_prototype = model.stored_activations[i] > 0.5
        if model.dataset_name == "mnist":
            concepts_string = map_concepts_of_prototype_to_actual_labels(concepts_of_prototype)
        elif model.dataset_name == "celebamask":
            concepts_string = map_concepts_of_prototype_to_actual_labels_celeba(concepts_of_prototype)
        elif model.dataset_name == "clevrhans":
            concepts_string = map_concepts_of_prototype_to_actual_labels_clevrhans(concepts_of_prototype)
        elif getattr(model, "concept_names", None) is not None:
            concepts_string = ", ".join(model.concept_names[i] for i, val in enumerate(concepts_of_prototype) if val)
        else:
            concepts_string = ", ".join(f"concept_{i}" for i, val in enumerate(concepts_of_prototype) if val)
        plt.title(concepts_string)  
        
        plt.savefig(os.path.join(model.output_folder, f"epoch_{current_epoch}/closest_objects/best_object_proto_{i}.png"), bbox_inches='tight', pad_inches=0)
        
        fig = plt.gcf()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        plot_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        plot_array = plot_array.reshape(height, width, 4)[:, :, 1:] # take only RGB channels
        plot_tensor = torch.tensor(plot_array)  

        if logger is not None:
            import wandb
            logger.experiment.log({f"best_objects/prototype_{i}": wandb.Image(plot_tensor.numpy())}, step=model.current_epoch)
        
        plt.close()

def _plot_concept_table_and_prototypes(model, used_indices, logger=None, current_e=None):
    import os
    current_epoch = current_e if current_e is not None else model.current_epoch
    prototypes_as_images = model.get_prototypes_as_images().squeeze()  # (nb_proto, 3, H, W)

    import numpy as np
    import matplotlib.pyplot as plt

    # All prototypes as images are saved in the output folder
    for i in range(prototypes_as_images.shape[0]):
        img = prototypes_as_images[i].permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.axis('off')
        concepts_of_prototype = model.stored_activations[i] > 0.5
        if model.dataset_name == "mnist":
            concepts_string = map_concepts_of_prototype_to_actual_labels(concepts_of_prototype)
        elif model.dataset_name == "celebamask":
            concepts_string = map_concepts_of_prototype_to_actual_labels_celeba(concepts_of_prototype)
        elif model.dataset_name == "clevrhans":
            concepts_string = map_concepts_of_prototype_to_actual_labels_clevrhans(concepts_of_prototype)
        elif getattr(model, "concept_names", None) is not None:
            concepts_string = ", ".join(model.concept_names[i] for i, val in enumerate(concepts_of_prototype) if val)
        else:
            concepts_string = ", ".join(f"concept_{i}" for i, val in enumerate(concepts_of_prototype) if val)
        plt.title(concepts_string)
        plt.savefig(os.path.join(model.output_folder, f"epoch_{current_epoch}", "prototypes", f"prototype_{i}.png"), bbox_inches='tight', pad_inches=0)

        fig = plt.gcf()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        plot_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        plot_array = plot_array.reshape(height, width, 4)[:, :, 1:]
        plot_tensor = torch.tensor(plot_array)

        if logger is not None:
            import wandb
            logger.experiment.log({f"prototypes/prototype_{i}": wandb.Image(plot_tensor.numpy())}, step=model.current_epoch)
        plt.close()

    # Table plotting
    fig, axes = plt.subplots(len(used_indices),  len(model.concept_names)+1, figsize=(len(model.concept_names) * 2, len(used_indices) * 2))
    
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, proto_idx in enumerate(used_indices):
        img = prototypes_as_images[proto_idx].permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        concepts_of_prototype = model.stored_activations[proto_idx] > 0.5
        for col_idx, concept in enumerate(model.concept_names):
            if concepts_of_prototype[col_idx]:
                axes[row_idx, col_idx+1].text(0.5, 0.5, "✔", fontsize=32, ha='center', va='center')
                axes[row_idx, col_idx+1].axis('off')
            else:
                axes[row_idx, col_idx+1].text(0.5, 0.5, "✘", fontsize=32, ha='center', va='center')
                axes[row_idx, col_idx+1].axis('off')
            axes[row_idx, col_idx+1].set_xticks([])
            axes[row_idx, col_idx+1].set_yticks([])

    for col_idx, concept in enumerate(["Prototypes"] + model.concept_names):
        axes[0, col_idx].set_title(concept, fontsize=16) 
        axes[0, col_idx].axis('off')

    plt.savefig(os.path.join(model.output_folder, f"epoch_{current_epoch}", "concept_table.png"), bbox_inches='tight', pad_inches=0)

    if logger is not None:
        import wandb
        # Keep WandB image size bounded for very large concept tables.
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        max_dim = 4096

        if max(width, height) <= max_dim:
            logger.experiment.log({"concept_table": wandb.Image(fig)}, step=current_epoch)
        else:
            print(f"Skipping WandB concept_table log because figure is too large ({width}x{height}px).")

    plt.close()

def _plot_posterior_prototypes_examples(model, logger=None, current_e=None):
    if model.last_prototypes_as_images_posterior is None:
        return
    import os
    current_epoch = current_e if current_e is not None else model.current_epoch
    
    if model.last_prototypes_as_images_posterior.shape[0] == 1:
        model.last_prototypes_as_images_posterior = model.last_prototypes_as_images_posterior.expand(model.last_X.shape[0], -1, -1, -1, -1)
    if model.last_prototype_probs_per_object.shape[0] == 1:
        model.last_prototype_probs_per_object = model.last_prototype_probs_per_object.expand(model.last_X.shape[0], -1, -1)

    prototype_idx_per_object_examples = torch.argmax(model.last_prototype_probs_per_object, dim=2)  # (b, nb_objects)
    prototype_per_object_examples = model.last_prototypes_as_images_posterior.unsqueeze(1).expand(-1, prototype_idx_per_object_examples.shape[1], -1, -1, -1, -1)  # (b, nb_objects, nb_proto, 3, H, W)
    selected_prototypes_per_object_examples = prototype_per_object_examples[torch.arange(prototype_idx_per_object_examples.shape[0]).unsqueeze(1), torch.arange(prototype_idx_per_object_examples.shape[1]).unsqueeze(0), prototype_idx_per_object_examples]  # (b, nb_objects, 3, H, W)
    
    import numpy as np
    import matplotlib.pyplot as plt
    num_examples = min(10, selected_prototypes_per_object_examples.shape[0])
    num_objects = selected_prototypes_per_object_examples.shape[1]
    
    fig, axes = plt.subplots(num_examples, num_objects + 1, figsize=((num_objects + 1) * 2, num_examples * 2))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_examples):
        orig_img = model.last_X[i].detach().cpu().numpy().transpose(1, 2, 0)
        orig_img = np.clip(orig_img, 0, 1)
        axes[i, 0].imshow(orig_img)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original')
        axes[i, 0].set_ylabel(f'Example {i}', rotation=90, size='large')
        
        for j in range(num_objects):
            img = selected_prototypes_per_object_examples[i, j].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            axes[i, j + 1].imshow(img)
            axes[i, j + 1].axis('off')
            if i == 0:
                axes[i, j + 1].set_title(f'Object {j}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model.output_folder, f"epoch_{current_epoch}", 'posterior_prototypes_per_object.png'))
    
    if logger is not None:
        import wandb
        logger.experiment.log({"posterior_prototypes/overview": wandb.Image(fig)}, step=current_epoch)
    
    plt.close()

def _plot_intervenability(model, logger=None, current_e=None):
    import os
    current_epoch = current_e if current_e is not None else model.current_epoch
    total_nb_concepts = model.nb_possibleobjects * model.nb_concepts

    nb_interventions_list = []
    y_accuracies, y_accuracies_standard = [], []
    c_accuracies, c_accuracies_standard = [], []

    shuffle_order = torch.randperm(model.nb_possibleobjects * model.nb_concepts)
    for nb_interventions in range(0, total_nb_concepts, max(1, total_nb_concepts // 10)):
        intervention_mask = torch.zeros(model.nb_possibleobjects * model.nb_concepts)
        intervention_mask[:nb_interventions] = 1
        intervention_mask = intervention_mask[shuffle_order].view(model.nb_possibleobjects, model.nb_concepts)
        y_acc, c_acc = compute_balanced_accuracy_loader(model, model.val_loader, interventions_mask=intervention_mask.to(model.device))
        y_acc_standard, c_acc_standard = compute_balanced_accuracy_loader(model, model.val_loader, standard_interventions_mask=intervention_mask.to(model.device))
        nb_interventions_list.append(nb_interventions)
        y_accuracies.append(y_acc)
        c_accuracies.append(c_acc)
        y_accuracies_standard.append(y_acc_standard)
        c_accuracies_standard.append(c_acc_standard)

    use_balanced = getattr(model, 'use_balanced_accuracy', False)
    concept_prefix = "Balanced " if use_balanced else ""

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(nb_interventions_list, y_accuracies, marker='o', label='Interventions')
    plt.plot(nb_interventions_list, y_accuracies_standard, marker='x', label='Standard Interventions')
    plt.xlabel('Number of Intervened Concepts')
    plt.ylabel('Task Accuracy')
    plt.title('Task Accuracy vs Number of Intervened Concepts')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(model.output_folder, f"epoch_{current_epoch}", 'intervenability_task_accuracy.png'))
    if logger is not None:
        import wandb
        logger.experiment.log({"intervenability/task_accuracy": wandb.Image(plt.gcf())}, step=current_epoch)
    plt.close()

    plt.figure()
    plt.plot(nb_interventions_list, c_accuracies, marker='o', label='Interventions')
    plt.plot(nb_interventions_list, c_accuracies_standard, marker='x', label='Standard Interventions')
    plt.xlabel('Number of Intervened Concepts')
    plt.ylabel(f'{concept_prefix}Concept Accuracy')
    plt.title(f'{concept_prefix}Concept Accuracy vs Number of Intervened Concepts')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(model.output_folder, f"epoch_{current_epoch}", 'intervenability_concept_accuracy.png'))
    if logger is not None:
        import wandb
        logger.experiment.log({"intervenability/concept_accuracy": wandb.Image(plt.gcf())}, step=current_epoch)
    plt.close()

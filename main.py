"""Training and evaluation entry point for PGCM and all competitor baselines.

Reads a YAML config (``--config``), builds the chosen dataset and model, runs
Lightning training with checkpointing and early stopping, and logs final
metrics to Weights & Biases.  After training the best checkpoint is restored
and evaluated on train / val / test splits, followed by intervenability
analysis.

Usage::

    python main.py --config configs/config_cubEMB.yml --device cuda:0
"""

from dataset import get_mnist, get_celeba_dataset, get_clevrhans_dataset, get_cub_dataset, load_presegmented_dataloaders
from dataset import CLEVR_HANS_ALL_CONCEPTS, CLEVR_HANS_NUM_CONCEPTS, CLEVR_HANS_NUM_CLASSES, CLEVR_HANS_MAX_OBJECTS
from dataset import (CUB_PRESEGM_PARTS, CUB_PRESEGM_NUM_CLASSES, CUB_PRESEGM_DIR,
                     _parse_cub_attribute_names, _build_cub_part_to_attribute_indices,
                     load_cub_presegmented_dataloaders, load_cub_presegmented_dataloaders_nonpgcm)
import torch
from model import Model
import lightning.pytorch as pl
import os
import argparse
from utils import set_seed, compute_balanced_accuracy_loader, compute_acc_unordered, get_concept_balance
from datetime import datetime
from competitors import DNN, CBMDeep, CRM, CMR, _plot_intervenability
from torchvision.utils import save_image
from lightning.pytorch.loggers import WandbLogger


def inspect_val_predictions_with_pause(model, val_loader, concept_names, output_folder, device):
    """Iterate over validation samples, save current image, print predicted concepts, and pause."""
    save_path = os.path.join(output_folder, "val_sample_current.png")
    model.eval()
    sample_idx = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(b.to(device) if torch.is_tensor(b) else b for b in batch)
            _, c_pred, _, _ = model(batch)

            images = batch[0].detach().cpu()
            concept_probs = c_pred.detach().cpu()
            if concept_probs.dim() == 2:
                concept_probs = concept_probs.unsqueeze(1)

            for i in range(images.shape[0]):
                save_image(images[i].clamp(0.0, 1.0), save_path)

                pred_binary = concept_probs[i] > 0.5
                pred_per_object = []
                nb_concepts = min(pred_binary.shape[-1], len(concept_names))
                for obj_idx in range(pred_binary.shape[0]):
                    active = [
                        concept_names[c_idx]
                        for c_idx in range(nb_concepts)
                        if pred_binary[obj_idx, c_idx].item()
                    ]
                    pred_per_object.append(active)

                print(f"[val sample {sample_idx}] saved image to: {save_path}")
                print(f"[val sample {sample_idx}] predicted concepts per object: {pred_per_object}")

                user_in = input("Press Enter for next sample (or type 'q' to quit): ").strip().lower()
                sample_idx += 1
                if user_in == 'q':
                    print("Stopped validation-sample inspection loop.")
                    return

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--device", type=str, default="cpu", help="Which device to use for training (e.g., 'cpu', 'cuda:0').")
    parser.add_argument("--extra", type=str, default="", help="Extra string to append to the log and run folder names.")
    parser.add_argument("--test-only", action='store_true', help="Whether to skip training and only evaluate the model.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="The path to the .ckpt file to load if --load-check is enabled.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility across runs.")
    parser.add_argument("--disable-early-stopping", action='store_true', help="Disable EarlyStopping callback.")

    cmd_args = parser.parse_args()

    import yaml
    with open(cmd_args.config, 'r') as file:
        config_dict = yaml.safe_load(file)

    # Merge cmd arguments and config_dict into a single namespace
    # cmd arguments taking precedence
    final_args = argparse.Namespace()
    for k, v in config_dict.items():
        setattr(final_args, k, v)
    
    # Override with command line arguments if explicitly provided
    setattr(final_args, "config", cmd_args.config)
    setattr(final_args, "device", cmd_args.device)
    setattr(final_args, "extra", cmd_args.extra)
    setattr(final_args, "test_only", cmd_args.test_only)
    setattr(final_args, "checkpoint_path", cmd_args.checkpoint_path)
    setattr(final_args, "seed", cmd_args.seed)
    setattr(final_args, "disable_early_stopping", cmd_args.disable_early_stopping)

    return final_args

def main(args):

    DEVICE = args.device
    epochs = args.epochs
    batch_size = args.batch_size
    dataset = args.dataset
    lr = args.lr
    lam_entropy = getattr(args, "lam_entropy", 0.0)
    lam_batch_entropy = getattr(args, "lam_batch_entropy", 0.0)
    decay_lam_entropy = getattr(args, "decay_lam_entropy", True)
    lam_reconstruction = getattr(args, "lam_reconstruction", 0.0)
    lam_segmentation = getattr(args, "lam_segmentation", 1.0)
    lam_orth = getattr(args, "lam_orth", 0.0)
    segmentation_method = getattr(args, "segmentation_method", "true")
    use_pretrained_segmenter = getattr(args, "use_pretrained_segmenter", False)
    presegmented_datasets_path = getattr(args, "presegmented_datasets_path", None)
    num_workers = getattr(args, "num_workers", 0)
    use_pretrained_autoencoder = getattr(args, "use_pretrained_autoencoder", False)
    nb_proto_override = getattr(args, "nb_proto", None)
    lam_kl = getattr(args, "lam_kl", 0.0)
    plot_frequency = getattr(args, "plot_frequency", 10)
    concepts_to_task = getattr(args, "concepts_to_task", "thresholding")
    warmup_epochs = getattr(args, "warmup_epochs", 0)
    intv_prob = getattr(args, "intv_prob", 0.0)
    fixed_lr = getattr(args, "fixed_lr", False)
    always_use_true_masks = getattr(args, "always_use_true_masks", False)
    use_initial_proto_embs = getattr(args, "use_initial_proto_embs", False)
    rule_emb_size = getattr(args, "rule_emb_size", 100)
    noisy_prob = getattr(args, "noisy_prob", 0.0)
    noisy_digit = getattr(args, "noisy_digit", None)
    noisy_target_digit = getattr(args, "noisy_target_digit", None)
    noisy_part = getattr(args, "noisy_part", None)
    noisy_target_part = getattr(args, "noisy_target_part", None)
    # CLEVR-Hans specific noisy concept parameters
    noisy_concept_names = getattr(args, "noisy_concept_names", None)
    target_noisy_concept_names = getattr(args, "target_noisy_concept_names", None)

    proto_size = getattr(args, "protosize", 64)
    embedding_size = args.embedding_size

    test_only = args.test_only

    extra_string = args.extra

    use_weights = args.use_weights
    disable_early_stopping = getattr(args, "disable_early_stopping", False)

    if use_weights:
        extra_string = "_useweights_" + extra_string

    checkpoint_path = args.checkpoint_path

    set_seed(args.seed)

    if use_pretrained_segmenter and args.model != "PGCM":
        raise NotImplementedError("use_pretrained_segmenter is currently only supported for PGCM.")
    if use_pretrained_segmenter and segmentation_method != 'mask':
        raise ValueError("use_pretrained_segmenter requires segmentation_method='mask'.")
    if use_pretrained_segmenter and presegmented_datasets_path is None and dataset != "cub_presegm":
        raise ValueError("When use_pretrained_segmenter=True, set presegmented_datasets_path.")
    if dataset == "cubEMB" and args.model not in ("PGCM", "CBM", "CRM", "CMR", "DNN"):
        raise NotImplementedError("CUBEMB training is only implemented for PGCM, CBM, CRM, CMR, and DNN.")
    
    # Create output folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.extra != "":
        base_output_folder = f"./new_outputs/{timestamp}_{dataset}_{args.extra}"
    else:
        base_output_folder = f"./new_outputs/{timestamp}_{dataset}"
    
    output_folder = os.path.join(base_output_folder, "outputs")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "prototypes"), exist_ok=True)

    with open(os.path.join(output_folder, "args.txt"), "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")


    if dataset == "mnist":
        color_as_concepts = False
        allowed_colors = ['red', 'blue', 'green']
        allowed_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        EXTRA_PROTOTYPES = 20

        all_concepts = [str(x) for x in allowed_digits]
        if color_as_concepts:
            all_concepts += allowed_colors

        number_of_digits = len(allowed_digits)
        if color_as_concepts:
            number_of_colors = len(allowed_colors)
        else:
            number_of_colors = 0

        number_of_colors_for_proto = len(allowed_colors) if color_as_concepts else 1
        num_prototypes = number_of_digits * number_of_colors_for_proto + EXTRA_PROTOTYPES  # each prototype corresponds to one valid combination of concepts
        num_concepts = number_of_digits + number_of_colors  # one concept per digit and one concept per color
        num_tasks = (number_of_digits * 2) - 1 # one task per valid combination of concepts
        num_objects = 2

        train_loader, val_loader, test_loader = get_mnist(batch_size=batch_size, allowed_digits=allowed_digits, allowed_colors=allowed_colors, 
                                                          color_as_concepts=color_as_concepts, noisy_digit=noisy_digit, noisy_target_digit=noisy_target_digit, noisy_prob=noisy_prob,                                         
                                                          device=DEVICE, num_workers=num_workers)

        resolution = (56, 28)

    elif dataset == "celebamask":
        # used_masks=['skin', 'hair']#, 'nose', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_lip', 'u_lip', 'neck']
        used_masks=['skin', 'hair', 'nose', 'lips']#, 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_lip', 'u_lip', 'neck']
        all_concepts = '5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young'.split(', ')

        # not used:  Attractive, Blurry, Male, Wearing_Earrings, Wearing_Hat
        # maybe predict:  Attractive, Male, Young
        noisy_part_index = None
        noisy_target_part_index = None
        if noisy_part is not None and noisy_target_part is not None:
            if noisy_part in all_concepts and noisy_target_part in all_concepts:
                noisy_part_index = all_concepts.index(noisy_part)
                noisy_target_part_index = all_concepts.index(noisy_target_part)
                print(f"Injecting noise in concept '{noisy_part}' (index {noisy_part_index}) to become '{noisy_target_part}' (index {noisy_target_part_index}) with probability {noisy_prob}.")
            else:
                raise ValueError("noisy_part and noisy_target_part must be valid concept names.")

        task_concepts = ['Attractive', 'Male', 'Young']

        masks_to_associated_concepts = {
            'skin': ['Pale_Skin', 'Rosy_Cheeks', 'Heavy_Makeup', 'Chubby', '5_o_Clock_Shadow', 'Bags_Under_Eyes', 'Eyeglasses', 'Goatee', 'High_Cheekbones', 'Mustache', 'No_Beard', 'Oval_Face'],
            'nose': ['Big_Nose', 'Pointy_Nose'],
            'l_lip': ['Big_Lips', 'Wearing_Lipstick', 'Mouth_Slightly_Open', 'Smiling'],
            'u_lip': ['Big_Lips', 'Wearing_Lipstick', 'Mouth_Slightly_Open', 'Smiling'],
            'hair': ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Bald', 'Wavy_Hair', 'Straight_Hair', 'Bangs', 'Receding_Hairline', 'Sideburns'],
            'l_eye': ['Narrow_Eyes'],
            'r_eye': ['Narrow_Eyes'],
            'neck': ['Double_Chin', 'Wearing_Necklace', 'Wearing_Necktie'],
            'l_brow': ['Arched_Eyebrows', 'Bushy_Eyebrows'],
            'r_brow': ['Arched_Eyebrows', 'Bushy_Eyebrows']
        }

        masks_to_associated_concepts_indices = {}
        for mask, concepts in masks_to_associated_concepts.items():
            indices = [all_concepts.index(concept) for concept in concepts]
            masks_to_associated_concepts_indices[mask] = indices

        task_indices = [all_concepts.index(concept) for concept in task_concepts]

        num_concepts = len(all_concepts)
        num_prototypes = 40  # arbitrary number, can be tuned
        num_tasks = max(len(task_concepts), 1)
        num_objects = len(used_masks)

        train_loader, val_loader, test_loader = get_celeba_dataset(batch_size=batch_size, used_masks=used_masks, concepts_for_masks=masks_to_associated_concepts_indices, task_indices=task_indices, noisy_part=noisy_part_index, 
                                                                   noisy_target_part=noisy_target_part_index, noisy_prob=noisy_prob, device=DEVICE, num_workers=num_workers)

        res = 128
        resolution = (res, res)
        patch_size = 128

    elif dataset == "clevrhans":
        all_concepts = CLEVR_HANS_ALL_CONCEPTS
        num_concepts = CLEVR_HANS_NUM_CONCEPTS
        num_tasks = CLEVR_HANS_NUM_CLASSES
        num_objects = CLEVR_HANS_MAX_OBJECTS
        num_prototypes = 50

        clevrhans_resolution = getattr(args, 'clevrhans_resolution', 128)

        # Skip slow raw loading if we are going to use presegmented datasets anyway
        if not use_pretrained_segmenter:
            train_loader, val_loader, test_loader = get_clevrhans_dataset(
                batch_size=batch_size,
                resolution=clevrhans_resolution,
                device=DEVICE,
                num_workers=num_workers,
                noisy_concept_names=noisy_concept_names,
                target_noisy_concept_names=target_noisy_concept_names,
                noisy_prob=noisy_prob,
            )
        else:
            print(f"[CLEVR-Hans3] Skipping raw loading as use_pretrained_segmenter=True. Loaders will be loaded from {presegmented_datasets_path}")
            train_loader, val_loader, test_loader = None, None, None

        resolution = (clevrhans_resolution, clevrhans_resolution)
        patch_size = clevrhans_resolution

    elif dataset == "cub_presegm":
        # ── CUB-200-2011 Pre-Segmented (6 bird parts) ──────────────────────────
        cub_presegm_dir = getattr(args, 'cub_presegm_data_path', CUB_PRESEGM_DIR)
        cub_val_split = getattr(args, 'cub_presegm_val_split', 0.1)

        # Parse attribute names from attributes.txt
        all_concepts = _parse_cub_attribute_names()
        num_concepts = len(all_concepts)  # 312

        # Part selection (overridable via config)
        used_parts = getattr(args, 'cub_presegm_used_parts', None) or CUB_PRESEGM_PARTS

        # Build concept-to-part mapping
        prefix_to_part = getattr(args, 'cub_presegm_attribute_prefix_to_part', None)
        part_to_attribute_indices = _build_cub_part_to_attribute_indices(
            all_concepts, prefix_to_part=prefix_to_part, used_parts=used_parts,
        )
        print(f"[CUB-presegm] Part → attribute counts: {{{', '.join(f'{p}: {len(v)}' for p, v in part_to_attribute_indices.items())}}}")

        num_tasks = CUB_PRESEGM_NUM_CLASSES  # 200
        num_objects = len(used_parts)  # 6
        num_prototypes = 100

        # Load presegmented data (handles val split creation internally).
        # PGCM expects 5-tuples; non-PGCM competitors expect 4-tuples.
        if args.model == "PGCM":
            train_loader, val_loader, test_loader = load_cub_presegmented_dataloaders(
                presegmented_dir=cub_presegm_dir,
                batch_size=batch_size,
                part_to_attribute_indices=part_to_attribute_indices,
                used_parts=used_parts,
                val_split=cub_val_split,
                device=DEVICE,
                num_workers=num_workers,
            )
        else:
            train_loader, val_loader, test_loader = load_cub_presegmented_dataloaders_nonpgcm(
                presegmented_dir=cub_presegm_dir,
                batch_size=batch_size,
                part_to_attribute_indices=part_to_attribute_indices,
                used_parts=used_parts,
                val_split=cub_val_split,
                device=DEVICE,
                num_workers=num_workers,
            )

        res = 128
        resolution = (res, res)
        patch_size = res

        # Override: data is already pre-segmented
        use_pretrained_segmenter = True
        use_pretrained_autoencoder = False

    elif dataset == "cubEMB":
        train_loader, val_loader, test_loader, num_concepts, num_tasks, resolution = get_cub_dataset(
            batch_size=batch_size,
            dataset_root=getattr(args, "cub_dataset_root", None),
            device=DEVICE,
            num_workers=num_workers,
        )
        num_objects = 1
        num_prototypes = 40
        all_concepts = [f"concept_{i}" for i in range(num_concepts)]
        patch_size = resolution[0] if resolution is not None else None
        use_pretrained_segmenter = False
        presegmented_datasets_path = None
        use_pretrained_autoencoder = False

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")

    autoencoder_encoder_path = None
    autoencoder_decoder_path = None
    if use_pretrained_autoencoder:
        ae_path = getattr(args, "autoencoder_path", None)
        if ae_path is not None:
            # If path is provided in config, use it. Check for "outputs" subfolder as per train_autoencoder script
            if os.path.exists(os.path.join(ae_path, "outputs")):
                autoencoder_encoder_path = os.path.join(ae_path, "outputs", "encoder_state_dict.pt")
                autoencoder_decoder_path = os.path.join(ae_path, "outputs", "decoder_state_dict.pt")
            else:
                autoencoder_encoder_path = os.path.join(ae_path, "encoder_state_dict.pt")
                autoencoder_decoder_path = os.path.join(ae_path, "decoder_state_dict.pt")
        else:
            # Fallback to hardcoded paths for backward compatibility / legacy runs
            if dataset == "mnist":
                ae_path = "/cw/dtaijupiter/NoCsBack/dtai/david/prototypes/HigherOrderCBMs/outputs/autoencoder_mnist_20260211_111350"
                autoencoder_encoder_path = f"{ae_path}/encoder_state_dict.pt"
                autoencoder_decoder_path = f"{ae_path}/decoder_state_dict.pt"
            elif dataset == "celebamask":
                ae_path = "outputs/autoencoder_celebamask_YYYYMMDD_HHMMSS"
                autoencoder_encoder_path = f"{ae_path}/encoder_state_dict.pt"
                autoencoder_decoder_path = f"{ae_path}/decoder_state_dict.pt"
            else:
                raise NotImplementedError(f"Dataset {dataset} not implemented for pretrained autoencoder.")
    
    if use_pretrained_segmenter and dataset != "cub_presegm":
        preseg_noisy_concept_names = noisy_concept_names if dataset == "clevrhans" else None
        preseg_target_noisy_concept_names = target_noisy_concept_names if dataset == "clevrhans" else None
        preseg_concept_names = all_concepts if dataset == "clevrhans" else None
        train_loader, val_loader, test_loader = load_presegmented_dataloaders(
            presegmented_datasets_path=presegmented_datasets_path,
            batch_size=batch_size,
            device=DEVICE,
            num_workers=num_workers,
            noisy_concept_names=preseg_noisy_concept_names,
            target_noisy_concept_names=preseg_target_noisy_concept_names,
            noisy_prob=noisy_prob,
            concept_names=preseg_concept_names,
        )

    pos_weights = None
    if use_weights:
        pos_weights, num_concepts_positives, num_concepts_negatives = get_concept_balance(train_loader, num_concepts, num_objects, DEVICE)

    use_balanced_accuracy = getattr(args, "use_balanced_accuracy", True)
    use_linear_task_predictor = getattr(args, "use_linear_task_predictor", False)


    # if we have nb proto arg, override the dataset's default
    if nb_proto_override is not None:
        num_prototypes = nb_proto_override

    if args.model == "PGCM":
        model = Model(
            nb_proto=num_prototypes, 
            proto_size=proto_size, 
            embedding_size=embedding_size, 
            nb_concepts=num_concepts, 
            nb_tasks=num_tasks, 
            lr=lr, 
            reconstruction = True, 
            lam=lam_entropy, 
            lam_batch_entropy=lam_batch_entropy,
            decay_lam_entropy=decay_lam_entropy,
            nb_possibleobjects=num_objects, 
            lam_reconstruction=lam_reconstruction, 
            segmentation_method=segmentation_method,
            resolution=resolution,
            patch_size=patch_size if dataset in ('celebamask', 'clevrhans', 'cubEMB', 'cub_presegm') else None,
            lam_kl=lam_kl,
            lam_segmentation=lam_segmentation,
            lam_orth=lam_orth,
            concept_names=all_concepts,
            val_loader=val_loader,
            train_loader=train_loader,
            output_folder=output_folder,
            plot_frequency=plot_frequency,
            concepts_to_task=concepts_to_task,
            batch_size=batch_size,
            dataset_name=dataset,
            map_to_train_set=True,
            pos_weights=pos_weights,
            pgcm=True,
            warmup_epochs=warmup_epochs,
            intv_prob=intv_prob,
            autoencoder_encoder_path=autoencoder_encoder_path,
            autoencoder_decoder_path=autoencoder_decoder_path,
            use_balanced_accuracy=use_balanced_accuracy,
            use_pretrained_autoencoder=use_pretrained_autoencoder,
            use_pretrained_segmenter=use_pretrained_segmenter,
            use_linear_task_predictor=use_linear_task_predictor,
        )
        model.FIXED_LR = True if fixed_lr else False   # TODO
        model.ALWAYS_USE_TRUE_MASKS = True if always_use_true_masks else False  # TODO
        model.USE_INITIAL_PROTO_EMBS = True if use_initial_proto_embs else False
    elif args.model == "CBM":
        model = CBMDeep(
            emb_size=embedding_size,
            n_tasks=num_tasks,
            n_concepts=num_concepts,
            concept_names=all_concepts,
            task_names=[f"task_{i}" for i in range(num_tasks)],
            lr=lr,
            concepts_to_task=concepts_to_task,
            dataset=dataset,
            nb_possibleobjects=num_objects,
            task_weight=1.0,
            plot_frequency=plot_frequency,
            val_loader=val_loader,
            output_folder=output_folder,
            pos_weights=pos_weights,
            warmup_epochs=warmup_epochs,
            intv_prob=intv_prob,
            use_balanced_accuracy=use_balanced_accuracy,
            use_linear_task_predictor=use_linear_task_predictor,
        )
    elif args.model == "CRM":
        model = CRM(
            emb_size=embedding_size,
            n_tasks=num_tasks,
            n_concepts=num_concepts,
            concept_names=all_concepts,
            task_names=[f"task_{i}" for i in range(num_tasks)],
            lr=lr,
            concepts_to_task=concepts_to_task,
            dataset=dataset,
            nb_possibleobjects=num_objects,
            task_weight=1.0,
            plot_frequency=plot_frequency,
            val_loader=val_loader,
            output_folder=output_folder,
            pos_weights=pos_weights,
            warmup_epochs=warmup_epochs,
            use_balanced_accuracy=use_balanced_accuracy,
            use_linear_task_predictor=use_linear_task_predictor,
        )

    elif args.model == "CMR":
        model = CMR(
            emb_size=embedding_size,
            n_tasks=num_tasks,
            n_concepts=num_concepts,
            concept_names=all_concepts,
            task_names=[f"task_{i}" for i in range(num_tasks)],
            lr=lr,
            concepts_to_task=concepts_to_task,
            dataset=dataset,
            nb_possibleobjects=num_objects,
            task_weight=1.0,
            plot_frequency=plot_frequency,
            val_loader=val_loader,
            output_folder=output_folder,
            pos_weights=pos_weights,
            rule_emb_size=rule_emb_size,
            warmup_epochs=warmup_epochs,
            intv_prob=intv_prob,
            use_balanced_accuracy=use_balanced_accuracy,
            use_linear_task_predictor=use_linear_task_predictor,
        )
    elif args.model == "DNN":
        model = DNN(
            emb_size=embedding_size,
            n_tasks=num_tasks,
            task_names=[f"task_{i}" for i in range(num_tasks)],
            lr=lr,
            dataset=dataset,
            nb_possibleobjects=num_objects,
            n_concepts=num_concepts,
            plot_frequency=plot_frequency,
            val_loader=val_loader,
            output_folder=output_folder,
            pos_weights=pos_weights,
            use_balanced_accuracy=use_balanced_accuracy,
        )
    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")

    model.to(DEVICE)

    # logger
    log_name = f"model_{dataset}_segm_{segmentation_method}_lr_{lr}_lament_{lam_entropy}_lamrec_{lam_reconstruction}_seed_{args.seed}_{extra_string}"
    if test_only:
        log_name += "_TESTING"
    
    # Save wandb logs to new_outputs/<timestamp>_<extra>/wandb
    # You can change the 'entity' argument below to your team name (e.g. entity="my-team-name")
    run_name = f"{timestamp}_{dataset}_{args.extra}" if args.extra != "" else f"{timestamp}_{dataset}"
    logger = WandbLogger(project="HigherOrderCBMs", entity="conceptlords", save_dir=base_output_folder, name=run_name)
    log_folder = os.path.join(base_output_folder, "checkpoints")

    # collect all hyperparameters and dataset data for logging
    logger.log_hyperparams({
        "device": DEVICE,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "lam_entropy": lam_entropy,
        "lam_reconstruction": lam_reconstruction,
        "dataset": dataset,
        "segmentation_method": segmentation_method,
        "proto_size": proto_size,
        "embedding_size": embedding_size,
        "seed": args.seed,
        "num_prototypes": num_prototypes,
        "num_concepts": num_concepts,
        "num_tasks": num_tasks,
        "resolution": resolution,
        "patch_size": patch_size if dataset in ('celebamask', 'cub', 'cub_presegm') else None,
        "concepts_to_task": concepts_to_task,
        "model": args.model,
        "use_pretrained_segmenter": use_pretrained_segmenter,
        "presegmented_datasets_path": presegmented_datasets_path,
        "num_workers": num_workers,
    })

    model._logger = logger

    if args.checkpoint_path is not None:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        # checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        # model.load_state_dict(checkpoint['state_dict'])
        # print("Model loaded.")
        print(f"Restoring best model from: {checkpoint_path}")  # pass val_loader and train_loader
        model = type(model).load_from_checkpoint(checkpoint_path, val_loader=val_loader, train_loader=train_loader)
        model.to(DEVICE)
                # do a forward pass on the train_loader
        for batch in train_loader:
            batch = tuple(b.to(DEVICE) if torch.is_tensor(b) else b for b in batch)
            model.training_step(batch, 0)
            break

    if args.model == "PGCM":
        # checkpoint saver callback for val loss before prototype swapping
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_total_loss_before_swap",
            dirpath=log_folder,
            filename="best-checkpoint-before-swap",
            save_top_k=1,
            mode="min",
        )

        checkpoint_callback_after_swap = pl.callbacks.ModelCheckpoint(
            monitor="val_total_loss_after_swap" if dataset != "cubEMB" else "val_task_acc",
            dirpath=log_folder,
            filename="best-checkpoint-after-swap",
            save_top_k=1,
            mode="min" if dataset != "cubEMB" else "max",
        )
        selected_callbacks = [checkpoint_callback, checkpoint_callback_after_swap]
    else:
        # checkpoint saver callback for val loss
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_total_loss" if dataset != "cubEMB" else "val_task_acc",
            dirpath=log_folder,
            filename="best-checkpoint",
            save_top_k=1,
            mode="min" if dataset != "cubEMB" else "max",
        )

        selected_callbacks = [checkpoint_callback]
        if not disable_early_stopping:
            early_stopping_callback = pl.callbacks.EarlyStopping(
                monitor="val_total_loss" if dataset != "cubEMB" else "val_task_acc",
                patience=20 if args.model != "PGCM" else 1000,
                mode="min" if dataset != "cubEMB" else "max"
            )
            selected_callbacks.append(early_stopping_callback)

    if DEVICE == "cpu":
        trainer = pl.Trainer(accelerator='cpu', max_epochs=epochs, logger=logger, callbacks=selected_callbacks)
    else:
        trainer = pl.Trainer(accelerator='gpu', devices=[int(DEVICE[-1])], max_epochs=epochs, logger=logger, callbacks=selected_callbacks)
    if not test_only:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # restore weights using callback
        if args.model == "PGCM":
            best_model_path = checkpoint_callback_after_swap.best_model_path
        else:
            best_model_path = checkpoint_callback.best_model_path
        print(f"Restoring best model from: {best_model_path}")  # pass val_loader and train_loader
        model = type(model).load_from_checkpoint(best_model_path, val_loader=val_loader, train_loader=train_loader)
        model.to(DEVICE)

        # print epoch number of best model
        print(f"Best model epoch: {trainer.current_epoch}")

        # do a forward pass on the train_loader
        for batch in train_loader:
            batch = tuple(b.to(DEVICE) if torch.is_tensor(b) else b for b in batch)
            model.training_step(batch, 0)
            break

    model.to(DEVICE)
    model.eval()

    # For the MNIST noise experiments, c_indices marks the concept indices that
    # should be treated as noisy when computing concept-level metrics.
    c_indices = None
    if dataset == "mnist" and noisy_digit is not None:
        c_indices = torch.tensor([*noisy_digit, *noisy_target_digit])
        c_indices = torch.tensor(noisy_target_digit)

    if segmentation_method == 'true' or segmentation_method == 'mask':
        val_y_acc, val_c_acc = compute_balanced_accuracy_loader(model, val_loader, c_indices=c_indices)
        train_y_acc, train_c_acc = compute_balanced_accuracy_loader(model, train_loader, c_indices=c_indices)
        test_y_acc, test_c_acc = compute_balanced_accuracy_loader(model, test_loader, c_indices=c_indices)
    elif segmentation_method in ['slot_attention']:
        val_y_acc, val_c_acc = compute_acc_unordered(model, val_loader, num_objects=2)
        train_y_acc, train_c_acc = compute_acc_unordered(model, train_loader, num_objects=2)

    print(f"Train y acc: {train_y_acc}, c acc: {train_c_acc}")
    print(f"Val y acc: {val_y_acc}, c acc: {val_c_acc}")
    print(f"Test y acc: {test_y_acc}, c acc: {test_c_acc}")

    logger.log_metrics({
        "final_train_y_acc": train_y_acc,
        "final_train_c_acc": train_c_acc,
        "final_val_y_acc": val_y_acc,
        "final_val_c_acc": val_c_acc,
        "final_test_y_acc": test_y_acc,
        "final_test_c_acc": test_c_acc
    })

    if isinstance(model, Model) and dataset != "cub":
        model.generate_plots(logger=logger, output_folder=output_folder)

    _plot_intervenability(model, logger=logger, nb_runs=1, specific_loader=test_loader, specific_epoch_nb="final_test", is_pgcm=args.model=="PGCM")

    if dataset == "celebamask" and args.model == "CBM":
        inspect_val_predictions_with_pause(
            model=model,
            val_loader=val_loader,
            concept_names=all_concepts,
            output_folder=output_folder,
            device=DEVICE,
        )

    if noisy_digit is not None or noisy_part is not None:
        # prompt for removed prototypes, a list
        masked_prototypes = []
        inp = input("Enter prototype indices to mask, separated by commas (or 'none' to skip): ")
        if inp.lower() != 'none':

            if False:
                masked_prototypes = [int(x.strip()) for x in inp.split(',')]
                model.update_masked_prototypes(masked_prototypes)
                print(f"Masked prototypes: {masked_prototypes}")
            else:
                # ask for true digits for each of these
                inp2 = input("Enter the true concepts for each masked prototype as digit1,digit2,... (e.g., '3,2'): ")
                true_concepts = [int(x.strip()) for x in inp2.split(',')]
                prototypes_to_intervene = [int(x.strip()) for x in inp.split(',')]
                # forced_concepts_per_proto: shape (nb_proto, nb_concepts)
                for nb_interventions in range(1, len(true_concepts) + 1):
                    forced_concepts_per_proto = torch.zeros((model.nb_proto, model.nb_concepts), device=DEVICE)
                    forced_concepts_per_proto_mask = torch.zeros((model.nb_proto, model.nb_concepts), device=DEVICE)
                    # set the corresponding prototpyes' concepts
                    for i in range(nb_interventions):
                        proto_idx = prototypes_to_intervene[i]
                        concept_idx = true_concepts[i]
                        forced_concepts_per_proto[proto_idx, concept_idx] = 1.0
                        for j in range(model.nb_concepts):
                            if j != concept_idx:
                                forced_concepts_per_proto[proto_idx, j] = 0.0
                        forced_concepts_per_proto_mask[proto_idx] = 1.0
                    model.set_forced_concepts_per_proto(forced_concepts_per_proto, forced_concepts_per_proto_mask)

                    # compute accs
                    if segmentation_method == 'true' or segmentation_method == 'mask':
                        test_y_acc, test_c_acc = compute_balanced_accuracy_loader(model, test_loader, c_indices=c_indices)
                    elif segmentation_method in ['slot_attention']:
                        test_y_acc, test_c_acc = compute_acc_unordered(model, test_loader, num_objects=2)
                    print(f"After forcing {nb_interventions} interventions per prototype:")
                    print(f"Test y acc: {test_y_acc}, c acc: {test_c_acc}")

        # re-evaluate on test set
        if segmentation_method == 'true' or segmentation_method == 'mask':
            test_y_acc, test_c_acc = compute_balanced_accuracy_loader(model, test_loader, c_indices=c_indices)
        elif segmentation_method in ['slot_attention']:
            test_y_acc, test_c_acc = compute_acc_unordered(model, test_loader, num_objects=2)
        
        print(f"After masking prototypes:")
        print(f"Test y acc: {test_y_acc}, c acc: {test_c_acc}")

        logger.log_metrics({
            "final_test_y_acc_after_masking": test_y_acc,
            "final_test_c_acc_after_masking": test_c_acc
        })



if __name__ == "__main__":
    args = get_args()
    main(args)

import argparse
import os
from collections import defaultdict
from datetime import datetime

import torch
import yaml

from dataset import (
    CLEVR_HANS_ALL_CONCEPTS,
    CLEVR_HANS_MAX_OBJECTS,
    CLEVR_HANS_NUM_CLASSES,
    CLEVR_HANS_NUM_CONCEPTS,
    get_celeba_dataset,
    get_clevrhans_dataset,
    get_cub_dataset,
    get_mnist,
    load_presegmented_dataloaders,
)
from model import Model
from utils import compute_balanced_accuracy_loader, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive PGCM checkpoint editor with delete/edit prototype interventions."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config used for the checkpoint run.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to PGCM checkpoint (.ckpt).")
    parser.add_argument("--mode", type=str, required=True, choices=["delete", "edit"], help="Intervention mode.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Data split used for evaluation.")
    parser.add_argument("--device", type=str, default=None, help="Optional override for device, e.g. cuda:0.")
    return parser.parse_args()


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return argparse.Namespace(**cfg)


def _build_celeba_metadata_and_loaders(batch_size, device, num_workers):
    used_masks = ["skin", "hair", "nose", "lips"]
    all_concepts = "5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young".split(", ")
    task_concepts = ["Attractive", "Male", "Young"]

    masks_to_associated_concepts = {
        "skin": ["Pale_Skin", "Rosy_Cheeks", "Heavy_Makeup", "Chubby", "5_o_Clock_Shadow", "Bags_Under_Eyes", "Eyeglasses", "Goatee", "High_Cheekbones", "Mustache", "No_Beard", "Oval_Face"],
        "nose": ["Big_Nose", "Pointy_Nose"],
        "l_lip": ["Big_Lips", "Wearing_Lipstick", "Mouth_Slightly_Open", "Smiling"],
        "u_lip": ["Big_Lips", "Wearing_Lipstick", "Mouth_Slightly_Open", "Smiling"],
        "hair": ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Bald", "Wavy_Hair", "Straight_Hair", "Bangs", "Receding_Hairline", "Sideburns"],
        "l_eye": ["Narrow_Eyes"],
        "r_eye": ["Narrow_Eyes"],
        "neck": ["Double_Chin", "Wearing_Necklace", "Wearing_Necktie"],
        "l_brow": ["Arched_Eyebrows", "Bushy_Eyebrows"],
        "r_brow": ["Arched_Eyebrows", "Bushy_Eyebrows"],
    }

    masks_to_associated_concepts_indices = {
        mask: [all_concepts.index(concept) for concept in concepts]
        for mask, concepts in masks_to_associated_concepts.items()
    }
    task_indices = [all_concepts.index(concept) for concept in task_concepts]

    train_loader, val_loader, test_loader = get_celeba_dataset(
        batch_size=batch_size,
        used_masks=used_masks,
        concepts_for_masks=masks_to_associated_concepts_indices,
        task_indices=task_indices,
        device=device,
        num_workers=num_workers,
    )

    metadata = {
        "all_concepts": all_concepts,
        "num_concepts": len(all_concepts),
        "num_tasks": max(len(task_concepts), 1),
        "num_objects": len(used_masks),
        "num_prototypes": 40,
        "resolution": (128, 128),
        "patch_size": 128,
    }
    return (train_loader, val_loader, test_loader), metadata


def _build_data_and_metadata(args):
    dataset = args.dataset
    batch_size = args.batch_size
    device = args.device
    num_workers = getattr(args, "num_workers", 0)
    use_pretrained_segmenter = getattr(args, "use_pretrained_segmenter", False)
    presegmented_datasets_path = getattr(args, "presegmented_datasets_path", None)

    if dataset == "mnist":
        allowed_colors = ["red", "blue", "green"]
        allowed_digits = list(range(10))
        color_as_concepts = False
        extra_prototypes = 20

        all_concepts = [str(x) for x in allowed_digits]
        num_concepts = len(all_concepts)
        num_tasks = (len(allowed_digits) * 2) - 1
        num_objects = 2
        num_prototypes = len(allowed_digits) + extra_prototypes
        resolution = (56, 28)
        patch_size = None

        noisy_digit = getattr(args, "noisy_digit", None)
        noisy_target_digit = getattr(args, "noisy_target_digit", None)
        noisy_prob = getattr(args, "noisy_prob", 0.0)

        train_loader, val_loader, test_loader = get_mnist(
            batch_size=batch_size,
            allowed_digits=allowed_digits,
            allowed_colors=allowed_colors,
            color_as_concepts=color_as_concepts,
            noisy_digit=noisy_digit,
            noisy_target_digit=noisy_target_digit,
            noisy_prob=noisy_prob,
            device=device,
            num_workers=num_workers,
        )

    elif dataset == "celebamask":
        (train_loader, val_loader, test_loader), meta = _build_celeba_metadata_and_loaders(batch_size, device, num_workers)
        all_concepts = meta["all_concepts"]
        num_concepts = meta["num_concepts"]
        num_tasks = meta["num_tasks"]
        num_objects = meta["num_objects"]
        num_prototypes = meta["num_prototypes"]
        resolution = meta["resolution"]
        patch_size = meta["patch_size"]

    elif dataset == "clevrhans":
        all_concepts = CLEVR_HANS_ALL_CONCEPTS
        num_concepts = CLEVR_HANS_NUM_CONCEPTS
        num_tasks = CLEVR_HANS_NUM_CLASSES
        num_objects = CLEVR_HANS_MAX_OBJECTS
        num_prototypes = 50

        clevrhans_resolution = getattr(args, "clevrhans_resolution", 128)
        noisy_concept_names = getattr(args, "noisy_concept_names", None)
        target_noisy_concept_names = getattr(args, "target_noisy_concept_names", None)
        noisy_prob = getattr(args, "noisy_prob", 0.0)

        if use_pretrained_segmenter:
            train_loader, val_loader, test_loader = load_presegmented_dataloaders(
                presegmented_datasets_path=presegmented_datasets_path,
                batch_size=batch_size,
                device=device,
                num_workers=num_workers,
                noisy_concept_names=noisy_concept_names,
                target_noisy_concept_names=target_noisy_concept_names,
                noisy_prob=noisy_prob,
                concept_names=all_concepts,
            )
        else:
            train_loader, val_loader, test_loader = get_clevrhans_dataset(
                batch_size=batch_size,
                resolution=clevrhans_resolution,
                device=device,
                num_workers=num_workers,
                noisy_concept_names=noisy_concept_names,
                target_noisy_concept_names=target_noisy_concept_names,
                noisy_prob=noisy_prob,
            )

        resolution = (clevrhans_resolution, clevrhans_resolution)
        patch_size = clevrhans_resolution

    elif dataset == "cubEMB":
        train_loader, val_loader, test_loader, num_concepts, num_tasks, resolution = get_cub_dataset(
            batch_size=batch_size,
            dataset_root=getattr(args, "cub_dataset_root", None),
            device=device,
            num_workers=num_workers,
        )
        num_objects = 1
        num_prototypes = 40
        patch_size = resolution[0] if resolution is not None else None
        all_concepts = [f"concept_{i}" for i in range(num_concepts)]

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")

    nb_proto_override = getattr(args, "nb_proto", None)
    if nb_proto_override is not None:
        num_prototypes = nb_proto_override

    metadata = {
        "all_concepts": all_concepts,
        "num_concepts": num_concepts,
        "num_tasks": num_tasks,
        "num_objects": num_objects,
        "num_prototypes": num_prototypes,
        "resolution": resolution,
        "patch_size": patch_size,
    }

    return (train_loader, val_loader, test_loader), metadata


def _resolve_autoencoder_paths(args):
    use_pretrained_autoencoder = getattr(args, "use_pretrained_autoencoder", False)
    if not use_pretrained_autoencoder:
        return None, None

    ae_path = getattr(args, "autoencoder_path", None)
    if ae_path is None:
        return None, None

    if os.path.exists(os.path.join(ae_path, "outputs")):
        encoder_path = os.path.join(ae_path, "outputs", "encoder_state_dict.pt")
        decoder_path = os.path.join(ae_path, "outputs", "decoder_state_dict.pt")
    else:
        encoder_path = os.path.join(ae_path, "encoder_state_dict.pt")
        decoder_path = os.path.join(ae_path, "decoder_state_dict.pt")

    return encoder_path, decoder_path


def _build_model(args, metadata, train_loader, val_loader):
    if getattr(args, "model", "PGCM") != "PGCM":
        raise ValueError("This script only supports PGCM checkpoints.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join("new_outputs", f"{timestamp}_{args.dataset}_interactive", "outputs")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "prototypes"), exist_ok=True)

    proto_size = getattr(args, "protosize", 64)
    embedding_size = getattr(args, "embedding_size")
    lr = getattr(args, "lr")

    autoencoder_encoder_path, autoencoder_decoder_path = _resolve_autoencoder_paths(args)
    use_balanced_accuracy = True

    model = Model(
        nb_proto=metadata["num_prototypes"],
        proto_size=proto_size,
        embedding_size=embedding_size,
        nb_concepts=metadata["num_concepts"],
        nb_tasks=metadata["num_tasks"],
        lr=lr,
        reconstruction=True,
        lam=getattr(args, "lam_entropy", 0.0),
        lam_batch_entropy=getattr(args, "lam_batch_entropy", 0.0),
        decay_lam_entropy=getattr(args, "decay_lam_entropy", True),
        nb_possibleobjects=metadata["num_objects"],
        lam_reconstruction=getattr(args, "lam_reconstruction", 0.0),
        segmentation_method=getattr(args, "segmentation_method", "true"),
        resolution=metadata["resolution"],
        patch_size=metadata["patch_size"],
        lam_kl=getattr(args, "lam_kl", 0.0),
        lam_segmentation=getattr(args, "lam_segmentation", 1.0),
        lam_orth=getattr(args, "lam_orth", 0.0),
        concept_names=metadata["all_concepts"],
        val_loader=val_loader,
        train_loader=train_loader,
        output_folder=output_folder,
        plot_frequency=getattr(args, "plot_frequency", 10),
        concepts_to_task=getattr(args, "concepts_to_task", "thresholding"),
        batch_size=getattr(args, "batch_size"),
        dataset_name=getattr(args, "dataset"),
        map_to_train_set=True,
        pos_weights=None,
        pgcm=True,
        warmup_epochs=getattr(args, "warmup_epochs", 0),
        intv_prob=getattr(args, "intv_prob", 0.0),
        autoencoder_encoder_path=autoencoder_encoder_path,
        autoencoder_decoder_path=autoencoder_decoder_path,
        use_balanced_accuracy=use_balanced_accuracy,
        use_pretrained_autoencoder=getattr(args, "use_pretrained_autoencoder", False),
        use_pretrained_segmenter=getattr(args, "use_pretrained_segmenter", False),
    )

    model.FIXED_LR = bool(getattr(args, "fixed_lr", False))
    model.ALWAYS_USE_TRUE_MASKS = bool(getattr(args, "always_use_true_masks", False))
    model.USE_INITIAL_PROTO_EMBS = bool(getattr(args, "use_initial_proto_embs", False))

    return model


def _evaluate_and_print(model, loader):
    y_acc, c_acc = compute_balanced_accuracy_loader(model, loader)
    print(f"Task balanced accuracy: {y_acc:.6f}")
    print(f"Concept balanced accuracy: {c_acc:.6f}")


def _evaluate_and_print_with_filter(model, loader, c_indices=None):
    y_acc, c_acc = compute_balanced_accuracy_loader(model, loader, c_indices=c_indices)
    print(f"Task balanced accuracy: {y_acc:.6f}")
    print(f"Concept balanced accuracy: {c_acc:.6f}")
    return y_acc, c_acc


def _unique_target_concept_indices(concept_names, target_noisy_concept_names):
    if not target_noisy_concept_names:
        return None

    unique_names = list(dict.fromkeys(target_noisy_concept_names))
    return [concept_names.index(name) for name in unique_names]


def _concept_families(concept_names):
    family_to_indices = defaultdict(list)
    for idx, name in enumerate(concept_names):
        if "_" in name:
            family = name.split("_", 1)[0]
            family_to_indices[family].append(idx)
    return dict(family_to_indices)


def _clear_forced_concepts(model):
    model.forced_concepts_per_proto = None
    model.forced_concepts_per_proto_mask = None


def _apply_delete_history(model, history):
    model.update_masked_prototypes(history.copy())


def _apply_edit_history(model, history, concept_names, family_to_indices):
    if not history:
        _clear_forced_concepts(model)
        return

    forced_values = torch.zeros((model.nb_proto, model.nb_concepts), device=model.device)
    forced_mask = torch.zeros((model.nb_proto, model.nb_concepts), device=model.device)

    for proto_idx, concept_idx in history:
        concept_name = concept_names[concept_idx]
        if "_" in concept_name:
            family = concept_name.split("_", 1)[0]
            family_indices = family_to_indices.get(family, [concept_idx])
        else:
            family_indices = [concept_idx]

        forced_mask[proto_idx, family_indices] = 1.0
        forced_values[proto_idx, family_indices] = 0.0
        forced_values[proto_idx, concept_idx] = 1.0

    model.set_forced_concepts_per_proto(forced_values, forced_mask)


def _run_delete_loop(model, loader, c_indices=None):
    history = []
    print("Delete mode started. Enter prototype index, UNDO, or QUIT.")

    while True:
        user_input = input("Delete> ").strip()
        cmd = user_input.upper()

        if cmd in {"QUIT", "EXIT", "Q"}:
            print("Exiting delete mode.")
            break

        if cmd == "UNDO":
            if history:
                removed = history.pop()
                _apply_delete_history(model, history)
                print(f"Removed latest masking action: prototype {removed}")
            else:
                print("Nothing to undo.")

            _evaluate_and_print_with_filter(model, loader, c_indices=c_indices)
            continue

        try:
            proto_idx = int(user_input)
        except ValueError:
            print("Invalid input. Enter an integer prototype index, UNDO, or QUIT.")
            continue

        if proto_idx < 0 or proto_idx >= model.nb_proto:
            print(f"Prototype index out of range. Valid range: [0, {model.nb_proto - 1}]")
            continue

        history.append(proto_idx)
        _apply_delete_history(model, history)
        print(f"Masked prototypes (history): {history}")
        _evaluate_and_print_with_filter(model, loader, c_indices=c_indices)


def _run_edit_loop(model, loader, c_indices=None):
    concept_names = list(model.concept_names)
    concept_to_idx = {name: idx for idx, name in enumerate(concept_names)}
    family_to_indices = _concept_families(concept_names)
    history = []

    families_str = ", ".join(sorted(family_to_indices.keys())) if family_to_indices else "none"
    print("Edit mode started. Enter prototype index, UNDO, or QUIT.")
    print(f"Detected concept families: {families_str}")
    print("Type LIST at concept prompt to print all concept names.")

    while True:
        user_input = input("Edit prototype> ").strip()
        cmd = user_input.upper()

        if cmd in {"QUIT", "EXIT", "Q"}:
            print("Exiting edit mode.")
            break

        if cmd == "UNDO":
            if history:
                removed_proto, removed_concept_idx = history.pop()
                print(f"Removed latest edit action: prototype {removed_proto}, concept {concept_names[removed_concept_idx]}")
                _apply_edit_history(model, history, concept_names, family_to_indices)
            else:
                print("Nothing to undo.")

            _evaluate_and_print_with_filter(model, loader, c_indices=c_indices)
            continue

        try:
            proto_idx = int(user_input)
        except ValueError:
            print("Invalid prototype index. Enter an integer, UNDO, or QUIT.")
            continue

        if proto_idx < 0 or proto_idx >= model.nb_proto:
            print(f"Prototype index out of range. Valid range: [0, {model.nb_proto - 1}]")
            continue

        concept_name = input("Concept name> ").strip()
        if concept_name.upper() == "LIST":
            print("Available concepts:")
            print("\n".join(concept_names))
            concept_name = input("Concept name> ").strip()

        if concept_name not in concept_to_idx:
            print(f"Unknown concept name: {concept_name}")
            continue

        concept_idx = concept_to_idx[concept_name]
        history.append((proto_idx, concept_idx))
        _apply_edit_history(model, history, concept_names, family_to_indices)

        if "_" in concept_name:
            family = concept_name.split("_", 1)[0]
            family_members = [concept_names[i] for i in family_to_indices.get(family, [concept_idx])]
            print(
                "Applied edit: "
                f"prototype {proto_idx}, family '{family}' -> set {concept_name}=1 and "
                f"{[x for x in family_members if x != concept_name]}=0"
            )
        else:
            print(f"Applied edit: prototype {proto_idx}, concept {concept_name}=1")

        _evaluate_and_print_with_filter(model, loader, c_indices=c_indices)


def main():
    cli_args = parse_args()
    cfg = _load_config(cli_args.config)

    cfg.checkpoint_path = cli_args.checkpoint_path
    cfg.mode = cli_args.mode
    cfg.split = cli_args.split
    if cli_args.device is not None:
        cfg.device = cli_args.device
    if not hasattr(cfg, "device") or cfg.device is None:
        cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    set_seed(getattr(cfg, "seed", 42))

    (train_loader, val_loader, test_loader), metadata = _build_data_and_metadata(cfg)
    model = _build_model(cfg, metadata, train_loader, val_loader)

    model = type(model).load_from_checkpoint(
        cfg.checkpoint_path,
        val_loader=val_loader,
        train_loader=train_loader,
    )
    model.to(cfg.device)
    model.eval()
    model.use_balanced_accuracy = True

    target_noisy_concept_names = getattr(cfg, "target_noisy_concept_names", None)
    filtered_c_indices = _unique_target_concept_indices(metadata["all_concepts"], target_noisy_concept_names)

    if cfg.split == "train":
        eval_loader = train_loader
    elif cfg.split == "val":
        eval_loader = val_loader
    else:
        eval_loader = test_loader

    print("Loaded checkpoint and prepared evaluation loader.")
    print(f"Checkpoint: {cfg.checkpoint_path}")
    print(f"Dataset: {cfg.dataset} | Split: {cfg.split} | Mode: {cfg.mode}")
    print("Initial evaluation:")
    _evaluate_and_print(model, eval_loader)
    if filtered_c_indices is not None:
        print("Initial filtered evaluation (target_noisy_concept_names only):")
        _evaluate_and_print_with_filter(model, eval_loader, c_indices=filtered_c_indices)
    else:
        print("Initial filtered evaluation skipped: target_noisy_concept_names is not set.")

    if cfg.mode == "delete":
        _run_delete_loop(model, eval_loader, c_indices=filtered_c_indices)
    else:
        _run_edit_loop(model, eval_loader, c_indices=filtered_c_indices)


if __name__ == "__main__":
    main()
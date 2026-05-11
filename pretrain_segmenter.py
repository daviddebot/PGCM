import argparse
import os
from datetime import datetime

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from dataset import get_celeba_dataset, get_mnist, save_presegmented_datasets
from neural_networks import build_segmenter
from utils import set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--device", type=str, default="cpu", help="Which device to use for training.")
    parser.add_argument("--extra", type=str, default="", help="Extra string to append to the output folder name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    cmd_args = parser.parse_args()

    import yaml
    with open(cmd_args.config, 'r') as file:
        config_dict = yaml.safe_load(file)

    final_args = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(final_args, key, value)

    setattr(final_args, "config", cmd_args.config)
    setattr(final_args, "device", cmd_args.device)
    setattr(final_args, "extra", cmd_args.extra)
    setattr(final_args, "seed", cmd_args.seed)

    return final_args


def dice_loss(pred_logits, target_mask, smooth=1.0):
    pred_probs = torch.sigmoid(pred_logits)
    pred_flat = pred_probs.reshape(-1)
    target_flat = target_mask.reshape(-1)

    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def calc_segmentation_loss(pred_logits, gt_mask):
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask)
    dice = dice_loss(pred_logits, gt_mask)
    return bce + dice


class SegmenterPretrainer(pl.LightningModule):
    def __init__(self, dataset_name, segmentation_method, nb_objects, lr, warmup_epochs=10):
        super().__init__()
        self.save_hyperparameters()

        self.segmenter = build_segmenter(
            dataset_name=dataset_name,
            segmentation_method=segmentation_method,
            n_class=nb_objects,
        )
        self.lr = lr
        self.warmup_epochs = warmup_epochs

    def forward(self, x):
        return self.segmenter(x)

    def training_step(self, batch, batch_idx):
        x, m, _, _ = batch
        pred_logits = self.segmenter(x)
        loss = calc_segmentation_loss(pred_logits, m)

        self.log('train_segmentation_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, m, _, _ = batch
        pred_logits = self.segmenter(x)
        loss = calc_segmentation_loss(pred_logits, m)

        self.log('val_segmentation_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        total_epochs = self.trainer.max_epochs
        if total_epochs <= self.warmup_epochs:
            warmup_epochs = 0
        else:
            warmup_epochs = min(self.warmup_epochs, total_epochs)

        min_lr = 1e-5
        base_lr = self.lr

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs

            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress)))
            return min_lr / base_lr + (1 - min_lr / base_lr) * cosine_decay

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }


def get_dataset_loaders(args, device):
    dataset = args.dataset
    batch_size = args.batch_size
    num_workers = getattr(args, 'num_workers', 0)

    if dataset == 'mnist':
        allowed_colors = ['red', 'blue', 'green']
        allowed_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_loader, val_loader, test_loader = get_mnist(
            batch_size=batch_size,
            allowed_digits=allowed_digits,
            allowed_colors=allowed_colors,
            color_as_concepts=False,
            noisy_digit=getattr(args, 'noisy_digit', None),
            noisy_target_digit=getattr(args, 'noisy_target_digit', None),
            noisy_prob=getattr(args, 'noisy_prob', 0.2),
            device=device,
            num_workers=num_workers,
        )
        num_objects = 2
    elif dataset == 'celebamask':
        used_masks = ['skin', 'hair', 'nose', 'lips']
        all_concepts = '5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young'.split(', ')
        noisy_part_index = None
        noisy_target_part_index = None

        if getattr(args, 'noisy_part', None) is not None and getattr(args, 'noisy_target_part', None) is not None:
            noisy_part_index = all_concepts.index(args.noisy_part)
            noisy_target_part_index = all_concepts.index(args.noisy_target_part)

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
            'r_brow': ['Arched_Eyebrows', 'Bushy_Eyebrows'],
        }
        masks_to_associated_concepts_indices = {
            mask: [all_concepts.index(concept) for concept in concepts]
            for mask, concepts in masks_to_associated_concepts.items()
        }
        task_indices = [all_concepts.index(concept) for concept in ['Attractive', 'Male', 'Young']]

        train_loader, val_loader, test_loader = get_celeba_dataset(
            batch_size=batch_size,
            used_masks=used_masks,
            concepts_for_masks=masks_to_associated_concepts_indices,
            task_indices=task_indices,
            noisy_part=noisy_part_index,
            noisy_target_part=noisy_target_part_index,
            noisy_prob=getattr(args, 'noisy_prob', 0.2),
            device=device,
            num_workers=num_workers,
        )
        num_objects = len(used_masks)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")

    return train_loader, val_loader, test_loader, num_objects


def main(args):
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.extra}" if args.extra else ""
    base_output_folder = f"./new_outputs/segmenter_{timestamp}_{args.dataset}{suffix}"
    output_folder = os.path.join(base_output_folder, 'outputs')
    checkpoints_folder = os.path.join(base_output_folder, 'checkpoints')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(checkpoints_folder, exist_ok=True)

    with open(os.path.join(output_folder, 'args.txt'), 'w') as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")

    train_loader, val_loader, test_loader, num_objects = get_dataset_loaders(args, args.device)

    model = SegmenterPretrainer(
        dataset_name=args.dataset,
        segmentation_method=args.segmentation_method,
        nb_objects=num_objects,
        lr=args.lr,
        warmup_epochs=getattr(args, 'warmup_epochs', 10),
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_segmentation_loss',
        dirpath=checkpoints_folder,
        filename='best-segmenter',
        save_top_k=1,
        mode='min',
    )

    if args.device == 'cpu':
        trainer = pl.Trainer(accelerator='cpu', max_epochs=args.epochs, callbacks=[checkpoint_callback])
    else:
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[int(args.device[-1])],
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback],
        )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model_path = checkpoint_callback.best_model_path
    best_model = SegmenterPretrainer.load_from_checkpoint(best_model_path)
    best_model.to(args.device)

    segmenter_state_path = os.path.join(output_folder, 'segmenter_state_dict.pt')
    torch.save(best_model.segmenter.state_dict(), segmenter_state_path)

    presegmented_datasets_dir = os.path.join(output_folder, 'presegmented_datasets')
    save_presegmented_datasets(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        segmenter=best_model.segmenter,
        output_dir=presegmented_datasets_dir,
    )

    test_results = trainer.validate(best_model, dataloaders=test_loader, verbose=False)
    with open(os.path.join(output_folder, 'metrics.txt'), 'w') as file:
        file.write(f"best_checkpoint: {best_model_path}\n")
        file.write(f"segmenter_state_dict: {segmenter_state_path}\n")
        file.write(f"presegmented_datasets: {presegmented_datasets_dir}\n")
        if len(test_results) > 0:
            for key, value in test_results[0].items():
                file.write(f"{key}: {value}\n")

    print(f"Saved pretrained segmenter to: {segmenter_state_path}")
    print(f"Saved presegmented datasets to: {presegmented_datasets_dir}")
    print(f"Best checkpoint: {best_model_path}")


if __name__ == '__main__':
    main(get_args())
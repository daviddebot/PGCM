"""Baseline concept bottleneck models used as competitors to PGCM.

Implements:
- **DNN** — black-box image-to-task baseline (no concept bottleneck).
- **CBMDeep** (``CBM``) — standard concept bottleneck model with a learned encoder.
- **CRM** — concept residual model that adds a residual bypass to CBM.
- **CMR** — concept-based meta-rule model that learns interpretable fuzzy rules.

Each class is a ``pl.LightningModule`` with the same forward signature as the
main PGCM model so they can share the same training loop in ``main.py``.
"""

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.nn.functional import binary_cross_entropy
from torch.optim.lr_scheduler import LambdaLR
import neural_networks
import matplotlib.pyplot as plt
import os
from utils import compute_balanced_accuracy_loader, GlobalBalancedAccuracy

# Adapted from https://github.com/daviddebot/CMR/blob/main/experiments/celeba/models.py


def _plot_intervenability(self, logger=None, nb_runs=1, specific_loader=None, specific_epoch_nb=None, is_pgcm=False):
    # Average over several random intervention orders so the curve reflects
    # the model behavior rather than a single lucky or unlucky shuffle.
        if not os.path.exists(os.path.join(self.output_folder, f"epoch_{specific_epoch_nb if specific_epoch_nb is not None else self.current_epoch}")):
            os.makedirs(os.path.join(self.output_folder, f"epoch_{specific_epoch_nb if specific_epoch_nb is not None else self.current_epoch}"))

        total_nb_concepts = self.nb_possibleobjects * self.nb_concepts

        y_accuracies_per_run = []
        c_accuracies_per_run = []

        y_accuracies_standard_pgcm_per_run = []
        c_accuracies_standard_pgcm_per_run = []

        for _run in range(nb_runs):

            nb_interventions_list = []
            y_accuracies = []
            c_accuracies = []

            y_accuracies_standard_pgcm = []
            c_accuracies_standard_pgcm = []

            shuffle_order = torch.randperm(self.nb_possibleobjects * self.nb_concepts)
            for nb_interventions in range(0, total_nb_concepts, total_nb_concepts // 10):
                intervention_mask = torch.zeros(self.nb_possibleobjects * self.nb_concepts)
                intervention_mask[:nb_interventions] = 1
                intervention_mask = intervention_mask[shuffle_order].view(self.nb_possibleobjects, self.nb_concepts)
                # now run the model on the validation set with this intervention mask
                if is_pgcm:
                    y_acc, c_acc = compute_balanced_accuracy_loader(self, specific_loader if specific_loader is not None else self.val_loader, interventions_mask=intervention_mask.to(self.device))
                    y_acc_standard, c_acc_standard = compute_balanced_accuracy_loader(self, specific_loader if specific_loader is not None else self.val_loader, standard_interventions_mask=intervention_mask.to(self.device)) 
                else:
                    y_acc, c_acc = compute_balanced_accuracy_loader(self, specific_loader if specific_loader is not None else self.val_loader, standard_interventions_mask=intervention_mask.to(self.device))
                nb_interventions_list.append(nb_interventions)
                y_accuracies.append(y_acc)
                c_accuracies.append(c_acc)
                if is_pgcm:
                    y_accuracies_standard_pgcm.append(y_acc_standard)
                    c_accuracies_standard_pgcm.append(c_acc_standard)
            y_accuracies_per_run.append(y_accuracies)
            c_accuracies_per_run.append(c_accuracies)
            if is_pgcm:
                y_accuracies_standard_pgcm_per_run.append(y_accuracies_standard_pgcm)
                c_accuracies_standard_pgcm_per_run.append(c_accuracies_standard_pgcm)

        
        y_accuracies = torch.tensor(y_accuracies_per_run).mean(dim=0).tolist()
        c_accuracies = torch.tensor(c_accuracies_per_run).mean(dim=0).tolist()

        if is_pgcm:
            y_accuracies_standard_pgcm = torch.tensor(y_accuracies_standard_pgcm_per_run).mean(dim=0).tolist()
            c_accuracies_standard_pgcm = torch.tensor(c_accuracies_standard_pgcm_per_run).mean(dim=0).tolist()

        plt.figure()
        plt.plot(nb_interventions_list, y_accuracies, marker='o', label='Interventions')
        plt.xlabel('Number of Intervened Concepts')
        plt.ylabel('Task Accuracy')
        plt.title('Task Accuracy vs Number of Intervened Concepts')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(self.output_folder, f"epoch_{specific_epoch_nb if specific_epoch_nb is not None else self.current_epoch}", 'intervenability_task_accuracy.png'))
        if logger is not None:
            import wandb
            logger.experiment.log({"intervenability/task_accuracy": wandb.Image(plt.gcf())}, step=self.current_epoch)
        plt.close()

        plt.figure()
        plt.plot(nb_interventions_list, c_accuracies, marker='o', label='Interventions')
        plt.xlabel('Number of Intervened Concepts')
        plt.ylabel('Concept Accuracy')
        plt.title('Concept Accuracy vs Number of Intervened Concepts')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(self.output_folder, f"epoch_{specific_epoch_nb if specific_epoch_nb is not None else self.current_epoch}", 'intervenability_concept_accuracy.png'))
        if logger is not None:
            import wandb
            logger.experiment.log({"intervenability/concept_accuracy": wandb.Image(plt.gcf())}, step=self.current_epoch)
        plt.close()

        # save to csv
        import pandas as pd
        df = pd.DataFrame({
            'nb_interventions': nb_interventions_list,
            'y_accuracies': y_accuracies,
            'c_accuracies': c_accuracies
        })
        df.to_csv(os.path.join(self.output_folder, f"epoch_{specific_epoch_nb if specific_epoch_nb is not None else self.current_epoch}", 'intervenability_results.csv'), index=False)

        if is_pgcm:
            plt.figure()
            plt.plot(nb_interventions_list, y_accuracies_standard_pgcm, marker='o', label='Standard Interventions')
            plt.xlabel('Number of Intervened Concepts')
            plt.ylabel('Task Accuracy')
            plt.title('Task Accuracy vs Number of Intervened Concepts (Standard Interventions)')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(self.output_folder, f"epoch_{specific_epoch_nb if specific_epoch_nb is not None else self.current_epoch}", 'intervenability_task_accuracy_standard_pgcm.png'))
            if logger is not None:
                import wandb
                logger.experiment.log({"intervenability/task_accuracy_standard_pgcm": wandb.Image(plt.gcf())}, step=self.current_epoch)
            plt.close()

            plt.figure()
            plt.plot(nb_interventions_list, c_accuracies_standard_pgcm, marker='o', label='Standard Interventions')
            plt.xlabel('Number of Intervened Concepts')
            plt.ylabel('Concept Accuracy')
            plt.title('Concept Accuracy vs Number of Intervened Concepts (Standard Interventions)')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(self.output_folder, f"epoch_{specific_epoch_nb if specific_epoch_nb is not None else self.current_epoch}", 'intervenability_concept_accuracy_standard_pgcm.png'))
            if logger is not None:
                import wandb
                logger.experiment.log({"intervenability/concept_accuracy_standard_pgcm": wandb.Image(plt.gcf())}, step=self.current_epoch)
            plt.close()

            df = pd.DataFrame({
                'nb_interventions': nb_interventions_list,
                'y_accuracies': y_accuracies_standard_pgcm,
                'c_accuracies': c_accuracies_standard_pgcm
            })
            df.to_csv(os.path.join(self.output_folder, f"epoch_{specific_epoch_nb if specific_epoch_nb is not None else self.current_epoch}", 'intervenability_results_standard_pgcm.csv'), index=False)

def _dice_loss(pred_logits, target_mask, smooth=1.0):
    pred_probs = torch.sigmoid(pred_logits)
    pred_flat = pred_probs.reshape(-1)
    target_flat = target_mask.reshape(-1)

    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def _calc_segmentation_loss(pred_logits, gt_mask):
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    bce = bce_criterion(pred_logits, gt_mask)
    dice = _dice_loss(pred_logits, gt_mask)
    return bce + dice


class DNN(pl.LightningModule):
    def __init__(self, emb_size, n_tasks, task_names, dataset, nb_possibleobjects, n_concepts, lr=0.001, plot_frequency=10, val_loader=None, output_folder=None, pos_weights=None, warmup_epochs=10,
                 use_balanced_accuracy=True, task_acc_pos_weights=None, task_acc_neg_weights=None):
        super().__init__()
        # do not save dataloader
        self.save_hyperparameters(ignore=['val_loader'])

        self.warmup_epochs = warmup_epochs

        self.lr = lr
        self.embedding_size = emb_size
        self.n_tasks = n_tasks
        self.task_names = task_names
        self.dataset = dataset
        self.is_cub = self.dataset in ("cub", "cubEMB", "cub_presegm")
        self.nb_possibleobjects = nb_possibleobjects
        self.nb_concepts = n_concepts
        if dataset == "cubEMB":
            self.cp_part_one = neural_networks.FeatureEmbedder(proto_size=emb_size, hidden_size=emb_size)
        elif dataset in ("celebamask", "clevrhans", "cub_presegm"):
            self.cp_part_one = neural_networks.BigImageEmbedder(proto_size=emb_size, single_image=True)
        else:  # MNIST
            self.cp_part_one = neural_networks.ImageEmbedder(embedding_size=emb_size, proto_size=emb_size, single_image=True)
        self.cp_part_two = neural_networks.ProtoToConcepts(emb_size, emb_size, hid=emb_size)  # these don't predict concepts, just an embedding
        self.task_predictor = neural_networks.TaskPredictor(1, emb_size, n_tasks, hidden=emb_size)

        self.plot_frequency = plot_frequency
        self.val_loader = val_loader
        self.output_folder = output_folder
        self.pos_weights = pos_weights
        self.use_balanced_accuracy = use_balanced_accuracy
        self.task_acc_pos_weights = task_acc_pos_weights
        self.task_acc_neg_weights = task_acc_neg_weights

    def concept_predictor(self, batch_x):
        emb = self.cp_part_one(batch_x)
        c_pred = self.cp_part_two(emb)
        return c_pred

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
    
    def forward(self, x, interventions_mask=None, standard_interventions_mask=None):
        batch_x, batch_m, batch_c, batch_y = x

        emb = self.concept_predictor(batch_x)
        task_logits = self.task_predictor(emb.flatten(start_dim=1))
        y_pred = torch.softmax(task_logits, dim=-1) if self.is_cub else torch.sigmoid(task_logits)

        return y_pred, torch.ones_like(batch_c), {"task_logits": task_logits}, None

    def training_step(self, batch, batch_idx):
        batch_x, batch_m, batch_c, batch_y = batch

        p_y, _, extras, _ = self.forward(batch)

        if self.is_cub:
            task_logits = extras["task_logits"]
            y_target = batch_y.argmax(dim=-1) if batch_y.dim() > 1 else batch_y.long()
            y_loss = F.cross_entropy(task_logits, y_target)
            y_acc = (task_logits.argmax(dim=-1) == y_target).float().mean()
        else:
            y_loss = binary_cross_entropy(p_y, batch_y)
            y_acc = ((p_y > 0.5).float() == batch_y).float().mean()
        self.log('train_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)

        self.log('train_y_loss', y_loss, prog_bar=True, on_step=False, on_epoch=True)
        return y_loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_m, batch_c, batch_y = batch

        p_y, _, extras, _ = self.forward(batch)

        if self.is_cub:
            task_logits = extras["task_logits"]
            y_target = batch_y.argmax(dim=-1) if batch_y.dim() > 1 else batch_y.long()
            y_loss = F.cross_entropy(task_logits, y_target)
            y_acc = (task_logits.argmax(dim=-1) == y_target).float().mean()
        else:
            y_loss = binary_cross_entropy(p_y, batch_y)
            y_acc = ((p_y > 0.5).float() == batch_y).float().mean()
        self.log('val_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)

        self.log('val_y_loss', y_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', y_loss, prog_bar=True, on_step=False, on_epoch=True)
        return y_loss

    def on_validation_epoch_end(self):
        pass
    
    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)

    def on_train_epoch_end(self):
        print()

class CBMCommon(pl.LightningModule):  # common functionality for CBMDeep and CBMLinear
    def __init__(self, emb_size, n_tasks, n_concepts, concept_names, task_names, dataset, lr=0.001, concepts_to_task="ground truth", nb_possibleobjects=None,
                 plot_frequency=10, val_loader=None, output_folder=None, pos_weights=None,
                 task_weight=1.0, warmup_epochs=10, intv_prob=0.20, use_balanced_accuracy=True,
                 task_acc_pos_weights=None, task_acc_neg_weights=None, concept_acc_pos_weights=None, concept_acc_neg_weights=None,
                 use_linear_task_predictor=False):
        super().__init__()
        self.save_hyperparameters(ignore=['val_loader'])

        self.warmup_epochs = warmup_epochs

        self.concepts_to_task = concepts_to_task
        self.lr = lr
        self.embedding_size = emb_size
        self.n_tasks = n_tasks
        self.n_concepts = self.nb_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.task_weight = task_weight
        self.dataset = dataset
        self.is_cub = self.dataset in ("cub", "cubEMB", "cub_presegm")
        self.is_cubEMB = self.dataset == "cubEMB"
        self.use_linear_task_predictor = use_linear_task_predictor
        self.nb_possibleobjects = nb_possibleobjects
        if dataset == "cubEMB":
            self.cp_part_one = neural_networks.FeatureEmbedder(proto_size=emb_size, hidden_size=emb_size)
        elif dataset in ("celebamask", "clevrhans", "cub_presegm"):
            self.cp_part_one = neural_networks.BigImageEmbedder(proto_size=emb_size, single_image=True)
        else:  # MNIST
            self.cp_part_one = neural_networks.ImageEmbedder(embedding_size=emb_size, proto_size=emb_size, single_image=True)
        self.cp_part_two = neural_networks.ProtoToConcepts(emb_size, nb_possibleobjects*n_concepts, hid=emb_size)

        self.plot_frequency = plot_frequency
        self.val_loader = val_loader
        self.output_folder = output_folder
        self.pos_weights = pos_weights
        self.use_balanced_accuracy = use_balanced_accuracy
        self.task_acc_pos_weights = task_acc_pos_weights
        self.task_acc_neg_weights = task_acc_neg_weights
        self.concept_acc_pos_weights = concept_acc_pos_weights
        self.concept_acc_neg_weights = concept_acc_neg_weights
        
        if self.use_linear_task_predictor:
            self.task_predictor = torch.nn.Linear(nb_possibleobjects * n_concepts, n_tasks)
        else:
            self.task_predictor = neural_networks.TaskPredictor(nb_possibleobjects, n_concepts, n_tasks, hidden=emb_size)

        self.intv_prob = intv_prob

        if self.use_balanced_accuracy:
            self.train_c_acc = GlobalBalancedAccuracy()
            self.val_c_acc = GlobalBalancedAccuracy()

    def concept_predictor(self, batch_x):
        emb = self.cp_part_one(batch_x)
        c_pred = self.cp_part_two(emb).view(-1, self.nb_possibleobjects, self.n_concepts)
        return c_pred

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
    
    def forward(self, x, interventions_mask=None, standard_interventions_mask=None):
        batch_x, batch_m, batch_c, batch_y = x
        c_pred = torch.sigmoid(self.concept_predictor(batch_x))

        if self.dataset == "cubEMB" and c_pred.dim() == 3:
            c_pred = c_pred[:, 0, :]

        if self.dataset == "cubEMB" and batch_c.dim() == 3:
            batch_c = batch_c[:, 0, :]

        if self.concepts_to_task == "ground truth":
            c_for_task = batch_c
        else:  # threshold
            c_for_task = (c_pred > 0.5).float()

        if self.dataset == "cubEMB" and c_for_task.dim() == 3:
            c_for_task = c_for_task[:, 0, :]

        # use standard_interventions_mask to set intervened concepts to ground truth, also intervene on c_pred
        if standard_interventions_mask is not None:
            c_for_task = c_for_task * (1 - standard_interventions_mask) + batch_c * standard_interventions_mask
            c_pred = c_pred * (1 - standard_interventions_mask) + batch_c * standard_interventions_mask

        task_logits = self.task_predictor(c_for_task.flatten(start_dim=1))
        y_pred = torch.softmax(task_logits, dim=-1) if self.is_cub else torch.sigmoid(task_logits)

        return y_pred, c_pred, {"task_logits": task_logits}, None

    def training_step(self, batch, batch_idx):
        batch_x, batch_m, batch_c, batch_y = batch
        if self.is_cubEMB and batch_c.dim() == 3:
            batch_c = batch_c[:, 0, :]

        if self.training and self.intv_prob > 0:
            interventions_mask = (torch.rand((self.nb_possibleobjects, self.nb_concepts), device=self.device) < self.intv_prob).float()
        else:
            interventions_mask = None

        p_y, p_c, extras, _ = self.forward(batch, standard_interventions_mask=interventions_mask)

        if self.pos_weights is not None:
            batch_weights = torch.where(batch_c == 1, self.pos_weights, torch.tensor(1.0, device=self.device))

        c_loss = binary_cross_entropy(p_c.flatten(start_dim=1), batch_c.flatten(start_dim=1), weight=batch_weights.flatten(start_dim=1) if self.pos_weights is not None else None)
        if self.is_cub:
            task_logits = extras["task_logits"]
            y_target = batch_y.argmax(dim=-1) if batch_y.dim() > 1 else batch_y.long()
            y_loss = F.cross_entropy(task_logits, y_target)
            y_acc = (task_logits.argmax(dim=-1) == y_target).float().mean()
        else:
            y_loss = binary_cross_entropy(p_y, batch_y)
            y_acc = ((p_y > 0.5).float() == batch_y).float().mean()

        self.log('train_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)
        if self.use_balanced_accuracy:
            self.train_c_acc(p_c > 0.5, batch_c)
        else:
            c_acc = ((p_c > 0.5).float() == batch_c).float().mean()
            self.log('train_concept_acc', c_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.log('train_y_loss', y_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_c_loss', c_loss, prog_bar=True, on_step=False, on_epoch=True)            

        loss = c_loss + self.task_weight * y_loss
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_m, batch_c, batch_y = batch
        if self.is_cubEMB and batch_c.dim() == 3:
            batch_c = batch_c[:, 0, :]

        p_y, p_c, extras, _ = self.forward(batch)

        c_loss = binary_cross_entropy(p_c.flatten(start_dim=1), batch_c.flatten(start_dim=1))     
        if self.is_cub:
            task_logits = extras["task_logits"]
            y_target = batch_y.argmax(dim=-1) if batch_y.dim() > 1 else batch_y.long()
            y_loss = F.cross_entropy(task_logits, y_target)
            y_acc = (task_logits.argmax(dim=-1) == y_target).float().mean()
        else:
            y_loss = binary_cross_entropy(p_y, batch_y)
            y_acc = ((p_y > 0.5).float() == batch_y).float().mean()

        self.log('val_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)
        if self.use_balanced_accuracy:
            self.val_c_acc(p_c > 0.5, batch_c)
        else:
            c_acc = ((p_c > 0.5).float() == batch_c).float().mean()
            self.log('val_concept_acc', c_acc, prog_bar=True, on_step=False, on_epoch=True)

        loss = c_loss + self.task_weight * y_loss

        self.log('val_y_loss', y_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_c_loss', c_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', loss, prog_bar=True, on_step=False, on_epoch=True)            

        return loss

    def on_validation_epoch_end(self):
        if self.use_balanced_accuracy:
            self.log('val_concept_acc_epoch', self.val_c_acc.compute(), prog_bar=True)
            self.val_c_acc.reset()
    
    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)

    def on_train_epoch_end(self):
        if self.use_balanced_accuracy:
            self.log('train_concept_acc_epoch', self.train_c_acc.compute(), prog_bar=True)
            self.train_c_acc.reset()
        print()
        if self.current_epoch % self.plot_frequency == 0:
            _plot_intervenability(self, logger=None)

class CBMLinear(CBMCommon):
    def __init__(self, emb_size, n_tasks, n_concepts, concept_names, task_names, dataset, lr=0.001, concepts_to_task="ground truth", nb_possibleobjects=None,
                 task_weight=1.0, plot_frequency=10, val_loader=None, output_folder=None, pos_weights=None, warmup_epochs=10, intv_prob=0.20,
                 use_balanced_accuracy=True, task_acc_pos_weights=None, task_acc_neg_weights=None, concept_acc_pos_weights=None, concept_acc_neg_weights=None,
                 use_linear_task_predictor=False):
        super().__init__(emb_size, n_tasks, n_concepts, concept_names, task_names, dataset, lr=lr, 
                         concepts_to_task=concepts_to_task, task_weight=task_weight, nb_possibleobjects=nb_possibleobjects, 
                         plot_frequency=plot_frequency, val_loader=val_loader, output_folder=output_folder, pos_weights=pos_weights, warmup_epochs=warmup_epochs, intv_prob=intv_prob,
                         use_balanced_accuracy=use_balanced_accuracy, task_acc_pos_weights=task_acc_pos_weights, task_acc_neg_weights=task_acc_neg_weights,
                         concept_acc_pos_weights=concept_acc_pos_weights, concept_acc_neg_weights=concept_acc_neg_weights,
                         use_linear_task_predictor=use_linear_task_predictor)
        self.save_hyperparameters(ignore=['val_loader'])

        self.task_predictor = torch.nn.Linear(nb_possibleobjects * self.n_concepts, n_tasks)


class CBMDeep(CBMCommon):
    def __init__(self, emb_size, n_tasks, n_concepts, concept_names, task_names, dataset, lr=0.001, concepts_to_task="ground truth", nb_possibleobjects=None,
                 task_weight=1.0, plot_frequency=10, val_loader=None, output_folder=None, pos_weights=None, warmup_epochs=10, intv_prob=0.20,
                 use_balanced_accuracy=True, task_acc_pos_weights=None, task_acc_neg_weights=None, concept_acc_pos_weights=None, concept_acc_neg_weights=None,
                 use_linear_task_predictor=False):        
        super().__init__(emb_size, n_tasks, n_concepts, concept_names, task_names, dataset, lr=lr, 
                         concepts_to_task=concepts_to_task, task_weight=task_weight, nb_possibleobjects=nb_possibleobjects, 
                         plot_frequency=plot_frequency, val_loader=val_loader, output_folder=output_folder, pos_weights=pos_weights, warmup_epochs=warmup_epochs, intv_prob=intv_prob,
                         use_balanced_accuracy=use_balanced_accuracy, task_acc_pos_weights=task_acc_pos_weights, task_acc_neg_weights=task_acc_neg_weights,
                         concept_acc_pos_weights=concept_acc_pos_weights, concept_acc_neg_weights=concept_acc_neg_weights,
                         use_linear_task_predictor=use_linear_task_predictor)
        self.save_hyperparameters(ignore=['val_loader'])





class CRM(pl.LightningModule): 
    def __init__(self, emb_size, n_tasks, n_concepts, concept_names, task_names, dataset, lr=0.001, concepts_to_task="ground truth", nb_possibleobjects=None,
                 task_weight=1.0, plot_frequency=10, val_loader=None, output_folder=None, pos_weights=None, warmup_epochs=10, intv_prob=0.20,
                 use_balanced_accuracy=True, task_acc_pos_weights=None, task_acc_neg_weights=None, concept_acc_pos_weights=None, concept_acc_neg_weights=None,
                 use_linear_task_predictor=False):
        super().__init__()
        self.save_hyperparameters(ignore=['val_loader'])

        self.warmup_epochs = warmup_epochs

        self.concepts_to_task = concepts_to_task
        self.lr = lr
        self.embedding_size = emb_size
        self.n_tasks = n_tasks
        self.n_concepts = self.nb_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.task_weight = task_weight
        self.dataset = dataset
        self.is_cub = self.dataset in ("cub", "cubEMB", "cub_presegm")
        self.is_cubEMB = self.dataset == "cubEMB"
        self.nb_possibleobjects = nb_possibleobjects
        self.use_linear_task_predictor = use_linear_task_predictor
        if dataset == "cubEMB":
            self.cp_part_one = neural_networks.FeatureEmbedder(proto_size=emb_size, hidden_size=emb_size)
        elif dataset in ("celebamask", "clevrhans", "cub_presegm"):
            self.cp_part_one = neural_networks.BigImageEmbedder(proto_size=emb_size, single_image=True)
        else:  # MNIST
            self.cp_part_one = neural_networks.ImageEmbedder(embedding_size=emb_size, proto_size=emb_size, single_image=True)
        self.cp_part_two = neural_networks.ProtoToConcepts(emb_size, nb_possibleobjects*n_concepts+emb_size, hid=emb_size)

        if self.use_linear_task_predictor:
            self.task_predictor = torch.nn.Linear(nb_possibleobjects * n_concepts + emb_size, n_tasks)
        else:
            self.task_predictor = neural_networks.TaskPredictor(nb_possibleobjects, n_concepts, n_tasks, hidden=emb_size, residual=emb_size)

        self.plot_frequency = plot_frequency
        self.val_loader = val_loader
        self.output_folder = output_folder
        self.pos_weights = pos_weights
        self.use_balanced_accuracy = use_balanced_accuracy
        self.task_acc_pos_weights = task_acc_pos_weights
        self.task_acc_neg_weights = task_acc_neg_weights
        self.concept_acc_pos_weights = concept_acc_pos_weights
        self.concept_acc_neg_weights = concept_acc_neg_weights

        # assert self.pos_weights is not None

        self.intv_prob = intv_prob

        if self.use_balanced_accuracy:
            self.train_c_acc = GlobalBalancedAccuracy()
            self.val_c_acc = GlobalBalancedAccuracy()

    def concept_predictor(self, batch_x):
        emb = self.cp_part_one(batch_x)
        c_pred_and_emb = self.cp_part_two(emb)
        c_pred = c_pred_and_emb[:, :self.nb_possibleobjects * self.n_concepts]
        emb = c_pred_and_emb[:, self.nb_possibleobjects * self.n_concepts:]
        c_pred = c_pred.view(-1, self.nb_possibleobjects, self.n_concepts)
        return c_pred, emb

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
    
    def forward(self, x, interventions_mask=None, standard_interventions_mask=None):
        batch_x, batch_m, batch_c, batch_y = x
        c_logits, emb = self.concept_predictor(batch_x)
        c_pred = torch.sigmoid(c_logits)

        if self.is_cubEMB and c_pred.dim() == 3:
            c_pred = c_pred[:, 0, :]

        if self.is_cubEMB and batch_c.dim() == 3:
            batch_c = batch_c[:, 0, :]

        if self.concepts_to_task == "ground truth":
            c_for_task = batch_c
        else:  # threshold
            c_for_task = (c_pred > 0.5).float()

        if self.is_cubEMB and c_for_task.dim() == 3:
            c_for_task = c_for_task[:, 0, :]

        # use standard_interventions_mask to set intervened concepts to ground truth, also intervene on c_pred
        if standard_interventions_mask is not None:
            c_for_task = c_for_task * (1 - standard_interventions_mask) + batch_c * standard_interventions_mask
            c_pred = c_pred * (1 - standard_interventions_mask) + batch_c * standard_interventions_mask

        task_logits = self.task_predictor(torch.cat([c_for_task.flatten(start_dim=1), emb], dim=1))
        y_pred = torch.softmax(task_logits, dim=-1) if self.is_cub else torch.sigmoid(task_logits)

        return y_pred, c_pred, {"task_logits": task_logits}, None

    def training_step(self, batch, batch_idx):
        batch_x, batch_m, batch_c, batch_y = batch
        if self.is_cubEMB and batch_c.dim() == 3:
            batch_c = batch_c[:, 0, :]

        if self.training and self.intv_prob > 0:
            interventions_mask = (torch.rand((self.nb_possibleobjects, self.nb_concepts), device=self.device) < self.intv_prob).float()
        else:
            interventions_mask = None

        p_y, p_c, aux, _ = self.forward(batch, standard_interventions_mask=interventions_mask)

        if self.pos_weights is not None:
            batch_weights = torch.where(batch_c == 1, self.pos_weights, torch.tensor(1.0, device=self.device))

        c_loss = binary_cross_entropy(p_c.flatten(start_dim=1), batch_c.flatten(start_dim=1), weight=batch_weights.flatten(start_dim=1) if self.pos_weights is not None else None)
        if self.is_cub:
            task_logits = aux["task_logits"]
            y_target = batch_y.argmax(dim=-1) if batch_y.dim() > 1 else batch_y.long()
            y_loss = F.cross_entropy(task_logits, y_target)
            y_acc = (task_logits.argmax(dim=-1) == y_target).float().mean()
        else:
            y_loss = binary_cross_entropy(p_y, batch_y)
            y_acc = ((p_y > 0.5).float() == batch_y).float().mean()

        self.log('train_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)
        if self.use_balanced_accuracy:
            self.train_c_acc(p_c > 0.5, batch_c)
        else:
            c_acc = ((p_c > 0.5).float() == batch_c).float().mean()
            self.log('train_concept_acc', c_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.log('train_y_loss', y_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_c_loss', c_loss, prog_bar=True, on_step=False, on_epoch=True)            

        loss = c_loss + self.task_weight * y_loss
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_m, batch_c, batch_y = batch
        if self.is_cubEMB and batch_c.dim() == 3:
            batch_c = batch_c[:, 0, :]

        if self.training and self.intv_prob > 0:
            interventions_mask = (torch.rand((self.nb_possibleobjects, self.nb_concepts), device=self.device) < self.intv_prob).float()
        else:
            interventions_mask = None

        p_y, p_c, aux, _ = self.forward(batch, standard_interventions_mask=interventions_mask)

        if self.pos_weights is not None:
            batch_weights = torch.where(batch_c == 1, self.pos_weights, torch.tensor(1.0, device=self.device))

        c_loss = binary_cross_entropy(p_c.flatten(start_dim=1), batch_c.flatten(start_dim=1), weight=batch_weights.flatten(start_dim=1) if self.pos_weights is not None else None)
        y_loss = binary_cross_entropy(p_y, batch_y)
        
        if self.is_cub:
            task_logits = aux["task_logits"]
            y_target = batch_y.argmax(dim=-1) if batch_y.dim() > 1 else batch_y.long()
            y_loss = F.cross_entropy(task_logits, y_target)
            y_acc = (task_logits.argmax(dim=-1) == y_target).float().mean()
        else:
            y_loss = binary_cross_entropy(p_y, batch_y)
            y_acc = ((p_y > 0.5).float() == batch_y).float().mean()

        self.log('val_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)
        if self.use_balanced_accuracy:
            self.val_c_acc(p_c > 0.5, batch_c)
        else:
            c_acc = ((p_c > 0.5).float() == batch_c).float().mean()
            self.log('val_concept_acc', c_acc, prog_bar=True, on_step=False, on_epoch=True)

        loss = c_loss + self.task_weight * y_loss

        self.log('val_y_loss', y_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_c_loss', c_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', loss, prog_bar=True, on_step=False, on_epoch=True)            

        return loss
    
    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)

    def on_validation_epoch_end(self):
        if self.use_balanced_accuracy:
            self.log('val_concept_acc_epoch', self.val_c_acc.compute(), prog_bar=True)
            self.val_c_acc.reset()

    def on_train_epoch_end(self):
        if self.use_balanced_accuracy:
            self.log('train_concept_acc_epoch', self.train_c_acc.compute(), prog_bar=True)
            self.train_c_acc.reset()
        print()
        if self.current_epoch % self.plot_frequency == 0:
            _plot_intervenability(self, logger=None)

    
class CMR(pl.LightningModule): 
    def __init__(self, emb_size, n_tasks, n_concepts, concept_names, task_names, dataset, lr=0.001, concepts_to_task="ground truth", nb_possibleobjects=None,
                 nb_rules=5, rule_emb_size=100, plot_frequency=10, val_loader=None, output_folder=None, pos_weights=None,
                 task_weight=1.0, warmup_epochs=10, intv_prob=0.20, use_balanced_accuracy=True,
                 task_acc_pos_weights=None, task_acc_neg_weights=None, concept_acc_pos_weights=None, concept_acc_neg_weights=None,
                 use_linear_task_predictor=False):
        super().__init__()
        self.save_hyperparameters(ignore=['val_loader'])

        self.warmup_epochs = warmup_epochs

        self.concepts_to_task = concepts_to_task
        self.lr = lr
        self.embedding_size = emb_size
        self.n_tasks = n_tasks
        self.n_concepts = self.nb_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.task_weight = task_weight
        self.dataset = dataset
        self.is_cub = self.dataset in ("cub", "cubEMB", "cub_presegm")
        self.is_cubEMB = self.dataset == "cubEMB"
        self.nb_possibleobjects = nb_possibleobjects
        self.nb_rules = nb_rules
        self.rule_emb_size = rule_emb_size
        if dataset in ("cub", "cubEMB"):
            self.cp_part_one = neural_networks.FeatureEmbedder(proto_size=emb_size, hidden_size=emb_size)
        elif dataset in ("celebamask", "clevrhans", "cub_presegm"):
            self.cp_part_one = neural_networks.BigImageEmbedder(proto_size=emb_size, single_image=True)
        else:  # MNIST
            self.cp_part_one = neural_networks.ImageEmbedder(embedding_size=emb_size, proto_size=emb_size, single_image=True)
        self.cp_part_two = neural_networks.ProtoToConcepts(emb_size, nb_possibleobjects*n_concepts+n_tasks*nb_rules, hid=emb_size)

        self.memory = torch.nn.Embedding(nb_rules*n_tasks, rule_emb_size)

        if n_concepts < 30:
            self.rule_decoder = torch.nn.Sequential(
                torch.nn.Linear(rule_emb_size, rule_emb_size),
                torch.nn.ReLU(),
                torch.nn.Linear(rule_emb_size, nb_possibleobjects * n_concepts * 3)
            )
        else:
            # bigger
            self.rule_decoder = torch.nn.Sequential(
                torch.nn.Linear(rule_emb_size, rule_emb_size),
                torch.nn.ReLU(),
                torch.nn.Linear(rule_emb_size, rule_emb_size),
                torch.nn.ReLU(),
                torch.nn.Linear(rule_emb_size, nb_possibleobjects * n_concepts * 3)
            )

        self.plot_frequency = plot_frequency
        self.val_loader = val_loader
        self.output_folder = output_folder
        self.pos_weights = pos_weights
        self.use_balanced_accuracy = use_balanced_accuracy
        self.task_acc_pos_weights = task_acc_pos_weights
        self.task_acc_neg_weights = task_acc_neg_weights
        self.concept_acc_pos_weights = concept_acc_pos_weights
        self.concept_acc_neg_weights = concept_acc_neg_weights

        if dataset != "cubEMB":
            assert self.pos_weights is not None

        self.intv_prob = intv_prob

        if self.use_balanced_accuracy:
            self.train_c_acc = GlobalBalancedAccuracy()
            self.val_c_acc = GlobalBalancedAccuracy()

    def concept_predictor(self, batch_x):
        emb = self.cp_part_one(batch_x)
        c_pred_and_emb = self.cp_part_two(emb)
        c_pred = c_pred_and_emb[:, :self.nb_possibleobjects * self.n_concepts]  # shape: batch_size, nb_possibleobjects*n_concepts
        c_pred = c_pred.view(-1, self.nb_possibleobjects, self.n_concepts)
        selector_logits = c_pred_and_emb[:, self.nb_possibleobjects * self.n_concepts:]  # shape: batch_size, n_tasks*nb_rules
        selector_probs = torch.softmax(selector_logits.view(-1, self.n_tasks, self.nb_rules), dim=-1)  # shape: batch_size, n_tasks, nb_rules
        return c_pred, selector_probs

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
    
    def forward(self, x, interventions_mask=None, standard_interventions_mask=None):
        batch_x, batch_m, batch_c, batch_y = x
        c_logits, rule_probs = self.concept_predictor(batch_x)  # rule_probs: batch_size, n_tasks, nb_rules | c_logits: batch_size, nb_possibleobjects, n_concepts
        c_pred = torch.sigmoid(c_logits)

        if self.is_cubEMB and batch_c.dim() == 2:
            batch_c = batch_c.unsqueeze(1)

        rule_embs = self.memory.weight.view(self.n_tasks, self.nb_rules, self.rule_emb_size)  # n_tasks, nb_rules, rule_emb_size
        rule_outputs = self.rule_decoder(rule_embs)  # n_tasks, nb_rules, nb_possibleobjects * n_concepts * 3
        rule_outputs = rule_outputs.view(self.n_tasks, self.nb_rules, self.nb_possibleobjects*self.n_concepts, 3)  # n_tasks, nb_rules, nb_possibleobjects*n_concepts, 3
        rule_vars = torch.softmax(rule_outputs, dim=-1)  # n_tasks, nb_rules, nb_possibleobjects*n_concepts, 3
        eps = 1e-4 if self.nb_concepts < 30 else 1e-6
        rule_vars = rule_vars * (1 - 2*eps)

        if self.concepts_to_task == "ground truth":
            c_for_task = batch_c
        else:  # threshold
            c_for_task = (c_pred > 0.5).float()

        # use standard_interventions_mask to set intervened concepts to ground truth, also intervene on c_pred
        if standard_interventions_mask is not None:
            c_for_task = c_for_task * (1 - standard_interventions_mask) + batch_c * standard_interventions_mask  # batch_size, nb_possibleobjects, n_concepts
            c_pred = c_pred * (1 - standard_interventions_mask) + batch_c * standard_interventions_mask

        # rule vars: set [2] to 1 and others to 0 if self.pos_weights is 0 for that concept
        if self.pos_weights is not None:    
            pos_weights_expanded = self.pos_weights.view(-1).unsqueeze(0).unsqueeze(0).expand(self.n_tasks, self.nb_rules, -1)  # n_tasks, nb_rules, nb_possibleobjects*n_concepts
            mask_zero = (pos_weights_expanded > 100000).unsqueeze(-1)  # n_tasks, nb_rules, nb_possibleobjects*n_concepts, 1
            rule_vars = torch.where(mask_zero, torch.tensor([0.0, 0.0, 1.0], device=self.device), rule_vars)

        # print('lll', torch.min(rule_vars).item(), torch.max(rule_vars).item())

        rule_vars_btroc3 = rule_vars.unsqueeze(0).expand(c_for_task.size(0), -1, -1, -1, -1)  # batch_size, n_tasks, nb_rules, nb_possibleobjects*n_concepts, 3
        c_for_task_btroc = c_for_task.unsqueeze(1).unsqueeze(2).expand(-1, self.n_tasks, self.nb_rules, -1, -1).reshape(c_for_task.size(0), self.n_tasks, self.nb_rules, self.nb_possibleobjects*self.n_concepts)  # batch_size, n_tasks, nb_rules, nb_possibleobjects*n_concepts
        rule_evals = rule_vars_btroc3[:, :, :, :, 0] * c_for_task_btroc + rule_vars_btroc3[:, :, :, :, 1] * (1 - c_for_task_btroc) + rule_vars_btroc3[:, :, :, :, 2]  # batch_size, n_tasks, nb_rules, nb_possibleobjects*n_concepts

        # print('rrr', torch.min(rule_evals).item(), torch.max(rule_evals).item())

        task_scores = (rule_evals.prod(dim=-1) * rule_probs).sum(dim=-1)  # batch_size, n_tasks
        task_logits = None
        y_pred = task_scores

        # print(torch.min(y_pred).item(), torch.max(y_pred).item())

        return y_pred, c_pred, {"task_logits": task_logits}, None

    def training_step(self, batch, batch_idx):
        batch_x, batch_m, batch_c, batch_y = batch
        if self.is_cubEMB and batch_c.dim() == 2:
            batch_c = batch_c.unsqueeze(1)

        # batch_y = torch.ones_like(batch_y)

        if self.training and self.intv_prob > 0:
            interventions_mask = (torch.rand((self.nb_possibleobjects, self.nb_concepts), device=self.device) < self.intv_prob).float()
        else:
            interventions_mask = None

        if self.pos_weights is not None:
            batch_weights = torch.where(batch_c == 1, self.pos_weights, torch.tensor(1.0, device=self.device))

        p_y, p_c, aux, _ = self.forward(batch, standard_interventions_mask=interventions_mask)

        c_loss = binary_cross_entropy(p_c.flatten(start_dim=1), batch_c.flatten(start_dim=1), weight=batch_weights.flatten(start_dim=1) if self.pos_weights is not None else None)
        y_loss = binary_cross_entropy(p_y, batch_y)
        if self.is_cub:
            y_target = batch_y.argmax(dim=-1) if batch_y.dim() > 1 else batch_y.long()
            y_acc = (p_y.argmax(dim=-1) == y_target).float().mean()
        else:
            y_acc = ((p_y > 0.5).float() == batch_y).float().mean()

        self.log('train_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)
        if self.use_balanced_accuracy:
            self.train_c_acc(p_c > 0.5, batch_c)
        else:
            c_acc = ((p_c > 0.5).float() == batch_c).float().mean()
            self.log('train_concept_acc', c_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.log('train_y_loss', y_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_c_loss', c_loss, prog_bar=True, on_step=False, on_epoch=True)            

        loss = c_loss + self.task_weight * y_loss
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_m, batch_c, batch_y = batch
        if self.is_cubEMB and batch_c.dim() == 2:
            batch_c = batch_c.unsqueeze(1)

        p_y, p_c, aux, _ = self.forward(batch)

        if self.pos_weights is not None:
            batch_weights = torch.where(batch_c == 1, self.pos_weights, torch.tensor(1.0, device=self.device))

        c_loss = binary_cross_entropy(p_c.flatten(start_dim=1), batch_c.flatten(start_dim=1), weight=batch_weights.flatten(start_dim=1) if self.pos_weights is not None else None)
        y_loss = binary_cross_entropy(p_y, batch_y)
        if self.is_cub:
            y_target = batch_y.argmax(dim=-1) if batch_y.dim() > 1 else batch_y.long()
            y_acc = (p_y.argmax(dim=-1) == y_target).float().mean()
        else:
            y_acc = ((p_y > 0.5).float() == batch_y).float().mean()

        self.log('val_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)
        if self.use_balanced_accuracy:
            self.val_c_acc(p_c > 0.5, batch_c)
        else:
            c_acc = ((p_c > 0.5).float() == batch_c).float().mean()
            self.log('val_concept_acc', c_acc, prog_bar=True, on_step=False, on_epoch=True)

        loss = c_loss + self.task_weight * y_loss

        self.log('val_y_loss', y_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_c_loss', c_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', loss, prog_bar=True, on_step=False, on_epoch=True)            

        return loss
    
    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)

    def on_validation_epoch_end(self):
        if self.use_balanced_accuracy:
            self.log('val_concept_acc_epoch', self.val_c_acc.compute(), prog_bar=True)
            self.val_c_acc.reset()

    def on_train_epoch_end(self):
        if self.use_balanced_accuracy:
            self.log('train_concept_acc_epoch', self.train_c_acc.compute(), prog_bar=True)
            self.train_c_acc.reset()
        print()
        if self.current_epoch % self.plot_frequency == 0:
            _plot_intervenability(self, logger=None)
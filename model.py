"""PGCM (Prototype-Grounded Concept Model) — core LightningModule.

This module defines the ``Model`` class, the main PGCM architecture used for
training and evaluation.  The model learns a set of prototype embeddings that
are decoded into concept probability vectors and optionally into
reconstructions.  At the midpoint of training, prototypes are *swapped* with
the closest real instances from the training set, anchoring them in actual data
(see ``swap_prototypes_with_instances``).

The file also contains the segmentation loss helpers (``dice_loss``,
``calc_segmentation_loss``) used when the model jointly learns an object
segmenter.
"""

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.nn.functional import binary_cross_entropy
from torch.optim.lr_scheduler import LambdaLR
import os
from utils import get_used_prototypes_indices, tokp_objects_for_prototype
from utils import get_objects_notraining, reconstruction_util, get_current_lam_entropy, generate_plots, GlobalBalancedAccuracy

import neural_networks

from scipy.optimize import linear_sum_assignment

eps = 0.0001

class Model(pl.LightningModule):
    def __init__(self, nb_proto, proto_size, embedding_size, nb_concepts, nb_tasks, lr, reconstruction, lam, concept_names, val_loader, train_loader, batch_size, concepts_to_task="ground truth",
                 nb_possibleobjects = 2, lam_reconstruction=1.0, segmentation_method='true', resolution = (56, 28), dataset_name='mnist', lam_proto_emb=1.0,
                 patch_size = 16, lam_kl=1.0, lam_segmentation=1.0, lam_orth=1.0, lam_batch_entropy=0.0, decay_lam_entropy=True, output_folder=None, plot_frequency=10, map_to_train_set=False, pos_weights=None, pgcm=False,
                 warmup_epochs=10, intv_prob=0.20, latent_prior_sdev=5.0, autoencoder_encoder_path=None, autoencoder_decoder_path=None, use_balanced_accuracy=True, use_pretrained_autoencoder=False,
                 use_pretrained_segmenter=False, use_linear_task_predictor=False):
        super(Model, self).__init__()

        # do NOT SAVE VAL LOADER AND TRAIN LOADER AS HYPERPARAMETERS
        self.save_hyperparameters(ignore=['val_loader', 'train_loader'])

        assert nb_concepts == len(concept_names), "Number of concepts must match the length of concept names."
        assert output_folder is not None, "Please provide an output folder to save the plots."

        self.use_balanced_accuracy = use_balanced_accuracy
        self.use_pretrained_autoencoder = use_pretrained_autoencoder
        self.use_pretrained_segmenter = use_pretrained_segmenter
        self.use_linear_task_predictor = use_linear_task_predictor
        self.is_cubEMB = dataset_name == 'cubEMB'
        self.is_multiclass_task = dataset_name in ('cubEMB', 'cub_presegm')
        # if self.is_multiclass_task:
        #     self.use_balanced_accuracy = False

        self.segmentation_method = segmentation_method

        self.map_to_train_set = map_to_train_set

        self.use_correct_loss = True

        self.patch_size = patch_size
        self.resolution = resolution

        self.warmup_epochs = warmup_epochs

        self.nb_proto = nb_proto
        self.prototypes = torch.nn.Embedding(nb_proto, proto_size)
        
        if self.is_cubEMB:
            # CUBEMB batches are feature tensors, so we learn a feature encoder/decoder
            # instead of the image segmentation and masking stack used for raw images.
            try:
                sample_x = next(iter(train_loader))[0]
            except StopIteration as exc:
                raise ValueError("CUBEMB requires a non-empty train_loader to infer feature dimensionality.") from exc
            feature_dim = int(sample_x.flatten(start_dim=1).shape[1])
            self.cub_feature_dim = feature_dim
            self.map_image_to_object_emb = neural_networks.FeatureEmbedder(
                input_dim=feature_dim,
                proto_size=proto_size,
                hidden_size=embedding_size,
            ).to(self.device)
        elif segmentation_method in ['true','slot_attention']:
            self.map_image_to_object_emb = neural_networks.ImageEmbedder(embedding_size=embedding_size, proto_size=proto_size)
        else:
            self.map_image_to_object_emb = neural_networks.BigImageEmbedder(proto_size=proto_size)
        self.nb_possibleobjects = nb_possibleobjects
        self.nb_concepts = nb_concepts

        self.lam_prototype_emb = lam_proto_emb
        self.dataset_name = dataset_name
        self.lr = lr
        self.lam_start = lam
        self.lam_batch_entropy = lam_batch_entropy
        self.decay_lam_entropy = decay_lam_entropy
        self.lam = lam
        self.lam_reconstruction = lam_reconstruction
        self.lam_kl = lam_kl
        self.lam_segmentation = lam_segmentation
        self.lam_orth = lam_orth
        self.reconstruction = reconstruction
        self.proto_size = proto_size
        self.output_folder = output_folder
        self.concept_names = concept_names
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.plot_frequency = plot_frequency
        self.batch_size = batch_size
        self.concepts_to_task = concepts_to_task
        assert concepts_to_task in ["ground truth", "thresholding"]

        self.pgcm = pgcm

        self.pos_weights = pos_weights

        self.map_proto_to_concepts = neural_networks.ProtoToConcepts(proto_size, nb_concepts, hid=embedding_size)

        if self.use_linear_task_predictor:
            self.task_predictor = torch.nn.Linear(nb_possibleobjects * nb_concepts, nb_tasks)
        else:
            self.task_predictor = neural_networks.TaskPredictor(nb_possibleobjects, nb_concepts, nb_tasks, hidden=embedding_size)

        if self.is_cubEMB:
            self.seg_model = None
        elif self.use_pretrained_segmenter:
            self.seg_model = None
        elif segmentation_method == 'slot_attention':
            # Slot-attention support is intentionally disabled here because the
            # dedicated implementation is no longer part of this workspace.
            raise NotImplementedError("slot_attention segmentation is no longer available in this workspace.")
        else:
            if self.dataset_name in ('celebamask', 'clevrhans', 'cub', 'cub_presegm'):
                self.seg_model = neural_networks.ResNetUNetSegmenter(n_class=nb_possibleobjects).to(self.device)
            else:
                self.seg_model = neural_networks.ResNetUNetSegmenterMNIST(n_class=nb_possibleobjects).to(self.device)

        if self.is_cubEMB:
            # In the CUB feature setting, prototype decoding maps back to the same
            # embedding space, and the posterior encoder mirrors that space as well.
            self.map_proto_to_image = neural_networks.FeatureDecoder(
                proto_size=proto_size,
                output_dim=self.cub_feature_dim,
                hidden_size=embedding_size,
            ).to(self.device)
            self.latent_posterior = neural_networks.FeatureEmbedder(
                input_dim=self.cub_feature_dim,
                proto_size=2 * proto_size,
                hidden_size=embedding_size,
            ).to(self.device)
            self.map_proto_image_to_proto = self.map_image_to_object_emb
        elif self.dataset_name in ('celebamask', 'clevrhans', 'cub_presegm'):
            self.map_proto_to_image = neural_networks.PrototypeDecoder(proto_size, out_channels=3, patch_size=self.patch_size).to(self.device)
            self.latent_posterior = neural_networks.BigImageEmbedder(proto_size=2*proto_size)
            self.map_proto_image_to_proto = self.map_image_to_object_emb  #neural_networks.BigImageEmbedder(proto_size=proto_size)
        else:  # MNIST
            self.map_proto_to_image = neural_networks.PrototypeDecoderMNIST(embedding_dim=proto_size).to(self.device)
            self.latent_posterior = neural_networks.ImageEmbedder(embedding_size=embedding_size, proto_size=2*proto_size)
            # self.map_proto_image_to_proto = neural_networks.ImageEmbedder(embedding_size=proto_size, proto_size=proto_size)
            self.map_proto_image_to_proto = self.map_image_to_object_emb  # use same encoder as for object embeddings => better guarantees

        encoder_state = None
        if autoencoder_encoder_path and self.dataset_name not in ('celebamask', 'clevrhans', 'cub', 'cub_presegm'):
            encoder_state = torch.load(autoencoder_encoder_path, map_location=self.device)
            if any(key.startswith("map_image_to_object_emb") for key in encoder_state.keys()):
                self.map_proto_image_to_proto = neural_networks.ImageEmbedder(
                    embedding_size=embedding_size,
                    proto_size=proto_size,
                )

        if autoencoder_encoder_path:
            self._load_autoencoder_weights(self.map_proto_image_to_proto, autoencoder_encoder_path, "encoder", state=encoder_state)
        if autoencoder_decoder_path:
            self._load_autoencoder_weights(self.map_proto_to_image, autoencoder_decoder_path, "decoder")

        # torch sequential
        self.latent_prior_mean = 0 
        self.latent_prior_sdev = latent_prior_sdev 
        self.combine_z_and_proto_emb = torch.nn.Sequential(
            torch.nn.Linear(proto_size + proto_size, proto_size),
            torch.nn.ReLU(),
        )

        self.stored_activations = None

        self.probabilistic_reconstruction = True  # if false, fuzzy
        # self.probabilistic_reconstruction = False  # if false, fuzzy

        self.last_prototypes_as_images_posterior = None
        self.last_prototype_probs_per_object = None

        self._logger = None

        self.closest_masked_images = None
        self.swapped = False

        self.intv_prob = intv_prob

        self.masked_prototypes = []

        self.forced_concepts_per_proto_mask = None
        self.forced_concepts_per_proto = None

        self.FIXED_LR = False
        self.ALWAYS_USE_TRUE_MASKS = False
        self.USE_INITIAL_PROTO_EMBS = False

        if self.use_balanced_accuracy and (self.segmentation_method == 'mask' or self.is_cubEMB):
            self.train_c_acc = GlobalBalancedAccuracy()
            self.val_c_acc = GlobalBalancedAccuracy()

    def _load_autoencoder_weights(self, module, path, name, state=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Autoencoder {name} weights not found: {path}")
        if state is None:
            state = torch.load(path, map_location=self.device)
        module.load_state_dict(state)
        module.eval()
        module.requires_grad_(False)
        print(f"Loaded autoencoder {name} weights from: {path}")

    def set_forced_concepts_per_proto(self, forced_concepts_per_proto, forced_concepts_per_proto_mask):
        self.forced_concepts_per_proto = forced_concepts_per_proto
        self.forced_concepts_per_proto_mask = forced_concepts_per_proto_mask

    def update_masked_prototypes(self, masked_prototypes):
        self.masked_prototypes = masked_prototypes

    def get_objects_objectcentric(self, x):
        return self.seg_model(x)

    def _forward_cubEMB(self, x, interventions_mask=None, standard_interventions_mask=None):
        X, M, C, Y = x
        if C.dim() == 2:
            C = C.unsqueeze(1)
        if X.dim() > 2:
            X = X.flatten(start_dim=1)

        object_embeddings = self.map_image_to_object_emb(X).unsqueeze(1)
        prototype_embeddings = self.prototypes.weight
        nb_proto = prototype_embeddings.shape[0]

        z_posterior_mean = torch.zeros((X.shape[0], self.proto_size), device=self.device)
        z_posterior_logvar = torch.zeros_like(z_posterior_mean)
        z_posterior_sdev = torch.ones_like(z_posterior_mean)

        if self.reconstruction and not self.swapped and not self.use_pretrained_autoencoder:
            prototypes_as_features = self.map_proto_to_image(prototype_embeddings)
            prototype_emb_rec = self.map_proto_image_to_proto(prototypes_as_features) if not self.USE_INITIAL_PROTO_EMBS else prototype_embeddings
            prototype_emb_loss = F.mse_loss(prototype_emb_rec, prototype_embeddings)
            concept_logits_per_proto = self.map_proto_to_concepts(prototype_emb_rec)
            concept_probs_per_proto = torch.sigmoid(concept_logits_per_proto)
            prototypes_as_images = prototypes_as_features.unsqueeze(0) if prototypes_as_features.dim() == 2 else prototypes_as_features
        else:
            prototype_emb_loss = torch.tensor(0.0, device=self.device)
            concept_logits_per_proto = self.map_proto_to_concepts(prototype_embeddings)
            concept_probs_per_proto = torch.sigmoid(concept_logits_per_proto)
            prototypes_as_images = self.closest_masked_images.to(self.device) if self.closest_masked_images is not None else None

        # dot product
        # prototype_object_similarities = (object_embeddings.unsqueeze(2) * prototype_embeddings.unsqueeze(0).unsqueeze(1)).sum(dim=-1)
        
        # COSINE
        prototype_object_similarities = F.cosine_similarity(object_embeddings.unsqueeze(2), prototype_embeddings.unsqueeze(0).unsqueeze(1), dim=-1)*15  # batch, nb_objects, nb_proto  (cosine similarity)


        if interventions_mask is not None:
            intervened_true_concepts = (C * interventions_mask).unsqueeze(2)
            intervened_proto_concepts = concept_probs_per_proto.unsqueeze(0).unsqueeze(0) * interventions_mask.unsqueeze(2)
            allowed_prototypes = torch.all((intervened_true_concepts > 0.5) == (intervened_proto_concepts > 0.5), dim=-1)
            has_allowed_prototypes = torch.any(allowed_prototypes, dim=-1, keepdim=True)
            allowed_prototypes = torch.where(has_allowed_prototypes, allowed_prototypes, torch.ones_like(allowed_prototypes, dtype=torch.bool))
            prototype_object_similarities = prototype_object_similarities.masked_fill(~allowed_prototypes, float('-inf'))
            standard_interventions_mask2 = torch.where(has_allowed_prototypes, torch.zeros_like(interventions_mask), interventions_mask)
        else:
            standard_interventions_mask2 = None

        prototype_object_probs = torch.softmax(prototype_object_similarities, dim=-1)
        entropy = -(prototype_object_probs * (prototype_object_probs + 1e-10).log()).sum(dim=-1).mean()
        batch_scores = prototype_object_probs.sum(dim=0)
        batch_normalized = batch_scores / (batch_scores.sum(dim=1, keepdim=True) + 1e-10)
        mean_object_entropy = -(batch_normalized * (batch_normalized + 1e-10).log()).sum(dim=-1).mean()
        concept_probs_per_object = torch.einsum('bop,pc->boc', prototype_object_probs, concept_probs_per_proto)

        if standard_interventions_mask is not None:
            concept_probs_per_object = (concept_probs_per_object * (1 - standard_interventions_mask.unsqueeze(0)) + C * standard_interventions_mask.unsqueeze(0)).squeeze()
            if concept_probs_per_object.dim() == 2:
                concept_probs_per_object = concept_probs_per_object.unsqueeze(1)
        if standard_interventions_mask2 is not None:
            concept_probs_per_object = (concept_probs_per_object * (1 - standard_interventions_mask2) + C * standard_interventions_mask2)

        self.stored_activations = concept_probs_per_proto

        if self.concepts_to_task == "thresholding" or not self.training:
            if concept_probs_per_object.dim() == 2:
                concept_probs_per_object = concept_probs_per_object.unsqueeze(0)
            task_logits = self.task_predictor((concept_probs_per_object.flatten(start_dim=1) > 0.5).float())
            final_task = torch.softmax(task_logits, dim=-1) if self.is_cubEMB else torch.sigmoid(task_logits)
        elif self.concepts_to_task == "ground truth":
            task_logits = self.task_predictor(C.flatten(start_dim=1))
            final_task = torch.softmax(task_logits, dim=-1) if self.is_cubEMB else torch.sigmoid(task_logits)
        else:
            raise NotImplementedError(f"concepts_to_task {self.concepts_to_task} not implemented.")

        return final_task, concept_probs_per_object, entropy, {
            'prototype_probs_per_object': prototype_object_probs,
            'objects': None,
            'prototypes_as_images': prototypes_as_images,
            'z_posterior_mean': z_posterior_mean,
            'z_posterior_sdev': z_posterior_sdev,
            'z_posterior_logvar': z_posterior_logvar,
            'segmentation_loss': torch.tensor(0.0, device=self.device),
            'generated_M': None,
            'object_embeddings': object_embeddings,
            'prototype_emb_loss': prototype_emb_loss,
            'concept_probs_per_proto': concept_probs_per_proto,
            'concept_logits_per_proto': concept_logits_per_proto,
            'task_logits': task_logits,
            'batch_entropy': mean_object_entropy,
        }

    def forward(self, x, interventions_mask=None, standard_interventions_mask=None):
        # interventions_mask: (nb_objects, nb_concepts) binary tensor indicating which concepts to intervene on
        interventions_mask = interventions_mask.unsqueeze(0).expand(x[0].shape[0], -1, -1) if interventions_mask is not None else None
        standard_interventions_mask = standard_interventions_mask.unsqueeze(0).expand(x[0].shape[0], -1, -1) if standard_interventions_mask is not None else None
        # assert interventions_mask is None or standard_interventions_mask is None, "Probably unintended that you use both."

        if self.is_cubEMB:
            return self._forward_cubEMB(x, interventions_mask=interventions_mask, standard_interventions_mask=standard_interventions_mask)

        if self.use_pretrained_segmenter:
            segmentation, generated_M, C, Y, X = x
            generated_M_logits = None
            M = generated_M
            segmentation_loss = torch.tensor(0.0, device=self.device)
        elif self.segmentation_method == 'mask' and not self.training:
            X, M_test, C, Y = x
            # extract masks from x using the segmenter
            generated_M_logits = self.seg_model(X)  # (batch, nb_objects, H, W)
            generated_M = torch.sigmoid(generated_M_logits) 
            if not self.ALWAYS_USE_TRUE_MASKS:
                M = generated_M
            else:
                M = M_test

            segmentation_loss = calc_segmentation_loss(generated_M_logits, M_test)
        else:
            X, M, C, Y = x

            generated_M_logits = self.seg_model(X)  # (batch, nb_objects, H, W)
            generated_M = torch.sigmoid(generated_M_logits)
            segmentation_loss = calc_segmentation_loss(generated_M_logits, M)

        b, c, h, w = X.shape

        if self.use_pretrained_segmenter:
            object_embeddings = self.map_proto_image_to_proto(segmentation)  # (batch, nb_objects, proto_size)
        elif self.segmentation_method == 'mask':
            segmentation = get_objects_notraining(X, M)  # (batch, nb_objects, nb_channels, H, W)
            object_embeddings = self.map_proto_image_to_proto(segmentation)  # (batch, nb_objects, proto_size)
        else:
            raise NotImplementedError(f"Segmentation method {self.segmentation_method} not implemented.")
        
        ############# compute Z (posterior)
        z_posterior = self.latent_posterior.forward(X.unsqueeze(1))  # (b, 2*proto_size)  | squeeze is trick
        z_posterior = z_posterior.squeeze(1)
        z_posterior_mean, z_posterior_logvar = torch.chunk(z_posterior, 2, dim=-1)  # each (b, proto_size)
        z_posterior_sdev = torch.exp(0.5 * z_posterior_logvar)  # TODO: careful for numerical stability
        z_sampled = z_posterior_mean + z_posterior_sdev * torch.randn_like(z_posterior_sdev)  # (b, proto_size) | reparametrization trick

        if not self.swapped:
            prototype_embeddings = self.prototypes.weight  # (nb_proto, proto_size)
        else:
            # decode from the stacked_images
            prototype_embeddings = self.map_image_to_object_emb(self.closest_masked_images.to(self.device)).squeeze()  # (nb_proto, proto_size)
        nb_proto = prototype_embeddings.shape[0]

        z_sampled_expanded = z_sampled.unsqueeze(1).expand(-1, nb_proto, -1)  # (b, nb_proto, proto_size)

        ############# C pred
        if self.reconstruction and not self.swapped and not self.use_pretrained_autoencoder: # else the prototypes_as_images are already reconstructed from autoencoder branch above
            prototypes_as_images = reconstruction_util(self.map_proto_to_image, prototype_embeddings, torch.zeros((1, nb_proto, self.proto_size), device=self.device))

            prototype_emb_rec = self.map_proto_image_to_proto(prototypes_as_images) if not self.USE_INITIAL_PROTO_EMBS else prototype_embeddings  # (b*nb_proto, proto_size)
            # prototype_emb_rec = prototype_emb_rec.view(b, nb_proto, -1)  # (b, nb_proto, proto_size)
            # prototype_embeddings_batch = prototype_embeddings.unsqueeze(0).expand(b, -1, -1)  # (b, nb_proto, proto_size)
            # prototype_emb_loss = F.mse_loss(prototype_emb_rec, prototype_embeddings_batch)
            prototype_emb_loss = F.mse_loss(prototype_emb_rec, prototype_embeddings)

            concept_logits_per_proto = self.map_proto_to_concepts(prototype_emb_rec.view(-1, prototype_emb_rec.shape[-1])) # (nb_proto, nb_concepts)
            concept_probs_per_proto = torch.sigmoid(concept_logits_per_proto)  # (nb_proto, nb_concepts)
        else:
            prototype_emb_loss = torch.tensor(0.0).to(self.device)
            concept_logits_per_proto = self.map_proto_to_concepts(prototype_embeddings) # (nb_proto, nb_concepts)
            concept_probs_per_proto = torch.sigmoid(concept_logits_per_proto)  # (nb_proto, nb_concepts)

            prototypes_as_images = self.closest_masked_images.to(x[0].device) if self.closest_masked_images is not None else None

        # TODO: is this not a bug? We want to select using the reconstructed embeddings, no?
        prototype_object_similarities = (object_embeddings.unsqueeze(2) * prototype_embeddings.unsqueeze(0).unsqueeze(1)).sum(dim=-1)  # batch, nb_objects, nb_proto
        
        # COSINE:
        # prototype_object_similarities = F.cosine_similarity(object_embeddings.unsqueeze(2), prototype_embeddings.unsqueeze(0).unsqueeze(1), dim=-1)*15  # batch, nb_objects, nb_proto  (cosine similarity)

        if self.use_pretrained_autoencoder:
            prototype_object_similarities = prototype_object_similarities * 100

        # force certain concepts per prototype if specified
        if self.forced_concepts_per_proto_mask is not None and self.forced_concepts_per_proto is not None:
            concept_probs_per_proto = (concept_probs_per_proto * (1 - self.forced_concepts_per_proto_mask) + 
                                       self.forced_concepts_per_proto * self.forced_concepts_per_proto_mask)

        # mask similarities to prototypes with wrong concepts if concept interventions are provided
        standard_interventions_mask2 = None
        if interventions_mask is not None:            
            # mask not intervened concepts
            intervened_true_concepts = (C * interventions_mask).unsqueeze(2)  # (b, nb_objects, 1, nb_concepts)
            intervened_proto_concepts = (concept_probs_per_proto.unsqueeze(0).unsqueeze(0) * interventions_mask.unsqueeze(2))  # (b, 1, nb_proto, nb_concepts)

            # check which prototypes match the intervened concepts
            allowed_prototypes = torch.all((intervened_true_concepts > 0.5) == (intervened_proto_concepts > 0.5), dim=-1)  # (b, nb_objects, nb_proto)

            # Check if any prototypes are allowed BEFORE modifying the mask
            has_allowed_prototypes = torch.any(allowed_prototypes, dim=-1, keepdim=True)  # (b, nb_objects, 1)
            
            # if this masks all prototypes for an object, do not mask any of them (ignore the intervention)
            allowed_prototypes = torch.where(has_allowed_prototypes, 
                                             allowed_prototypes, 
                                             torch.ones_like(allowed_prototypes, dtype=torch.bool))
            # TODO: alternative: just intervene on the c_pred? (goes o.o.d. but we probably don't care?)
            # for objects with no allowed, set standard_intervention_mask to 1 for those concepts (so that c_pred is set to C for these objects)
            standard_interventions_mask2 = torch.where(has_allowed_prototypes,
                                                      torch.zeros_like(interventions_mask),
                                                      interventions_mask)  # (b, nb_objects, nb_concepts)

            # set similarities to -Inf for prototypes that do not match the intervened concepts. They will become 0 due to the softmax.
            prototype_object_similarities = prototype_object_similarities.masked_fill(~allowed_prototypes, float('-inf'))
        
        if len(self.masked_prototypes) > 0:
            for proto_idx in self.masked_prototypes:
                prototype_object_similarities[:, :, proto_idx] = float('-inf')

        prototype_object_probs = torch.softmax(prototype_object_similarities, dim=-1)  # (b, nb_objects, nb_proto)

        entropy = -(prototype_object_probs * (prototype_object_probs + 1e-10).log()).sum(dim=-1).mean()
        batch_scores =  prototype_object_probs.sum(dim=0)   # nb_objects, nb_proto
        batch_normalized = batch_scores / (batch_scores.sum(dim=1, keepdim=True) + 1e-10)  # nb_objects, nb_proto)
        mean_object_entropy = -(batch_normalized * (batch_normalized + 1e-10).log()).sum(dim=-1).mean()



        concept_probs_per_object = torch.einsum('bop,pc->boc', prototype_object_probs, concept_probs_per_proto)  # (b, nb_obj, nb_concepts)

        # intervene on this using standard_interventions_mask if provided
        if standard_interventions_mask is not None:
            concept_probs_per_object = (concept_probs_per_object * (1 - standard_interventions_mask.unsqueeze(0)) + C * standard_interventions_mask.unsqueeze(0)).squeeze()
            if concept_probs_per_object.dim() == 2:
                concept_probs_per_object = concept_probs_per_object.unsqueeze(1)
        if standard_interventions_mask2 is not None:
            concept_probs_per_object = (concept_probs_per_object * (1 - standard_interventions_mask2) + C * standard_interventions_mask2)

        self.stored_activations = concept_probs_per_proto

        ############# Y pred

        if self.concepts_to_task == "thresholding" or not self.training:
            if concept_probs_per_object.dim() == 2:
                concept_probs_per_object = concept_probs_per_object.unsqueeze(0)
            task_logits = self.task_predictor((concept_probs_per_object.flatten(start_dim=1) > 0.5).float())
            final_task = torch.softmax(task_logits, dim=-1) if self.is_multiclass_task else torch.sigmoid(task_logits)  # (b, nb_tasks)
            if False:  # this is the more probabilistic way, but too expensive + we won't do this for other methods, so it would be unfair
                concept_probs_per_proto_thresholded = (concept_probs_per_proto > 0.5).float()  # (nb_proto, nb_concepts)
                concept_probs_per_proto_tuple = []
                for i in range(self.nb_possibleobjects):
                    shape = [1] * self.nb_possibleobjects + [self.nb_concepts]  # shape: (1, 1, ..., nb_concepts)
                    shape[i] = nb_proto  # shape: (1, ..., nb_proto, ..., 1, nb_concepts)
                    concept_probs_per_proto_tuple.append(concept_probs_per_proto_thresholded.view(*shape))  # each element: (1, ..., nb_proto, ..., 1, nb_concepts)
                concept_probs_per_proto_tuple = torch.broadcast_tensors(*concept_probs_per_proto_tuple)  # each element: (nb_proto, nb_proto, ..., nb_proto, nb_concepts)
                concept_probs_per_proto_tuple = torch.cat(concept_probs_per_proto_tuple, dim=-1)  # (nb_proto, nb_proto, ..., nb_proto, nb_possibleobjects * nb_concepts)

                task_logits_per_proto_tuple =  self.task_predictor(concept_probs_per_proto_tuple) # (nb_proto, nb_proto, ..., nb_proto, nb_tasks)
                task_per_proto_tuple = torch.sigmoid(task_logits_per_proto_tuple)  # (nb_proto, nb_proto, ..., nb_proto, nb_tasks)

                prototype_tuples_probs = []
                for k in range(self.nb_possibleobjects):
                    shape = [prototype_object_probs.shape[0]] + [1]*self.nb_possibleobjects  # (batch, 1, 1, ..., 1)
                    shape[k+1] = nb_proto   # +1 because batch dim is first | (batch, 1, ..., nb_proto, ..., 1)
                    prototype_tuples_probs.append(prototype_object_probs[:, k].view(*shape))  # each element: (batch, 1, ..., nb_proto, ..., 1)
                prototype_tuples_probs = torch.broadcast_tensors(*prototype_tuples_probs)  # each element: (batch, nb_proto, nb_proto, ..., nb_proto)
                prototype_tuples_probs = torch.prod(torch.stack(prototype_tuples_probs), dim=0)  # (batch, nb_proto, nb_proto, ..., nb_proto)

                final_task = (prototype_tuples_probs.flatten(start_dim=1, end_dim=self.nb_possibleobjects).unsqueeze(-1)  # (b, nb_proto**nb_possibleobjects, 1)
                            * task_per_proto_tuple.flatten(start_dim=0, end_dim=self.nb_possibleobjects-1).unsqueeze(0)   # (1, nb_proto**nb_possibleobjects, nb_tasks)
                            ).sum(dim=1)  #  (b, nb_tasks)
        elif self.concepts_to_task == "ground truth":
            task_logits_gt = self.task_predictor(C.flatten(start_dim=1))
            final_task = torch.softmax(task_logits_gt, dim=-1) if self.is_multiclass_task else torch.sigmoid(task_logits_gt)  # (b, nb_tasks)
        else:
            raise NotImplementedError(f"concepts_to_task {self.concepts_to_task} not implemented.")

        task_logits_out = task_logits if (self.concepts_to_task == "thresholding" or not self.training) else task_logits_gt
        return final_task, concept_probs_per_object, entropy, {'prototype_probs_per_object': prototype_object_probs, 
                                                               'objects': segmentation, 'prototypes_as_images': prototypes_as_images, 
                                                               'z_posterior_mean': z_posterior_mean, 'z_posterior_sdev': z_posterior_sdev,
                                                               'z_posterior_logvar': z_posterior_logvar,
                                                               'segmentation_loss': segmentation_loss,
                                                               'generated_M': generated_M,
                                                               'object_embeddings': object_embeddings,
                                                               'prototype_emb_loss': prototype_emb_loss,
                                                               'concept_probs_per_proto': concept_probs_per_proto,
                                                               'concept_logits_per_proto': concept_logits_per_proto,
                                                               'batch_entropy': mean_object_entropy,
                                                               'task_logits': task_logits_out,
                                                               },

    def get_closest_prototype(self, object_emb):
        pass  # TODO

    def swap_prototypes_with_instances(self, data_loader):
        if self.is_cubEMB:
            with torch.no_grad():
                prototype_embeddings = self.prototypes.weight
                nb_proto = prototype_embeddings.shape[0]
                best_probs = torch.full((nb_proto,), float("-inf"), device=self.device)
                closest_feature_vectors = [None] * nb_proto

                for batch in data_loader:
                    batch = tuple(item.to(self.device) if torch.is_tensor(item) else item for item in batch)
                    x = batch[0]
                    if x.dim() > 2:
                        x = x.flatten(start_dim=1)
                    object_embeddings = self.map_image_to_object_emb(x)
                    prototype_object_similarities = (object_embeddings.unsqueeze(1) * prototype_embeddings.unsqueeze(0)).sum(dim=-1)
                    prototype_object_probs = torch.softmax(prototype_object_similarities, dim=-1)

                    batch_best_probs, batch_best_indices = torch.max(prototype_object_probs, dim=0)
                    for proto_idx in range(nb_proto):
                        if batch_best_probs[proto_idx] > best_probs[proto_idx]:
                            best_probs[proto_idx] = batch_best_probs[proto_idx]
                            closest_feature_vectors[proto_idx] = x[batch_best_indices[proto_idx]].detach().cpu()

                assert None not in closest_feature_vectors, "Some prototypes did not find any closest CUB feature vectors. This is a bug."
                closest_feature_vectors = torch.stack([vec for vec in closest_feature_vectors]).to(self.device)
                self.closest_masked_images = closest_feature_vectors.unsqueeze(0)
                self.prototypes.weight.data = self.map_image_to_object_emb(closest_feature_vectors).detach()
                self.swapped = True
            return

        # loop over the dataloader. For each prototype, we want to take the object embedding the closest to that prototype embedding.
        # Afterwards, we want to replace each prototype embedding with its corresponding closest object embedding
        # Also, store the corresponding masked images that generated these object embeddings in an attribute for visualization later
        
        with torch.no_grad():
                # Get prototype embeddings
                prototype_embeddings = self.prototypes.weight  # (nb_proto, proto_size)
                nb_proto = prototype_embeddings.shape[0]
                
                # Initialize storage for closest object embeddings and images
                closest_object_embeddings = torch.zeros_like(prototype_embeddings)
                closest_masked_images = [None] * nb_proto  # Initialize with correct size
                
                # Loop over dataloader to find top-1 object embeddings
                best_probs, best_objects = tokp_objects_for_prototype(self, data_loader, top_k=1)
                
                # Concatenate results from all batches
                best_probs = torch.cat(best_probs, dim=0)  # (total_topk, nb_proto)
                best_objects = torch.cat(best_objects, dim=0)  # (total_topk, nb_proto, nb_channels, H, W)

                top1_prob = None
                
                for proto_idx in range(nb_proto):
                    # Get the top-1 object for this prototype
                    # take most likely using argmax
                    top1_idx = torch.argmax(best_probs[:, proto_idx])
                    top1_prob = best_probs[top1_idx, proto_idx]
                    top1_object = best_objects[top1_idx, proto_idx]  # (nb_channels, H, W)
                    
                    closest_object_embeddings[proto_idx] = self.map_image_to_object_emb(top1_object.unsqueeze(0).unsqueeze(1).to(self.device)).squeeze(0)
                    closest_masked_images[proto_idx] = top1_object.cpu()
            
        # Replace prototype embeddings with closest object embeddings
        self.prototypes.weight.data = closest_object_embeddings.detach()
            
        # Store masked images for visualization
        assert None not in closest_masked_images, "Some prototypes did not find any closest masked images. This is a bug."
        self.closest_masked_images = torch.stack([img for img in closest_masked_images]).to(self.device).unsqueeze(0)  # (1, nb_proto, nb_channels, H, W)

        self.swapped = True

        # self.seg_model.requires_grad_(False)
        # self.map_image_to_object_emb.requires_grad_(False)
        # self.map_proto_to_image.requires_grad_(False)
            
    def get_prototypes_as_images(self):
        if not self.reconstruction:
            return None

        if self.closest_masked_images is None:
            prototype_embeddings = self.prototypes.weight  # (nb_proto, proto_size)
            prototypes_as_images = reconstruction_util(self.map_proto_to_image, prototype_embeddings, torch.zeros((1, prototype_embeddings.shape[0], self.proto_size), device=self.device))
            # prototypes_as_images = self.map_proto_to_image(self.combine_z_and_proto_emb(torch.cat([prototype_embeddings, self.latent_prior_mean*torch.ones_like(prototype_embeddings)], dim=-1)).unsqueeze(0))  # (nb_proto, 3, H, W)
            return prototypes_as_images.squeeze()
        else:
            return self.closest_masked_images
    
    def hungarian_loss(self, c_true, c_pred, nb_objects):
        # c_true: (b, nb_objects, nb_concepts)
        # c_pred: (b, nb_objects, nb_concepts)
        total_loss = 0.0
        b = c_true.shape[0]
        for i in range(b):
            cost_matrix = torch.zeros((nb_objects, nb_objects), device=self.device)
            for j in range(nb_objects):
                for k in range(nb_objects):
                    cost_matrix[j, k] = binary_cross_entropy(c_pred[i, j], c_true[i, k], reduction='mean')
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            loss = 0.0
            for j in range(len(row_ind)):
                loss += cost_matrix[row_ind[j], col_ind[j]]
            total_loss += loss
        return total_loss / b

    def training_step(self, batch, batch_idx):
        if self.decay_lam_entropy:
            try:
                max_epochs = self.trainer.max_epochs
                curr_epoch = self.current_epoch
            except RuntimeError:
                # If model is detached from trainer (e.g. loaded from checkpoint in main.py)
                max_epochs = 100
                curr_epoch = max_epochs
            self.lam = get_current_lam_entropy(curr_epoch, max_epochs, self.lam_start)
        else:
            self.lam = self.lam_start

        x, m, c_true, y_true = batch[:4]

        # ==== New, generate random interventions_mask for this batch ====
        # interventions_mask: (nb_objects, nb_concepts) binary tensor indicating which concepts to intervene on
        if self.training and self.intv_prob > 0:
            interventions_mask = (torch.rand((self.nb_possibleobjects, self.nb_concepts), device=self.device) < self.intv_prob).float()
            # interventions_mask = None  # TODO temp disable
        else:
            interventions_mask = None
        # =================================================================

        y_pred, c_pred, e, extras = self.forward(batch, interventions_mask=interventions_mask)

        self.last_prototypes_as_images_posterior = extras.get('prototypes_as_images').detach() if extras.get('prototypes_as_images') is not None else None
        self.last_prototype_probs_per_object = extras['prototype_probs_per_object'].detach()
        self.last_X = (batch[4] if len(batch) > 4 else x).detach()  # Store original images for visualization
        batch_entropy = extras['batch_entropy']

        prototype_emb_loss = extras['prototype_emb_loss']
        self.log('prototype_emb_loss', prototype_emb_loss, prog_bar=False, on_step=False, on_epoch=True)

        if self.segmentation_method in ['mask','slot_attention']:
            c_true = c_true.view(c_true.shape[0], self.nb_possibleobjects, -1) # reshape to nb_objects
        else:
            raise NotImplementedError(f"Segmentation method {self.segmentation_method} not implemented.")

        rescaling_c_pred = eps + (1 - 2 * eps) * c_pred

        assert torch.all(rescaling_c_pred > 0), torch.min(rescaling_c_pred)
        assert torch.all(rescaling_c_pred < 1), torch.max(rescaling_c_pred)

        if self.is_multiclass_task:
            y_logits = extras['task_logits']
            y_target = y_true.argmax(dim=-1) if y_true.dim() > 1 else y_true.long()
            y_loss = F.cross_entropy(y_logits, y_target)
        else:
            rescaling_y_pred = eps + (1 - 2 * eps) * y_pred
            assert torch.all(rescaling_y_pred > 0), torch.min(rescaling_y_pred)
            assert torch.all(rescaling_y_pred < 1), torch.max(rescaling_y_pred)
            y_loss = binary_cross_entropy(rescaling_y_pred, y_true)
        
        if self.segmentation_method == 'mask':
            if self.pos_weights is not None:
                self.pos_weights = self.pos_weights.to(self.device)
                batch_weights = torch.where(c_true == 1, self.pos_weights, torch.tensor(1.0, device=c_true.device))
                c_loss = binary_cross_entropy(rescaling_c_pred, c_true, weight=batch_weights)
            else:
                c_loss = binary_cross_entropy(rescaling_c_pred, c_true)  # binary cross entropy
        elif self.segmentation_method == 'slot_attention':
            raise NotImplementedError("Hungarian loss for slot attention not implemented yet.")
            c_loss = self.hungarian_loss(c_true, rescaling_c_pred, self.nb_possibleobjects)
        else:
            raise NotImplementedError(f"Segmentation method {self.segmentation_method} not implemented.")

        if self.is_multiclass_task and not self.is_cubEMB:
            # CUB pre-segmented: multiclass task with balanced accuracy support
            y_logits = extras['task_logits']
            y_target = y_true.argmax(dim=-1) if y_true.dim() > 1 else y_true.long()
            y_acc = (y_logits.argmax(dim=-1) == y_target).float().mean()
            c_acc = ((c_pred > 0.5).float() == c_true).float().mean() if self.segmentation_method == 'mask' else torch.tensor(0.0, device=self.device)
            self.log('train_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)
            if self.segmentation_method == 'mask':
                self.log('train_concept_acc', c_acc, prog_bar=True, on_step=False, on_epoch=True)
        elif self.is_cubEMB:
            y_acc = (y_logits.argmax(dim=-1) == y_target).float().mean()
            self.log('train_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)
            if self.use_balanced_accuracy:
                self.train_c_acc(c_pred > 0.5, c_true)
            else:
                c_acc = ((c_pred > 0.5).float() == c_true).float().mean()
                self.log('train_concept_acc', c_acc, prog_bar=True, on_step=False, on_epoch=True)
        else:
            y_acc = ((y_pred > 0.5) == y_true).float().mean()
            self.log('train_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)
            if self.use_balanced_accuracy and self.segmentation_method == 'mask':
                self.train_c_acc(c_pred > 0.5, c_true)
            else:
                c_acc = ((c_pred > 0.5) == c_true).float().mean() if self.segmentation_method == 'mask' else torch.tensor(0.0, device=c_true.device)
                if self.segmentation_method == 'mask':
                    self.log('train_concept_acc', c_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.log('train_y_loss', y_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_c_loss', c_loss, prog_bar=True, on_step=False, on_epoch=True)
        

        if self.reconstruction and extras.get('prototypes_as_images') is not None and not self.use_pretrained_autoencoder:
            prototype_probs_per_object = extras['prototype_probs_per_object']  # (b, nb_objects, nb_proto)

            if self.is_cubEMB:
                prototype_features = extras['prototypes_as_images']
                if prototype_features.dim() == 2:
                    prototype_features = prototype_features.unsqueeze(0)
                input_features = x.flatten(start_dim=1) if x.dim() > 2 else x
                input_features = input_features.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, feat_dim)
                prototype_features = prototype_features.expand(input_features.shape[0], -1, -1).unsqueeze(1)  # (b, 1, nb_proto, feat_dim)
                input_features = input_features.expand(-1, prototype_features.shape[1], prototype_features.shape[2], -1)
                recons_error = F.mse_loss(prototype_features, input_features, reduction='none').mean(dim=-1)  # (b, 1, nb_proto)
            else:
                prototypes_as_images = extras['prototypes_as_images']  # (b, nb_proto, 3, H, W)
                # if it does not have a batch, add it
                if prototypes_as_images.dim() == 4:
                    prototypes_as_images = prototypes_as_images.unsqueeze(0)

                if self.use_pretrained_segmenter:
                    masked_images = x.unsqueeze(2)  # (b, nb_objects, 1, 3, H, W)
                else:
                    m = m.unsqueeze(2)  # (b, nb_objects, 1, H, W)
                    x = x.unsqueeze(1)  # (b, 1, 3, H, W)

                    masked_images = m * x  # (b, nb_objects, 3, H, W)
                    masked_images = masked_images.unsqueeze(2)  # (b, nb_objects, 1, 3, H, W)

                prototypes_as_images = prototypes_as_images.unsqueeze(1)  # (b, 1, nb_proto, 3, H, W)

                recons_error = F.mse_loss(prototypes_as_images, masked_images, reduction='none')  # (b, nb_objects, nb_proto, 3, H, W)
                recons_error = recons_error.mean(dim=[3, 4, 5])  # (b, nb_objects, nb_proto)

                # torch distributions Normal
                logprob_rec = torch.distributions.Normal(loc=prototypes_as_images, scale=1.0).log_prob(masked_images).sum(dim=[3, 4, 5])  # (b, nb_objects, nb_proto)

            recons_loss = (prototype_probs_per_object * recons_error).sum(dim=-1).mean()

            self.log('train_recons_loss', recons_loss, prog_bar=True, on_step=False, on_epoch=True)

            # KL divergence between posterior and prior
            z_posterior_logvar = extras['z_posterior_logvar']  # (b, proto_size)

            # TODO: careful for numerical stability
            # kl_divergence = calculate_kl_loss(z_posterior_mean, z_posterior_logvar)

            # self.log('train_kl_divergence', kl_divergence, prog_bar=True, on_step=False, on_epoch=True)
        else:
            recons_loss = 0.0
            recons_error = torch.tensor(0.0).to(x.device)

        self.log('batch_entropy', batch_entropy, prog_bar=False, on_step=False, on_epoch=True)
        self.log('entropy', e, prog_bar=False, on_step=False, on_epoch=True)
        self.log('segmentation_loss', extras['segmentation_loss'] if extras['segmentation_loss'] is not None else 0.0, prog_bar=False, on_step=False, on_epoch=True)

        if self.use_correct_loss:
            yloss_w = 1
            prototype_probs_per_object_bop = extras["prototype_probs_per_object"]
            concept_probs_per_proto_bopc = extras["concept_probs_per_proto"].unsqueeze(0).unsqueeze(0)
            # if self.current_epoch > 1 and batch_idx == 0:
            #     input(extras["concept_probs_per_proto"][0])
            recons_error_bop = recons_error  # (b, nb_objects, nb_proto)

            # concept_probs_correct_per_proto_bopc = concept_probs_per_proto_bopc * c_true.unsqueeze(2) + (1 - concept_probs_per_proto_bopc) * (1 - c_true.unsqueeze(2))  # (b, nb_objects, nb_proto, nb_concepts)
            concept_logprobs_correct_per_proto_bopc = -binary_cross_entropy(concept_probs_per_proto_bopc.expand(c_true.shape[0], c_true.shape[1], -1, -1), c_true.unsqueeze(2).expand(-1, -1, concept_probs_per_proto_bopc.shape[2], -1), reduction='none')  # (b, nb_objects, nb_proto, nb_concepts)

            if self.pos_weights is not None:
                pos_weights_bopc = self.pos_weights.view(1, self.nb_possibleobjects, 1, self.nb_concepts)  # (1, nb_objects, 1, nb_concepts)
                weights = torch.where(c_true.unsqueeze(2).expand(-1, -1, concept_probs_per_proto_bopc.shape[2], -1) == 1, pos_weights_bopc, torch.ones((1,), device=self.device))
            else:
                weights = 1

            NEW_LOSS =  (self.lam_prototype_emb * prototype_emb_loss - self.lam *  e - self.lam_batch_entropy * batch_entropy + yloss_w* y_loss - torch.sum(prototype_probs_per_object_bop * (1/self.nb_concepts * (weights * concept_logprobs_correct_per_proto_bopc).sum(dim=-1) - self.lam_reconstruction * recons_error_bop), dim=2).mean(dim=1)).mean()

            NEW_LOSS += self.lam_segmentation * extras['segmentation_loss']
            self.log('concept_loss_real', (- torch.sum(prototype_probs_per_object_bop * (concept_logprobs_correct_per_proto_bopc.sum(dim=-1)), dim=2).sum(dim=1)).mean(), prog_bar=False, on_step=False, on_epoch=True)

            self.log('total_loss', NEW_LOSS, prog_bar=False, on_step=False, on_epoch=True)
            return NEW_LOSS
        else:
            if self.lam > 0 or self.lam_batch_entropy > 0:
                total_loss = y_loss + c_loss - self.lam * e - self.lam_batch_entropy * batch_entropy + self.lam_reconstruction * recons_loss + self.lam_prototype_emb * prototype_emb_loss  # careful, keep val_loss up-to-date
                if extras['segmentation_loss'] is not None:
                    total_loss += self.lam_segmentation * extras['segmentation_loss']
                
                self.log('total_loss', total_loss, prog_bar=False, on_step=False, on_epoch=True)
                return total_loss
            if self.swapped:
                self.log('train_total_loss_after_swap', total_loss, prog_bar=False, on_step=False, on_epoch=True)
                self.log('train_total_loss_before_swap', 1000, prog_bar=False, on_step=False, on_epoch=True)
            else:
                self.log('train_total_loss_before_swap', total_loss, prog_bar=False, on_step=False, on_epoch=True)
                self.log('train_total_loss_after_swap', 1000, prog_bar=False, on_step=False, on_epoch=True)
        
        self.log('total_loss', total_loss, prog_bar=False, on_step=False, on_epoch=True)
        return total_loss

    def _log_train_epoch_metrics(self):
        if self.use_balanced_accuracy and self.segmentation_method == 'mask':
            self.log('train_concept_acc_epoch', self.train_c_acc.compute(), prog_bar=True)
            self.train_c_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, m, c_true, y_true = batch[:4]

        y_pred, c_pred, e, extras = self.forward(batch)

        prototype_emb_loss = extras['prototype_emb_loss']
        batch_entropy = extras['batch_entropy']
        self.log('val_prototype_emb_loss', prototype_emb_loss, prog_bar=False, on_step=False, on_epoch=True)

        if self.segmentation_method in ['mask','slot_attention']:
            c_true = c_true.view(c_true.shape[0], self.nb_possibleobjects, -1) # reshape to nb_objects
        else:
            raise NotImplementedError(f"Segmentation method {self.segmentation_method} not implemented.")

        rescaling_c_pred = eps + (1 - 2 * eps) * c_pred

        if self.is_multiclass_task:
            y_logits = extras['task_logits']
            y_target = y_true.argmax(dim=-1) if y_true.dim() > 1 else y_true.long()
            y_loss = F.cross_entropy(y_logits, y_target)
        else:
            rescaling_y_pred = eps + (1 - 2 * eps) * y_pred
            y_loss = binary_cross_entropy(rescaling_y_pred, y_true)
        if self.segmentation_method == 'mask':
            if self.pos_weights is not None:
                batch_weights = torch.where(c_true == 1, self.pos_weights, torch.tensor(1.0, device=self.device))
                c_loss = binary_cross_entropy(rescaling_c_pred, c_true, weight=batch_weights)
            else:
                c_loss = binary_cross_entropy(rescaling_c_pred, c_true)  # binary cross entropy
        elif self.segmentation_method == 'slot_attention':
            raise NotImplementedError("Hungarian loss for slot attention not implemented yet.")
            c_loss = self.hungarian_loss(c_true, rescaling_c_pred, self.nb_possibleobjects)
        else:
            raise NotImplementedError(f"Segmentation method {self.segmentation_method} not implemented.")
            
        if self.is_multiclass_task and not self.is_cubEMB:
            y_acc = (y_logits.argmax(dim=-1) == y_target).float().mean()
            c_acc = ((c_pred > 0.5).float() == c_true).float().mean() if self.segmentation_method == 'mask' else torch.tensor(0.0, device=self.device)
            self.log('val_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)
            if self.segmentation_method == 'mask':
                self.log('val_concept_acc', c_acc, prog_bar=True, on_step=False, on_epoch=True)
        elif self.is_cubEMB:
            y_acc = (y_logits.argmax(dim=-1) == y_target).float().mean()
            self.log('val_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)
            if self.use_balanced_accuracy:
                self.val_c_acc(c_pred > 0.5, c_true)
            else:
                c_acc = ((c_pred > 0.5).float() == c_true).float().mean()
                self.log('val_concept_acc', c_acc, prog_bar=True, on_step=False, on_epoch=True)
        else:
            y_acc = ((y_pred > 0.5) == y_true).float().mean()
            self.log('val_task_acc', y_acc, prog_bar=False, on_step=False, on_epoch=True)
            if self.use_balanced_accuracy and self.segmentation_method == 'mask':
                self.val_c_acc(c_pred > 0.5, c_true)
            else:
                c_acc = ((c_pred > 0.5) == c_true).float().mean() if self.segmentation_method == 'mask' else torch.tensor(0.0, device=self.device)
                if self.segmentation_method == 'mask':
                    self.log('val_concept_acc', c_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        # also recon loss
        if self.reconstruction and extras.get('prototypes_as_images') is not None and not self.use_pretrained_autoencoder:

            prototype_probs_per_object = extras['prototype_probs_per_object']  # (b, nb_objects, nb_proto)

            if self.is_cubEMB:
                prototype_features = extras['prototypes_as_images']
                if prototype_features.dim() == 2:
                    prototype_features = prototype_features.unsqueeze(0)
                input_features = x.flatten(start_dim=1) if x.dim() > 2 else x
                input_features = input_features.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, feat_dim)
                prototype_features = prototype_features.expand(input_features.shape[0], -1, -1).unsqueeze(1)  # (b, 1, nb_proto, feat_dim)
                input_features = input_features.expand(-1, prototype_features.shape[1], prototype_features.shape[2], -1)
                recons_error = F.mse_loss(prototype_features, input_features, reduction='none').mean(dim=-1)  # (b, 1, nb_proto)
            else:
                m = extras['generated_M']

                prototypes_as_images = extras['prototypes_as_images']  # (b, nb_proto, 3, H, W)

                # if it does not have a batch, add it
                if prototypes_as_images.dim() == 4:
                    prototypes_as_images = prototypes_as_images.unsqueeze(0)

                if self.use_pretrained_segmenter:
                    masked_images = x.unsqueeze(2)  # (b, nb_objects, 1, 3, H, W)
                else:
                    m = m.unsqueeze(2)  # (b, nb_objects, 1, H, W)
                    x = x.unsqueeze(1)  # (b, 1, 3, H, W)

                    masked_images = m * x  # (b, nb_objects, 3, H, W)
                    masked_images = masked_images.unsqueeze(2)  # (b, nb_objects, 1, 3, H, W)

                prototypes_as_images = prototypes_as_images.unsqueeze(1)  # (b, 1, nb_proto, 3, H, W)

                recons_error = F.mse_loss(prototypes_as_images, masked_images, reduction='none')  # (b, nb_objects, nb_proto, 3, H, W)
                recons_error = recons_error.mean(dim=[3, 4, 5])  # (b, nb_objects, nb_proto)

            recons_loss = (prototype_probs_per_object * recons_error).sum(dim=-1).mean()

            self.log('val_recons_loss', recons_loss, on_step=False, on_epoch=True)
        else:
            recons_loss = 0.0
            recons_error = torch.tensor(0.0).to(x.device)

        self.log('val_y_loss', y_loss, on_step=False, on_epoch=True)
        self.log('val_c_loss', c_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        self.log('val_entropy', e, prog_bar=False, on_step=False, on_epoch=True)
        

        if extras['segmentation_loss'] is not None:
            self.log('val_segmentation_loss', extras['segmentation_loss'], prog_bar=False, on_step=False, on_epoch=True)

        # total_loss = y_loss + c_loss + self.lam_reconstruction * recons_loss + self.lam_segmentation *  extras['segmentation_loss'] + self.lam_prototype_emb * prototype_emb_loss  # careful, keep up-to-date
        # if self.pgcm:
        #     if self.swapped:
        #         self.log('val_total_loss_after_swap', total_loss, prog_bar=False, on_step=False, on_epoch=True)
        #         self.log('val_total_loss_before_swap', 1000, prog_bar=False, on_step=False, on_epoch=True)
        #     else:
        #         self.log('val_total_loss_before_swap', total_loss, prog_bar=False, on_step=False, on_epoch=True)
        #         self.log('val_total_loss_after_swap', 1000, prog_bar=False, on_step=False, on_epoch=True)
        
        # self.log('val_total_loss', total_loss, prog_bar=False, on_step=False, on_epoch=True)
        if self.use_correct_loss:
            prototype_probs_per_object_bop = extras["prototype_probs_per_object"]
            concept_probs_per_proto_bopc = extras["concept_probs_per_proto"].unsqueeze(0).unsqueeze(0)
            # if self.current_epoch > 1 and batch_idx == 0:
            #     input(extras["concept_probs_per_proto"][0])
            recons_error_bop = recons_error  # (b, nb_objects, nb_proto)

            # concept_probs_correct_per_proto_bopc = concept_probs_per_proto_bopc * c_true.unsqueeze(2) + (1 - concept_probs_per_proto_bopc) * (1 - c_true.unsqueeze(2))  # (b, nb_objects, nb_proto, nb_concepts)
            concept_logprobs_correct_per_proto_bopc = -binary_cross_entropy(concept_probs_per_proto_bopc.expand(c_true.shape[0], c_true.shape[1], -1, -1), c_true.unsqueeze(2).expand(-1, -1, concept_probs_per_proto_bopc.shape[2], -1), reduction='none')  # (b, nb_objects, nb_proto, nb_concepts)

            # create a weigth matrix of shape (batch ,objects, prototypes, concepts) with value 10 when c_true = 1 and 1 otherwise
            if self.pos_weights is not None:
                pos_weights_bopc = self.pos_weights.view(1, self.nb_possibleobjects, 1, self.nb_concepts)  # (1, nb_objects, 1, nb_concepts)
                weights = torch.where(c_true.unsqueeze(2).expand(-1, -1, concept_probs_per_proto_bopc.shape[2], -1) == 1, pos_weights_bopc, torch.ones((1,), device=self.device))
            else:
                weights = 1

            NEW_LOSS =  (- self.lam * e - self.lam_batch_entropy * batch_entropy + y_loss - torch.sum(prototype_probs_per_object_bop * (1/self.nb_concepts * (weights * concept_logprobs_correct_per_proto_bopc).sum(dim=-1) - self.lam_reconstruction * recons_error_bop), dim=2).mean(dim=1)).mean()
            NEW_LOSS += self.lam_segmentation * extras['segmentation_loss']
            self.log('val_concept_loss_real', (- torch.sum(prototype_probs_per_object_bop * (concept_logprobs_correct_per_proto_bopc.sum(dim=-1)), dim=2).sum(dim=1)).mean(), prog_bar=False, on_step=False, on_epoch=True)

            self.log('val_total_loss', NEW_LOSS, prog_bar=False, on_step=False, on_epoch=True)
            if self.swapped:
                self.log('val_total_loss_after_swap', NEW_LOSS, prog_bar=False, on_step=False, on_epoch=True)
                self.log('val_total_loss_before_swap', 1000, prog_bar=False, on_step=False, on_epoch=True)
            else:
                self.log('val_total_loss_before_swap', NEW_LOSS, prog_bar=False, on_step=False, on_epoch=True)
                self.log('val_total_loss_after_swap', 1000, prog_bar=False, on_step=False, on_epoch=True)
        else:
            total_loss = y_loss + c_loss + self.lam_reconstruction * recons_loss + self.lam_segmentation *  extras['segmentation_loss'] + self.lam_prototype_emb * prototype_emb_loss  # careful, keep up-to-date
            self.log('val_total_loss', total_loss, prog_bar=False, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        if self.use_balanced_accuracy and self.segmentation_method == 'mask':
            self.log('val_concept_acc_epoch', self.val_c_acc.compute(), prog_bar=True)
            self.val_c_acc.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.FIXED_LR:
            return optimizer

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
    
    def on_save_checkpoint(self, checkpoint):
        """Save custom attributes to checkpoint"""
        # State related to prototype swapping
        checkpoint['swapped'] = self.swapped
        if self.closest_masked_images is not None:
            checkpoint['closest_masked_images'] = self.closest_masked_images.cpu()
        else:
            checkpoint['closest_masked_images'] = None
        
        # User-specified masked prototypes
        checkpoint['masked_prototypes'] = self.masked_prototypes
        
        # Training activations for statistics/visualization
        if self.stored_activations is not None:
            checkpoint['stored_activations'] = self.stored_activations.cpu()
        else:
            checkpoint['stored_activations'] = None
        
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        """Restore custom attributes from checkpoint"""
        # Prototype swapping state
        self.swapped = checkpoint.get('swapped', False)
        closest_masked_images = checkpoint.get('closest_masked_images', None)
        if closest_masked_images is not None:
            self.closest_masked_images = closest_masked_images.to(self.device)
        else:
            self.closest_masked_images = None
        
        # User-specified masked prototypes
        self.masked_prototypes = checkpoint.get('masked_prototypes', [])
        
        # Training activations for statistics/visualization
        stored_activations = checkpoint.get('stored_activations', None)
        if stored_activations is not None:
            self.stored_activations = stored_activations.to(self.device)
        else:
            self.stored_activations = None
    
    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)

    def on_train_epoch_end(self):
        self._log_train_epoch_metrics()
        print()
        if self.is_cubEMB and self.current_epoch % self.plot_frequency == 0:
            used_indices = get_used_prototypes_indices(self, self.val_loader)
            print(f"Used {len(used_indices)} prototypes out of {self.prototypes.weight.shape[0]}")
            print(f"Used prototypes: {used_indices}")
        if not self.is_cubEMB and self.current_epoch % self.plot_frequency == 0:
            self.generate_plots(logger=None)

        if self.current_epoch == self.trainer.max_epochs // 2:

            # save the best on validation for the current model before the swapping
            print("Saving model before prototype swapping...")
            save_path = os.path.join(self.output_folder, f"model_before_swapping_epoch_{self.current_epoch}.pt")
            torch.save(self.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            print("Swapping prototypes with closest instances from training set...")
            self.swap_prototypes_with_instances(self.train_loader)
            print("Done swapping prototypes.")

    def on_train_end(self):
        # print("stored", self.stored_activations)
        # now print it in decimal format roundd to 4 decimals
        print("stored:\n", self.stored_activations.detach().cpu().numpy().round(4) > 0.5)

    def generate_plots(self, logger=None, output_folder=None):
        if self.is_cubEMB:
            return
        generate_plots(self, logger=logger, output_folder=output_folder)
    
    def orthogonality_loss(self, prototypes):
        """
        Calculates the orthogonality loss to push prototypes apart.
        
        Args:
            prototypes: Tensor of shape (num_proto, proto_size)
            
        Returns:
            loss: Scalar tensor representing the penalty.
        """
        # 1. Normalize prototypes to lie on the unit sphere
        # eps adds stability to avoid division by zero
        prototypes_norm = F.normalize(prototypes, p=2, dim=1, eps=1e-12)
        
        # 2. Compute the Gram Matrix (Cosine Similarity Matrix)
        # Shape: (num_proto, num_proto)
        gram_matrix = torch.mm(prototypes_norm, prototypes_norm.t())
        
        # 3. Create the Identity Matrix target
        # Shape: (num_proto, num_proto)
        num_proto = prototypes.shape[0]
        identity = torch.eye(num_proto, device=prototypes.device)
        
        # 4. Compute MSE between Gram Matrix and Identity
        # This forces diagonal -> 1 and off-diagonal -> 0
        loss = torch.mean((gram_matrix - identity) ** 2)
        
        return loss

    def concept_diversity_loss(self, concept_probs):
        """
        concept_probs: (nb_proto, nb_concepts)
        Forces prototypes to represent distinct concept combinations.
        """
        # 1. Center the probabilities so 0.5 is the origin
        # (This makes 'True' and 'False' point in opposite directions)
        centered_concepts = concept_probs - 0.5 
        
        # 2. Normalize rows (concept vectors) to unit length
        centered_norm = F.normalize(centered_concepts, p=2, dim=1, eps=1e-12)
        
        # 3. Compute Gram Matrix (Cosine Similarity)
        # We want the off-diagonal elements to be close to 0 (uncorrelated)
        gram_matrix = torch.mm(centered_norm, centered_norm.t())
        
        # 4. Remove diagonal (self-similarity is always 1)
        identity = torch.eye(gram_matrix.shape[0], device=self.device)
        off_diagonal = gram_matrix * (1 - identity)
        
        # 5. Minimize the squared sum of correlations
        return torch.sum(off_diagonal ** 2) / (gram_matrix.shape[0] ** 2)
        
def dice_loss(pred_logits, target_mask, smooth=1.0):
    pred_probs = torch.sigmoid(pred_logits)
    pred_flat = pred_probs.view(-1)
    target_flat = target_mask.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def calc_segmentation_loss(pred_logits, gt_mask):
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    bce = bce_criterion(pred_logits, gt_mask)
    dice = dice_loss(pred_logits, gt_mask)
    return bce + dice




    

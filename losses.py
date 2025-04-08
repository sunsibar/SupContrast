"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
Modified (Added TargetVectorMSELoss): 2024-11-28
"""
from __future__ import print_function

import torch
import torch.nn as nn

from utils.uniform_points import random_uniform_points

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    :param neg_only: Train on the denominator of contrastive loss only. 
                     The values of the loss will be the same, but we detach the positive part.
                     For a pretraining stage.
    :param label_smoothing: Amount of label smoothing to apply (0.0 to disable)
    :param clip_pos: Amount of clipping to apply to the positives (0.0 to disable; between 0 and < 2; we will clip to below (1-clip_pos))
    :param clip_neg: Amount of clipping to apply to the negatives (0.0 to disable; between 0 and < 2; we will clip to above -(1-clip_neg))
    :param clip_neg_top_k: Only include the top k negatives in the denominator of the loss (per sample). Set to -1 to deactivate.
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, neg_only=False, label_smoothing=0.0, 
                 clip_pos=0.0, clip_neg=0.0, clip_neg_top_k=-1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.neg_only = neg_only
        self.label_smoothing = label_smoothing
        self.clip_neg_top_k = clip_neg_top_k
        if clip_neg_top_k != -1:
            if clip_neg_top_k < 1:
                raise ValueError('clip_neg_top_k must be greater than 1, since one negative will be the positive pair which is likely the largest. (or -1 to deactivate)')
            if clip_neg > 0 :
                raise ValueError('clip_neg_top_k must be -1 when clip_neg is set  > 0 (keep it simple)')
            if label_smoothing > 0:
                raise ValueError('label_smoothing must be 0 when clip_neg_top_k is set (havent implemented their interaction)')

        if self.neg_only:
            assert clip_pos == 0.0, 'clip_pos must be 0.0 when neg_only is True (not implemented otherwise)'
            assert clip_neg == 0.0, 'clip_neg must be 0.0 when neg_only is True (not implemented otherwise)'
            assert clip_neg_top_k == -1, 'clip_neg_top_k must be -1 when neg_only is True (not implemented otherwise)'
            self.clip_pos = None
            self.clip_neg = None
        else:
            assert clip_pos < 2., 'clip_pos must be less than 2.0, use "neg_only" to disable pos alltogether instead'
            assert clip_neg < 2., 'clip_neg must be less than 2.0 (we will clip to above -(1-clip_neg), so you pass the distance to the max dissimilarity)'
            assert clip_pos >= 0, 'clip_pos must be greater than or equal to 0.0'
            assert clip_neg >= 0, 'clip_neg must be greater than or equal to 0.0 (we will clip to above -(1-clip_neg), so you pass the distance to the max dissimilarity)'
            if clip_pos > 0:
                self.clip_pos = (1 - clip_pos)/self.temperature
            else:
                self.clip_pos = None
            if clip_neg > 0:
                self.clip_neg = (1 - clip_neg)/self.temperature
            else:
                self.clip_neg = None
    

    def forward(self, features, labels=None, mask=None, use_label_smoothing=True):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # "flatten" the view dimension, by first splitting into 2 (n_views) tensors along dim=1
        #  and then concatenating them along dim=0, so that all second views come after all first views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one': 
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        if self.label_smoothing > 0 and use_label_smoothing:
            # We have contrast_count * batch_size - 1 views we compare to, but also contrast_count positive views so ... 
            # Okay this doesn't make maybe so much sense in the case of supcon. Not using that anyways for now.
            # So again: For simclr, we have contrast_count * batch_size - 1 views we compare to.  
            # But only one positive view per anchor. So let's down-weight the negative views furter by contrast_count.
            mask = mask * (1 - self.label_smoothing) + (1 - mask) * (self.label_smoothing / (contrast_count * (batch_size - 1)))
    
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) 
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases (but if I'm not wrong this keeps the different views of the same sample in the denominator)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask 

        if self.neg_only:

            # for numerical stability
            logits = anchor_dot_contrast - logits_max.detach()

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask

            log_prob = logits.detach() - torch.log(exp_logits.sum(1, keepdim=True))

        else:
            # Optional clipping 
            if self.clip_pos is not None:
                # Clipping the positives on the upper end might reduce overfitting: https://arxiv.org/pdf/2407.15863 
                logits_pos = torch.clamp(anchor_dot_contrast,  max=self.clip_pos)
                # assert min >= -1/self.temperature
                logits_max = torch.clamp(logits_max, max=self.clip_pos)
            else:
                logits_pos = anchor_dot_contrast

            if self.clip_neg is not None:
                # Clip the logits similar to whats recommended here: https://arxiv.org/pdf/1910.06222 
                # But we don't clip on the upper end since this would allow collapse
                anchor_dot_contrast = torch.clamp(anchor_dot_contrast, min = -self.clip_neg)
                # logits_max = torch.max(logits_max, self.clip_neg) 

            # for numerical stability
            # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits_pos = logits_pos - logits_max.detach()
            anchor_dot_contrast = anchor_dot_contrast - logits_max.detach()

            # compute log_prob
            exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
            if self.clip_neg_top_k > 1:
                # sort the exp_logits and keep only the top k
                exp_logits, _ = torch.topk(exp_logits, k=self.clip_neg_top_k, dim=1, sorted=False)
                # exp_logits, _ = torch.sort(exp_logits, dim=1, descending=True)
                # exp_logits = exp_logits[:, :self.clip_neg_top_k] 

            log_prob = logits_pos - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        
        # Mask implements a weighted sum; originally, we summed over masked log_prob terms where only one element per row was not masked out. 
        # Now with label smoothing, we can either take a masked sum just the same way, weighting the (equal) denominator term as well so that 
        # it sums to once itself as well. Let's do that.  
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss




class TargetVectorMSELoss(nn.Module):
    """MSE Loss between embeddings and target vectors assigned to each class.
       Vectors are initialized as random uniform points on the unit hypersphere.
    For each class i, samples of that class should be mapped to the i-th target vector."""
    def __init__(self, num_classes, embedding_dim=128):
        super(TargetVectorMSELoss, self).__init__()
        self.num_classes = num_classes
        
        # Initialize target vectors using the uniform points method
        target_vectors = random_uniform_points(num_classes, embedding_dim)
        self.register_buffer('target_vectors', torch.from_numpy(target_vectors).float())

    def forward(self, features, labels=None):
        """Compute MSE loss between features and their target vectors.
        
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        if labels is None:
            raise ValueError('Labels must be provided for TargetVectorMSELoss')
            
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                           'at least 3 dimensions are required')
                           
        # Get batch size and number of views
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        # Reshape features to [bsz * n_views, ...]
        features = features.view(-1, features.shape[-1])
        
        # Repeat labels for each view
        labels_expanded = labels.repeat_interleave(n_views)
        
        # Get target vectors for each sample based on their labels
        targets = self.target_vectors[labels_expanded]
        
        # Compute MSE loss
        loss = torch.mean((features - targets) ** 2)
        
        return loss
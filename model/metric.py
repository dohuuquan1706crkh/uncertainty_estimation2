import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def intersection_over_union(pred, target):
    """
    Calculate the Intersection over Union (IoU) for each class, averaged over the batch.
    
    Args:
    - pred (Tensor): Predicted segmentation map with shape (B, N, H, W) where B is the batch size.
    - target (Tensor): Ground truth segmentation map with shape (B, H, W).
    - num_classes (int): Number of classes.
    
    Returns:
    - iou (Tensor): IoU for each class, averaged over the batch.
    """
    # Initialize IoU tensor
    num_classes = pred.shape[1]
    iou = torch.zeros(num_classes, dtype=torch.float32)
    
    # Iterate over each class
    for cls in range(num_classes):
        # Create binary masks for the current class
        pred_cls = (pred == cls).float()  # Shape: (B, N, H, W)
        target_cls = (target == cls).float()  # Shape: (B, H, W)

        # Expand dimensions to match shapes for element-wise multiplication
        target_cls_expanded = target_cls.unsqueeze(1)  # Shape: (B, 1, H, W)

        # Compute intersection and union
        intersection = (pred_cls * target_cls_expanded).sum(dim=[0, 2, 3])  # Shape: (N,)
        union = pred_cls.sum(dim=[0, 2, 3]) + target_cls_expanded.sum(dim=[0, 2, 3]) - intersection  # Shape: (N,)

        # Compute IoU for the current class
        iou[cls] = (intersection.sum() / (union.sum() + 1e-6)).item()  # Add epsilon to avoid division by zero

    return torch.sum(iou)



def dice_coefficient(pred, target):
    """
    Calculate the Dice Coefficient for each class, averaged over the batch.
    
    Args:
    - pred (Tensor): Predicted segmentation map with shape (B, N, H, W).
    - target (Tensor): Ground truth segmentation map with shape (B, H, W).
    - num_classes (int): Number of classes.
    
    Returns:
    - dice (Tensor): Dice Coefficient for each class, averaged over the batch.
    """
    num_classes = pred.shape[1]
    dice = torch.zeros(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls.unsqueeze(1)).sum(dim=[0, 2, 3])
        dice[cls] = 2 * intersection.sum() / (pred_cls.sum(dim=[0, 2, 3]).sum() + target_cls.sum(dim=[0, 2, 3]).sum() + 1e-6)
    return dice
def pixel_accuracy(pred, target):
    """
    Calculate the pixel accuracy, averaged over the batch.
    
    Args:
    - pred (Tensor): Predicted segmentation map with shape (B, N, H, W).
    - target (Tensor): Ground truth segmentation map with shape (B, H, W).
    
    Returns:
    - accuracy (float): Pixel accuracy averaged over the batch.
    """
    num_classes = pred.shape[1]
    pred_cls = torch.argmax(pred, dim=1)
    correct = (pred_cls == target).float().sum()
    total = target.numel()
    accuracy = correct / total
    return accuracy
def mean_pixel_accuracy(pred, target):
    """
    Calculate the mean pixel accuracy, averaged over the batch.
    
    Args:
    - pred (Tensor): Predicted segmentation map with shape (B, N, H, W).
    - target (Tensor): Ground truth segmentation map with shape (B, H, W).
    - num_classes (int): Number of classes.
    
    Returns:
    - mean_accuracy (float): Mean pixel accuracy, averaged over the batch.
    """
    num_classes = pred.shape[1]
    pred_cls = torch.argmax(pred, dim=1)
    accuracies = []
    for cls in range(num_classes):
        pred_cls_c = (pred_cls == cls).float()
        target_cls_c = (target == cls).float()
        # print((pred_cls_c * target_cls_c).shape)
        correct = (pred_cls_c * target_cls_c).sum(dim=[0, 1, 2])
        total = target_cls_c.sum(dim=[0, 1, 2]) + 1e-6
        accuracies.append(correct / total)
    return torch.mean(torch.stack(accuracies))
def mean_intersection_over_union(pred, target, num_classes):
    """
    Calculate the mean Intersection over Union (mIoU), averaged over the batch.
    
    Args:
    - pred (Tensor): Predicted segmentation map with shape (B, N, H, W).
    - target (Tensor): Ground truth segmentation map with shape (B, H, W).
    - num_classes (int): Number of classes.
    
    Returns:
    - mean_iou (float): Mean Intersection over Union, averaged over the batch.
    """
    iou = intersection_over_union(pred, target)
    return iou.mean()
def precision(pred, target):
    """
    Calculate precision for each class, averaged over the batch.
    
    Args:
    - pred (Tensor): Predicted segmentation map with shape (B, N, H, W).
    - target (Tensor): Ground truth segmentation map with shape (B, H, W).
    - num_classes (int): Number of classes.
    
    Returns:
    - precision (Tensor): Precision for each class, averaged over the batch.
    """
    num_classes = pred.shape[1]
    precision = torch.zeros(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        true_positive = (pred_cls * target_cls.unsqueeze(1)).sum(dim=[0, 2, 3])
        predicted_positive = pred_cls.sum(dim=[0, 2, 3])
        precision[cls] = true_positive.sum() / (predicted_positive.sum() + 1e-6)
    return precision
def recall(pred, target):
    """
    Calculate recall for each class, averaged over the batch.
    
    Args:
    - pred (Tensor): Predicted segmentation map with shape (B, N, H, W).
    - target (Tensor): Ground truth segmentation map with shape (B, H, W).
    - num_classes (int): Number of classes.
    
    Returns:
    - recall (Tensor): Recall for each class, averaged over the batch.
    """
    num_classes = pred.shape[1]
    recall = torch.zeros(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        true_positive = (pred_cls * target_cls.unsqueeze(1)).sum(dim=[0, 2, 3])
        actual_positive = target_cls.sum(dim=[0, 1, 2])
        recall[cls] = true_positive.sum() / (actual_positive.sum() + 1e-6)
    return recall
def f1_score(pred, target):
    """
    Calculate F1 score for each class, averaged over the batch.
    
    Args:
    - pred (Tensor): Predicted segmentation map with shape (B, N, H, W).
    - target (Tensor): Ground truth segmentation map with shape (B, H, W).
    - num_classes (int): Number of classes.
    
    Returns:
    - f1 (Tensor): F1 score for each class, averaged over the batch.
    """
    num_classes = pred.shape[1]
    precision_scores = precision(pred, target)
    recall_scores = recall(pred, target)
    f1 = 2 * (precision_scores * recall_scores) / (precision_scores + recall_scores + 1e-6)
    return f1


def correlation(y_hat, target_label, uncertainty):
    
    
    
    # Step 1: Convert segmentation map to predicted class labels
    predicted_labels = torch.argmax(y_hat, dim=1)  # Shape: [B, W, H]
    # target_label = torch.argmax(target, dim=1)
    # print(predicted_labels.shape)
    # print(target_label.shape)
    

    # Step 2: Compare with ground truth
    error_map = (predicted_labels != target_label).float()  # Shape: [B, W, H]
    # Flatten the tensors
    error_flat = error_map.view(-1).float()
    uncertainty_flat = uncertainty.view(-1).float()

    # Calculate the mean of each tensor
    error_mean = torch.mean(error_flat)
    uncertainty_mean = torch.mean(uncertainty_flat)

    # Subtract the mean from each tensor (center the data)
    error_centered = error_flat - error_mean
    uncertainty_centered = uncertainty_flat - uncertainty_mean

    # Calculate the covariance between x and y
    covariance = torch.sum(error_centered * uncertainty_centered) / (error_flat.size(0) - 1)

    # Calculate the standard deviations
    error_std = torch.std(error_flat, unbiased=False)
    uncertainty_std = torch.std(uncertainty_flat, unbiased=False)

    # Calculate the Pearson correlation coefficient
    correlation = covariance / (error_std * uncertainty_std)
    return correlation


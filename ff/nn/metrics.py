import torch
import numpy as np

class ConfusionMatrix():
    def __init__(self, num_classes: int):
        """Confusion matrix for multi-class classification.
        
        Args:
            num_classes: Number of classes in the dataset.
        """

        self.num_class = num_classes
        self.cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    
    @torch.no_grad()
    def add(self, pred: torch.Tensor|np.ndarray, target: torch.Tensor|np.ndarray):
        """Add predictions and targets to the confusion matrix.
        
        Args:
            pred: Predictions from the model.
            target: Targets from the data.
        """

        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        pred, target = pred.view(-1), target.view(-1)
        index = target * self.num_class + pred
        bin_count = torch.bincount(index, minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        self.cm += bin_count.to(self.cm.device)
    
    def reset(self):
        """Reset the confusion matrix."""
        self.cm.zero_()
    
    def get_tp_per_class(self) -> torch.Tensor:
        """Get the true positive predictions per class with shape: (num_classes,)."""
        return torch.diag(self.cm)
    
    def get_fp_per_class(self) -> torch.Tensor:
        """Get the false positive predictions per class with shape: (num_classes,)."""
        return self.cm.sum(dim=0) - self.get_tp_per_class()
    
    def get_fn_per_class(self) -> torch.Tensor:
        """Get the false negative predictions per class with shape: (num_classes,)."""
        return self.cm.sum(dim=1) - self.get_tp_per_class()
    
    def get_tp(self) -> int:
        """Get the total true positive predictions."""
        return self.get_tp_per_class().sum().item()
    
    def get_fp(self) -> int:
        """Get the total false positive predictions."""
        return self.get_fp_per_class().sum().item()
    
    def get_fn(self) -> int:
        """Get the total false negative predictions."""
        return self.get_fn_per_class().sum().item()
    
    def get_precision_per_class(self) -> torch.Tensor:
        """Get the precision per class with shape: (num_classes,)."""
        tp = self.get_tp_per_class().float()
        fp = self.get_fp_per_class().float()
        precision = torch.nan_to_num(tp / (tp + fp), nan=0.0)
        return precision
    
    def get_recall_per_class(self) -> torch.Tensor:
        """Get the recall per class with shape: (num_classes,)."""
        tp = self.get_tp_per_class().float()
        fn = self.get_fn_per_class().float()
        recall = torch.nan_to_num(tp / (tp + fn), nan=0.0)
        return recall
    
    def get_f1_per_class(self) -> torch.Tensor:
        """Get the F1 score per class with shape: (num_classes,)."""
        precision = self.get_precision_per_class()
        recall = self.get_recall_per_class()
        f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), nan=0.0)
        return f1
    
    def get_precision(self, macro: bool = True) -> float:
        """Get the precision.

        Macro averaging computes the average of the precision per class.
        Micro averaging computes the precision based on the total true positives and false positives.
        
        Args:
            macro: Whether to use macro averaging.
        """

        if macro:
            return self.get_precision_per_class().mean().item()
        else:
            tp = self.get_tp()
            fp = self.get_fp()
            if tp + fp <= 0:
                return 0.0
            return tp / (tp + fp)
    
    def get_recall(self, macro: bool = True) -> float:
        """Get the recall.
        
        Macro averaging computes the average of the recall per class.
        Micro averaging computes the recall based on the total true positives and false negatives.
        
        Args:
            macro: Whether to use macro averaging.
        """

        if macro:
            return self.get_recall_per_class().mean().item()
        else:
            tp = self.get_tp()
            fn = self.get_fn()
            if tp + fn <= 0:
                return 0.0
            return tp / (tp + fn)
    
    def get_accuracy(self) -> float:
        """Get the overall accuracy."""
        total = self.cm.sum().item()
        if total <= 0:
            return 0.0
        return self.get_tp() / total
    
    def get_f1(self, macro: bool = True) -> float:
        """Get the F1 score.
        
        Macro averaging computes the average of the F1 score per class.
        Micro averaging computes the F1 score based on the micro averaged precision and recall.
        
        Args:
            macro: Whether to use macro averaging.
        """

        if macro:
            return self.get_f1_per_class().mean().item()
        else:
            precision = self.get_precision(False)
            recall = self.get_recall(False)
            if precision + recall <= 0:
                return 0.0
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
    
    def get_statistics(self) -> dict:
        """Get the statistics of the confusion matrix."""
        return {
            'precision': self.get_precision(),
            'recall': self.get_recall(),
            'accuracy': self.get_accuracy(),
            'f1': self.get_f1()
        }

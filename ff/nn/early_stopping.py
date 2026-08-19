from typing import Any

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, higher_is_better: bool = True):
        """Early stopping for training.

        Args:
            patience: Number of epochs to wait before stopping if no improvement.
            min_delta: Minimum improvement required to consider as an improvement.
            higher_is_better: Whether higher metric values are better.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.higher_is_better = higher_is_better

        self.best_score = None
        self.counter = 0
    
    def reset(self):
        """Reset the early stopping."""
        self.best_score = None
        self.counter = 0
    
    def update(self, score: float) -> tuple[bool, bool]:
        """Update the score and check if early stopping should be triggered.
        
        Args:
            score: The current score to compare to the best score.

        Returns:
            done: Whether early stopping should be triggered.
            better: Whether the current score is an improvement over the best score.
        """

        # Update the best score and counter
        better = self._is_better(score)
        if better:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1 
        
        # Check if early stopping should be triggered
        done = self.counter >= self.patience
        
        return done, better
    
    def state_dict(self) -> dict[str, Any]:
        """Get the state of the early stopping."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "higher_is_better": self.higher_is_better,
            "best_score": self.best_score,
            "counter": self.counter
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load the state of the early stopping."""
        self.patience = state_dict["patience"]
        self.min_delta = state_dict["min_delta"]
        self.higher_is_better = state_dict["higher_is_better"]
        self.best_score = state_dict["best_score"]
        self.counter = state_dict["counter"]

    def _is_better(self, score: float) -> bool:
        """Check if the current score is an improvement over the best score."""
        # Initialize the best score
        if self.best_score is None:
            return True
        # Check if the current score is an improvement
        if self.higher_is_better:
            return score > (self.best_score + self.min_delta)
        else:
            return score < (self.best_score - self.min_delta)

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoderProcessor(nn.Module):
    def __init__(self, model_id: str='openai/clip-vit-large-patch14'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode(self, text: str | list[str], length: int=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text: Text to encode.
            length: Maximum length of the text.
        Returns:
            Tuple of (embedding, attention_mask).
        """
        outputs, mask = self._encode_common(text, length)
        return outputs.last_hidden_state.detach(), mask
    
    @torch.no_grad()
    def encode_pooled(self, text: str | list[str], length: int=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text: Text to encode.
            length: Maximum length of the text.
        Returns:
            Tuple of (pooled embedding, attention_mask).
        """
        outputs, mask = self._encode_common(text, length)
        return outputs.pooler_output.detach(), mask
    
    def _encode_common(self, text: str | list[str], length: int=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text: Text to encode.
            length: Maximum length of the text.
        Returns:
            Tuple of (embedding, attention_mask).
        """
        # Get length
        if length is None:
            length = self.tokenizer.model_max_length
        # Tokenize
        inputs = self.tokenizer(text, padding="max_length", max_length=length,
                                truncation=True, return_tensors="pt").to(self.model.device)
        # Extract mask
        mask = inputs.attention_mask.detach() # (B, L)
        # Model forward
        outputs = self.model(**inputs)

        return outputs, mask

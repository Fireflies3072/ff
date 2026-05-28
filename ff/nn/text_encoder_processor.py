import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, CLIPTextModel, T5EncoderModel

class TextEncoderProcessor(nn.Module):
    def __init__(self, model_id: str='openai/clip-vit-large-patch14'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        config = AutoConfig.from_pretrained(model_id)
        if config.model_type == "clip":
            # If CLIP, use the text encoder part
            self.model = CLIPTextModel.from_pretrained(model_id)
        elif config.model_type == "t5":
            # If T5, use the encoder part
            self.model = T5EncoderModel.from_pretrained(model_id)
        else:
            # Other text models (BERT, RoBERTa, etc.)
            self.model = AutoModel.from_pretrained(model_id)
        self.model.eval().requires_grad_(False)

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
        # Pool the output
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        return pooled.detach(), mask
    
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
            length = getattr(self.tokenizer, 'model_max_length', 512)
            length = min(length, 512)
        # Tokenize
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, padding="max_length", max_length=length,
                                truncation=True, return_tensors="pt").to(device)
        # Extract mask
        mask = inputs.attention_mask.detach() # (B, L)
        # Model forward
        outputs = self.model(**inputs)

        return outputs, mask

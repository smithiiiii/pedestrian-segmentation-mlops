import torch

from src.model import SegmentationModel


def load_trained_model(model_path: str = "best_model.pt", device: str = "cpu") -> SegmentationModel:
    """
    Loads trained weights safely into the SegmentationModel instance.
    Handles both state_dict and full model objects.
    """
    model = SegmentationModel()

    checkpoint = torch.load(model_path, map_location=torch.device(device))

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint

    model.to(device)
    model.eval()
    return model

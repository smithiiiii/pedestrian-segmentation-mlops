import segmentation_models_pytorch as smp
import torch.nn as nn


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.arc = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, images, masks=None):
        logits = self.arc(images)
        if masks is not None:
            loss1 = smp.losses.DiceLoss(mode="binary")(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1 + loss2
        return logits

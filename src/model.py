import open_clip
import timm
from torch import nn
from torchvision.models import convnext_base


class ViTB16(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.0):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.img_size = self.vit.default_cfg["input_size"][-1]
        self.embedding_dim = self.vit.embed_dim
        self.mean = self.vit.default_cfg["mean"]
        self.std = self.vit.default_cfg["std"]

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.embedding_dim, num_classes),
        )

    def forward(self, x):
        embedding = self.vit(x)
        logits = self.head(embedding)
        return logits

    def freeze_backbone(self):
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.vit.parameters():
            param.requires_grad = True


class ConvNeXtB(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = convnext_base(weights='IMAGENET1K_V1')
        in_feats = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_feats, num_classes)

        self.img_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.model(x)
        return x

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True


class CLIPConvNeXtB(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model, train_transform, _ = open_clip.create_model_and_transforms("convnext_base_w", "laion2b_s13b_b82k_augreg")
        self.model = model.visual
        self.model.head.proj.out_features = num_classes

        self.img_size = self.model.image_size
        self.mean = self.model.image_mean
        self.std = self.model.image_std

    def forward(self, x):
        x = self.model(x)
        return x

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    pass


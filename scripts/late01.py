from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x: torch.Tensor, p: int = 3, eps: float = 1e-6) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Residual3DBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
        )

    def forward(self, x):
        shortcut = x
        h = self.block(x)
        h = self.block2(h)
        out = F.relu(h + shortcut)
        return out


class Model(nn.Module):
    """
    Reference:
    [1] https://www.kaggle.com/competitions/nfl-player-contact-detection/discussion/392402#2170010
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, num_classes=1, in_chans=3)
        self.mlp = nn.Sequential(
            nn.Linear(68, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        n_hidden = 1024
        self.conv_proj = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )

        self.neck = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.triple_layer = nn.Sequential(
            Residual3DBlock(),
        )

        self.pool = GeM()

        self.fc = nn.Linear(256 + 1024, 1)

    def forward(
        self,
        images: torch.Tensor,
        feature: torch.Tensor,
        target: torch.Tensor | None = None,
        mixup_hidden: bool = False,
        mixup_alpha: float = 0.1,
        layer_mix=None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, h, w = images.shape
        print(b, t, h, w)
        # 後でin_chans=3にfeature_forwardに通すため
        # batch側に寄せてencodeする
        images = images.view(b * t // 3, 3, h, w)
        # print("image_view.shape", images.shape)
        encoded_feature = self.backbone.forward_features(images)
        # print("encoded_feature.shape: ", encoded_feature.shape)
        feature_maps = self.conv_proj(encoded_feature)
        # print("1 feature_maps.shape: ", feature_maps.shape)
        _, c, h, w = feature_maps.shape
        feature_maps = feature_maps.contiguous().view(b * 2, c, t // 2 // 3, h, w)
        # print("2 feature_maps.shape: ", feature_maps.shape)
        feature_maps = self.triple_layer(feature_maps)
        # print("3 feature_maps.shape: ", feature_maps.shape)
        # middle_maps: (16, 512, 5, 16, 16)
        # 真ん中を取り出すのは対象のフレーム<=>着目してるフレーム
        middle_maps = feature_maps[:, :, 2, :, :]
        # print("middle_maps.shape: ", feature_maps.shape)
        # 抽出した着目してるフレームの特徴量をpoolingすることでフレーム内のコンテキストの情報を集約する
        # pooled_maps: (16, 512, 1, 1)
        pooled_maps = self.pool(middle_maps)
        # print("pooled_maps.shape: ", pooled_maps.shape)
        # reshpaed_pooled_maps: (8, 512*2)
        nn_feature = self.neck(pooled_maps.reshape(b, -1))
        # print(f"nn_feature.shape: {nn_feature.shape}")

        # print(f"1 feature.shape: {feature.shape}")
        # 単に特徴から学習につかう特徴を抽出する
        feature = self.mlp(feature)
        # print(f"2 feature.shape: {feature.shape}")
        cat_feature = torch.cat([nn_feature, feature], dim=1)
        # print(f"cat_feature.shape: {cat_feature.shape}")
        if target is not None:
            cat_feature, y_a, y_b, lam = self.mixup(cat_feature, target, mixup_alpha)
            y = self.fc(cat_feature)
            return y, y_a, y_b, lam
        else:
            y = self.fc(cat_feature)
            return y


def _test_run_model():
    model = Model()
    model.eval()
    im = torch.randn((8, 30, 512, 512))
    feature = torch.randn((8, 68))
    y = model(im, feature)
    print(y)
    print(y.shape)


if __name__ == "__main__":
    _test_run_model()

import typing

import torch
from lerf.data.utils.feature_dataloader import FeatureDataloader
from torchvision import transforms
from tqdm import tqdm

import timm

class SamDataloader(FeatureDataloader):
    # dino_model_type: str = "dino_vits8"
    # dino_stride: int = 8
    # dino_model_type = "dinov2_vitb14_reg"
    # dino_stride = 14
    # 994, 738
    # dino_load_size: typing.Union[int, typing.Tuple[int, int]] = 500
    # dino_layer = 11
    # dino_facet = "key"
    # dino_bin = False
    load_size = 464
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    model: torch.nn.Module

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
    ):
        assert "image_shape" in cfg

        if 0:
            self.model = timm.create_model(
                'samvit_base_patch16',
                True,
                num_classes=0,
            ).eval().to(self.device)

        self.model = timm.models.vision_transformer_sam.samvit_base_patch16(pretrained=True).eval().to(device)

        # self.transform = timm.data.create_transform(**timm.data.resolve_model_data_config(self.model), is_training=False)
        
        super().__init__(cfg, device, image_list, cache_path)

    def create(self, image_list):
        print("Image size =", image_list.size())
        
        preproc_image_list: torch.Tensor = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(self.load_size, antialias=None),
            # transforms.CenterCrop(load_size),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])(image_list).to(self.device)

        # print("Image size =", preproc_image_list.size())
        # 4096 * 3 * 464 * 624

        sam_embeds = []
        for image in tqdm(preproc_image_list, desc="sam", total=len(preproc_image_list), leave=False):
            # 3 * 464 * 624
            # print("C * H * W =", image.size())
            image = image.unsqueeze(0)
            with torch.no_grad():
                descriptors: torch.Tensor = self.model.forward_features(image)
            # 1, 64, 29, 39
            print("[SAM] 1 * D * ((H // P) * (W // P)) =", descriptors.size())
            descriptors = descriptors.squeeze(0).permute(1, 2, 0)
            # 29, 39, 64
            print("[SAM] (H // P) * (W // P) * D =", descriptors.size())
            sam_embeds.append(descriptors.cpu().detach())

        # 4080, 29, 39, 64
        self.data = torch.stack(sam_embeds, dim=0)

    def __call__(self, img_points: torch.Tensor):
        # 4096, 3
        # print("Points size =", img_points.size())
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)
        # return self.model.forward_features(img_points)


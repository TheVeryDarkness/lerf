import typing

import torch
from lerf.data.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm


class SamDataloader(FeatureDataloader):
    dino_model_type: str = "dino_vits8"
    dino_stride: int = 8
    # dino_model_type = "dinov2_vitb14_reg"
    # dino_stride = 14
    # 994, 738
    dino_load_size: typing.Union[int, typing.Tuple[int, int]] = 500
    dino_layer = 11
    dino_facet = "key"
    dino_bin = False

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
    ):
        assert "image_shape" in cfg
        super().__init__(cfg, device, image_list, cache_path)

    def create(self, image_list):
        extractor = ViTExtractor(self.dino_model_type, self.dino_stride)
        preproc_image_lst = extractor.preprocess(image_list, self.dino_load_size)[0].to(self.device)

        dino_embeds = []
        for image in tqdm(preproc_image_lst, desc="dino", total=len(image_list), leave=False):
            print("C * H * W =", image.size())
            image = image.unsqueeze(0)
            if False: # Dino v2
                P = 14
                _, C, H, W = image.size()
                image = torch.nn.functional.interpolate(image, (H // P * P, W // P * P))
            with torch.no_grad():
                descriptors = extractor.extract_descriptors(
                    image,
                    [self.dino_layer],
                    self.dino_facet,
                    self.dino_bin,
                )
            print("1 * T * D =", descriptors.size())
            descriptors = descriptors.reshape(extractor.num_patches[0], extractor.num_patches[1], -1)
            # descriptors = torch.nn.functional.interpolate(descriptors, (extractor.num_patches[0], extractor.num_patches[1]))
            print("1 * (T / P) * (D / P) =", descriptors.size())
            dino_embeds.append(descriptors.cpu().detach())

        self.data = torch.stack(dino_embeds, dim=0)

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)

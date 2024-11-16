#@article{BiRefNet,
# title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
# author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
# journal={CAAI Artificial Intelligence Research},
# year={2024}

import os
import torch
from PIL import Image, ImageFilter, ImageOps
from PIL.Image import Resampling
from torchvision import transforms
import numpy as np
import folder_paths
from transformers import AutoModelForImageSegmentation

device = "cuda" if torch.cuda.is_available() else "cpu"

folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

MODEL_NAME = "briaai/RMBG-2.0"
MODEL_PATH = os.path.join(folder_paths.models_dir, "RMBG", MODEL_NAME.replace("/", "--"))

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class RMBG2_0Node:
    """
    RMBG 2.0 Node for background removal with direct processing of the original image.
    
    使用说明 | Instructions:
    1. 直接使用原始图像进行背景移除。
    2. 使用背景移除模型生成透明图像和对应的蒙版。
    3. 可通过 invert_mask 参数对蒙版进行反转。
    """

    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        """
        定义节点的输入参数，包括图像、缩放方式和其他参数。
        """
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图像 | Input image for background removal."}),
                "invert_mask": (["False", "True"], {
                    "default": "False",
                    "tooltip": "是否反转蒙版颜色：False 表示不反转 | Whether to invert the mask colors: False means no inversion."
                }),
                "feather_amount": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "羽化蒙版的半径，单位像素 | Radius for feathering the mask, in pixels."
                }),
                "dilate_amount": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "扩展蒙版的半径，单位像素 | Radius for dilating the mask, in pixels."
                }),
                "process_res": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "预处理时将图像缩放到的分辨率 | Resolution to resize the image during preprocessing."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "image/preprocessing"
    NODE_NAME = "RMBG 2.0"

    def load_model(self):
        if self.model is None:
            if not os.path.exists(MODEL_PATH):
                print(f"Downloading {MODEL_NAME} model... This may take a while.")
            
            self.model = AutoModelForImageSegmentation.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                cache_dir=MODEL_PATH,
                revision="main",
                local_files_only=False
            )
            self.model.to(device)
            self.model.eval()
            print(f"Loaded model: {MODEL_NAME}")

    def remove_background(self, image, invert_mask, feather_amount, dilate_amount, process_res=1024):
        """
        执行背景移除和蒙版处理 | Perform background removal and mask processing.
        """
        try:
            self.load_model()

            torch.cuda.empty_cache()

            transform_image = transforms.Compose([
                transforms.Resize((process_res, process_res)),  # 添加缩放操作
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            processed_images = []
            processed_masks = []

            for img in image:
                orig_image = tensor2pil(img)

                # Step 1: Convert original image to tensor and send to model
                input_tensor = transform_image(orig_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    preds = self.model(input_tensor)[-1].sigmoid().cpu()
                    pred = preds[0].squeeze()
                    pred = (pred > 0.5).float()

                    mask = transforms.ToPILImage()(pred)
                    mask = mask.resize(orig_image.size)  # 将蒙版恢复到原始图像的分辨率

                    # Step 2: Process the mask
                    final_mask = mask

                    # If invert_mask is True, invert the mask
                    if invert_mask == "True":
                        final_mask = ImageOps.invert(mask.convert("L"))
                        print("[DEBUG] Mask inverted for visual black-white inversion.")

                    # Apply feathering
                    if feather_amount > 0:
                        final_mask = final_mask.filter(ImageFilter.GaussianBlur(radius=feather_amount))

                    # Apply dilation
                    if dilate_amount > 0:
                        final_mask = final_mask.filter(ImageFilter.MaxFilter(size=dilate_amount + 1))

                    # Step 3: Generate output image with alpha channel
                    new_im = orig_image.copy()
                    new_im.putalpha(mask)

                    # Convert results to tensor
                    new_im_tensor = pil2tensor(new_im)
                    mask_tensor = pil2tensor(final_mask)

                    processed_images.append(new_im_tensor)
                    processed_masks.append(mask_tensor)

            torch.cuda.empty_cache()

            new_ims = torch.cat(processed_images, dim=0)
            new_masks = torch.cat(processed_masks, dim=0)

            return (new_ims, new_masks)

        except Exception as e:
            raise RuntimeError(f"Error loading or processing RMBG model: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "RMBG2_0Node": RMBG2_0Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RMBG2_0Node": "RMBG 2.0"
}
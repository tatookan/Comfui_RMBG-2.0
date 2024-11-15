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

def preprocess_image(image, width, height, mode):
    """
    Preprocess the image by resizing or cropping based on the specified mode:
    1. Resize to fit the longest edge while maintaining aspect ratio.
    2. Crop to the specified width and height centered.
    3. No resizing or cropping if width and height are both 0.
    4. Ensure the dimensions are multiples of 32.
    """
    if width == 0 or height == 0:
        # No resizing or cropping
        return image

    if mode == "longest_edge":
        # Scale proportionally to fit the longest edge
        original_width, original_height = image.size
        scale = min(width / original_width, height / original_height)
        new_size = (int(original_width * scale), int(original_height * scale))
        image = image.resize(new_size, Resampling.LANCZOS)
    elif mode == "center_crop":
        # Center crop to the specified width and height
        image = ImageOps.fit(image, (width, height), method=Resampling.LANCZOS, centering=(0.5, 0.5))

    # Ensure the dimensions are multiples of 32
    new_width = (image.width + 31) // 32 * 32
    new_height = (image.height + 31) // 32 * 32
    image = image.resize((new_width, new_height), Resampling.LANCZOS)

    return image

class RMBG2_0Node:
    """
    RMBG 2.0 Node for background removal with advanced preprocessing options.
    
    使用说明 | Instructions:
    1. 通过参数设置输入图像的缩放方式。
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
                "resize_width": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "图像缩放的目标宽度，0 表示不缩放 | Target width for resizing, 0 means no resizing."
                }),
                "resize_height": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "图像缩放的目标高度，0 表示不缩放 | Target height for resizing, 0 means no resizing."
                }),
                "resize_mode": (["none", "longest_edge", "center_crop"], {
                    "default": "none",
                    "tooltip": "图像缩放方式：\n"
                               "none: 不进行缩放或裁剪\n"
                               "longest_edge: 按最长边等比例缩放\n"
                               "center_crop: 以指定宽高居中裁剪 | "
                               "Image resizing mode:\n"
                               "none: No resizing or cropping\n"
                               "longest_edge: Proportional scaling to fit the longest edge\n"
                               "center_crop: Center cropping with specified width and height."
                }),
                "invert_mask": (["False", "True"], {
                    "default": "False",
                    "tooltip": "是否反转蒙版颜色：False 表示不反转 | "
                               "Whether to invert the mask colors: False means no inversion."
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

    def remove_background(self, image, resize_width, resize_height, resize_mode, invert_mask, feather_amount, dilate_amount):
        """
        执行背景移除和蒙版处理 | Perform background removal and mask processing.
        """
        try:
            self.load_model()

            torch.cuda.empty_cache()

            transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            processed_images = []
            processed_masks = []

            for img in image:
                orig_image = tensor2pil(img)

                # Step 1: Preprocess the image (resize/crop)
                processed_image = preprocess_image(orig_image, resize_width, resize_height, resize_mode)

                # Step 2: Convert preprocessed image to tensor and send to model
                input_tensor = transform_image(processed_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    preds = self.model(input_tensor)[-1].sigmoid().cpu()
                    pred = preds[0].squeeze()
                    pred = (pred > 0.5).float()

                    mask = transforms.ToPILImage()(pred)
                    mask = mask.resize(processed_image.size)

                    # Step 3: Process the mask
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

                    # Step 4: Generate output image with alpha channel
                    new_im = processed_image.copy()
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
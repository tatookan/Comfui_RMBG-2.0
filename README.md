# RMBG 2.0 - Background Removal Node
## Overview
RMBG 2.0 is an advanced background removal node designed for use in image processing pipelines. It leverages the briaai/RMBG-2.0 model from Hugging Face to perform high-quality background removal and provides various preprocessing options to enhance the final output.



## Installation
### Clone the Repository:

bash
git clone https://github.com/tatookan/Comfui_RMBG-2.0 
### Install Dependencies:

bash
pip install -r requirements.txt
### Download Model: The model will be automatically downloaded when you run the node for the first time. Alternatively, you can manually download it from Hugging Face and place it in the specified model directory.

## Usage
### Node Configuration
The RMBG 2.0 node can be configured with the following parameters:

Resize Mode:
none: No resizing or cropping.
Invert Mask: Whether to invert the mask colors.
Feather Amount: Radius for feathering the mask, in pixels.
Dilate Amount: Radius for dilating the mask, in pixels.
process res:Image scaling specifies the resolution, the maximum value of the test is 1024, adjusting this value will have different results.

## Model Details
### Model Name: briaai/RMBG-2.0
### Model Path: models/RMBG/briaai--RMBG-2.0

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Thanks to the Hugging Face team for providing the briaai/RMBG-2.0 model.
Special thanks to the contributors who have helped improve this project.

# credits
RMBG-2.0: https://huggingface.co/briaai/RMBG-2.0

# Contact Details
Email: dianyuanan@vip.qq.com  
加入我的粉丝群: 联系微信: Miss-Y-s-Honey, 并注明来意
查看我的教程频道 [bilibili@深深蓝hana](https://space.bilibili.com/618554?spm_id_from=333.1007.0.0)
日常作品分享 [douyin@深深蓝](https://www.douyin.com/user/MS4wLjABAAAAJGu7yCfV3XwKoklBX62bivvat3micLxemdDT0FAmdcGfqbuFS3ItsKWKrBt5Hg16?from_tab_name=)

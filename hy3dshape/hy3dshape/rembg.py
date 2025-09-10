# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from PIL import Image
from rembg import remove, new_session
import os
from typing import Optional


DEFAULT_REMBG_MODEL_PATH = \
    "/opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/rembg/u2net.onnx"


class BackgroundRemover():
    def __init__(self, model: Optional[str] = None):
        # model 可以是模型名（如 'u2net', 'isnet-general'）或本地 .onnx 文件路径
        # 也可通过环境变量 REMBG_MODEL 指定；都未提供时，优先尝试默认本地 onnx 路径
        # 若默认路径不存在，则回退到 rembg 的默认模型下载/缓存逻辑
        env_model = os.getenv("REMBG_MODEL", None)
        self.model = model or env_model or (
            DEFAULT_REMBG_MODEL_PATH if os.path.isfile(DEFAULT_REMBG_MODEL_PATH) else None
        )
        if self.model:
            self.session = new_session(self.model)
        else:
            self.session = new_session()

    def __call__(self, image: Image.Image):
        output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0])
        return output

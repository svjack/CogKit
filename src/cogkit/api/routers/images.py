# -*- coding: utf-8 -*-


import base64
import time
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends

from cogkit.api.dependencies import get_image_generation_service
from cogkit.api.models.images import ImageGenerationParams, ImageInResponse, ImagesResponse
from cogkit.api.services import ImageGenerationService

router = APIRouter()

def np_to_base64(image_array: np.ndarray) -> str:
    byte_stream = image_array.tobytes()
    base64_str = base64.b64encode(byte_stream).decode('utf-8')
    return base64_str

@router.post("/generations")
def generations(
    image_generation: Annotated[ImageGenerationService, Depends(get_image_generation_service)],
    params: ImageGenerationParams,
) -> ImagesResponse:
    images_list = image_generation.generate(
        model=params.model, prompt=params.prompt, size=params.size, num_images=params.n
    )
    images_base64 = [ImageInResponse(b64_json=np_to_base64(image)) for image in images_list]
    return ImagesResponse(created=int(time.time()), data=images_base64)

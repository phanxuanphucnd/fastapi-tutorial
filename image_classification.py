# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan


import io
import json
import base64
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from datetime import datetime

class ImageClassification:
    def __int__(self):
        # model
        self.model = models.densenet121(pretrained=True)
        self.model.eval()
        # imagenet classes
        self.imagenet_class_index = json.load(open('./data/imagenet_class_index.json'))

    def transform_image(self, img_bytes):
        my_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(img_bytes))
        return my_transforms(image).unsqueeze(0)

    def get_prediction(self, img_bytes):
        tensor = self.transform_image(img_bytes=img_bytes)
        outputs = self.model.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())

        return self.imagenet_class_index[predicted_idx]

    def get_result(self, data, is_api=False):
        starttime = datetime.now()
        img_bytes = data.file.read()

        class_id, class_name = self.get_prediction(img_bytes)

        endtime = datetime.now()
        time_diff = (endtime - starttime)

        executiontime = f"{round(time_diff.total_seconds() * 1000)} ms"

        encoded_string = base64.b64encode(img_bytes)
        bs64 = encoded_string.decode('utf-8')
        img_data = f"data:image/jpeg;base64,{bs64}"

        result = {
            'inference_time': executiontime,
            'predictions': {
                'class_id': class_id,
                'class_name': class_name
            }
        }

        if not is_api:
            result['image_data'] = img_data

        return result

    def router_api(self):
        from fastapi import APIRouter, UploadFile, File, Request
        api = APIRouter(prefix="/face_classification")

        @api.post("/face_classify/predict")
        async def generate_output(
                data: UploadFile = File(...)
        ):
            return self.get_result(data, is_api=True)

        return api

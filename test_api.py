# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

from fastapi.testclient import TestClient

import serving

client = TestClient(serving.app)

def test_home_rout():
    response = client.get('/')

    assert response.status_code == 20


def test_predict_route():
    file_name = 'data/dog.jpg'

    response = client.post(
        "/face_classify/predict", files={"file": ("dog_image", open(file_name, "rb"), "image/jpeg")}
    )
    print(response.status_code)

test_predict_route()
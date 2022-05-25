# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import uvicorn
import datetime
from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel, conlist
from starlette.middleware.cors import CORSMiddleware
from image_classification import ImageClassification
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

#TODO: Initialize
app = FastAPI()
image_classification = ImageClassification()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(image_classification.router_api())


if __name__ == '__main__':
    try:
        uvicorn.run(app, host='0.0.0.0', port=8001)
    except Exception as msg:
        print(msg)



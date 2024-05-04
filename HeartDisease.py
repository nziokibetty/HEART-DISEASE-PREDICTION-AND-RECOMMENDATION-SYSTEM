# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:48:29 2024

@author: hp
"""
from pydantic import BaseModel
class HeartDisease(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: int
    slope: int
    ca: int
    thal: int

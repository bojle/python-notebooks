#!/usr/bin/env python
# coding: utf-8

import onnx
import numpy as np
import re
import math
import sys
from onnx import shape_inference

## SHAPE INFERENCE

model = onnx.load(sys.argv[1])
#onnx.checker.check_model(model)
inferred = shape_inference.infer_shapes(model)
print("done")
onnx.checker.check_model(inferred)
print("saving")
onnx.save(inferred, sys.argv[2])

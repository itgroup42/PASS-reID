import tensorrt as trt

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import PIL
sys.path.append(".")

from collections import OrderedDict, namedtuple
from tqdm import tqdm
import torchvision.transforms as T


Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, namespace="")

w = "/home/mscherbina/Documents/github_repos/PASS-reID/PASS_transreid/models/trt_model.engine"

with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())
    if model is None:
        raise RuntimeError('Failed to load engine')
    context = model.create_execution_context()
    if context is None:
        raise RuntimeError('Failed to create execution context')

import torch
device = torch.device('cuda:0')

bindings = OrderedDict()
for index in range(model.num_bindings):
    name = model.get_binding_name(index)
    dtype = trt.nptype(model.get_binding_dtype(index))
    shape = tuple(model.get_binding_shape(index))
    data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
    bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())


exctract_func = lambda x: Path(
    x
).parent.name

gallery = "/home/mscherbina/Documents/github_repos/centroids-reid/data/gallery-subfolders"
pids = os.listdir(gallery)
pids = [p for p in pids if os.path.isdir(os.path.join(gallery, p))]
pids.sort()
path_embeddings = {}
big_batch = torch.zeros(8, 3, 256, 128).to(device)

val_transforms = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

preprocessing = 'cv2'

for pid in tqdm(pids):
    pid_path = os.path.join(gallery, pid)
    images = os.listdir(pid_path)
    images = [os.path.join(pid_path, i) for i in images]
    images.sort()
    for path in images:
        if preprocessing == 'cv2':
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).float()
            image = image / 255.0
            image -= torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            image /= torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            image = image.unsqueeze(0)
            image = image.to(device)
        elif preprocessing == "pil":
            image = PIL.Image.open(path)
            image = val_transforms(image)
        else:
            raise ValueError("Bad preprocessing type!!")

        big_batch[0] = image
        binding_addrs['input'] = int(big_batch.data_ptr())
        context.execute_v2(list(binding_addrs.values()))
        embedding = bindings['output'].data[0].cpu().numpy()
        path_embeddings[path] = embedding

np.save("/home/mscherbina/Documents/github_repos/PASS-reID/PASS_transreid/inference/output-dir/path_embeddings_tensorrt.npy", path_embeddings)
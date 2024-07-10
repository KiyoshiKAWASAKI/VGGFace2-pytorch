
import os
import sys
import models.resnet as ResNet
import models.senet as SENet
from tqdm import tqdm
import os
import numpy as np
import PIL.Image
import torch
from torch.utils import data
import torchvision.transforms
import pickle
import torchvision.transforms



# Paths and parameters
N_IDENTITY = 8631
include_top = False

# Image directory contains 90 morphs
# In each morph, there is enriched_tail and uniform
img_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
          "face_morph_v4_5_sets_processed"


# Resnet50
# save_embed_dir = "/project01/cvrl/jhuang24/" \
#                  "face_morph_v4_5_sets_features/vgg_resnet"
# model_path = "/project01/cvrl/jhuang24/vgg_models/" \
#              "resnet50_scratch_weight.pkl"
# model_name = "resnet"

# senet50
save_embed_dir = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_features/vgg_senet"
model_path = "/project01/cvrl/jhuang24/vgg_models/senet50_scratch_weight.pkl"
model_name = "senet"


# CUDA
cuda = torch.cuda.is_available()
if cuda:
    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

torch.manual_seed(1337)
if cuda:
    torch.cuda.manual_seed(1337)


# Select model
if model_name =='resnet':
    model = ResNet.resnet50(num_classes=N_IDENTITY,
                            include_top=include_top)
else:
    model = SENet.senet50(num_classes=N_IDENTITY,
                          include_top=include_top)

# Load pretrain model
with open(model_path, 'rb') as f:
    obj = f.read()

weights = {key: torch.from_numpy(arr) for key, arr in
           pickle.loads(obj, encoding='latin1').items()}

device = torch.device('cuda:0')
model = model.to(device)
model.load_state_dict(weights)
model.eval()


transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])

# Extract feature for each image
all_morphs = os.listdir(img_dir)

# Loop over each morph
for one_morph in tqdm(all_morphs):
    # For each morphs, loop through uniform and enriched tail
    all_types = os.listdir(os.path.join(img_dir, one_morph))

    for one_type in all_types:
        all_imgs = os.listdir(os.path.join(img_dir, one_morph, one_type))

        for one_img in all_imgs:
            with torch.no_grad():
                img = PIL.Image.open(os.path.join(img_dir, one_morph, one_type, one_img))
                img = torchvision.transforms.Resize(256)(img)
                img = torchvision.transforms.CenterCrop(224)(img)
                img = transform(img)
                img = img.unsqueeze(0)

                # img = Variable(img, volatile=True)
                img = img.to(device)

                output = model(img.float())  # N C H W torch.Size([1, 1, 401, 600])
                output = output.view(output.size(0), -1)
                output = output.data.cpu().numpy()
                output = np.squeeze(output)
                # print(output.shape)

                final_save_path = os.path.join(save_embed_dir, one_type, one_morph + "_" + one_img + ".npy")
                np.save(final_save_path, output)
                # sys.exit()




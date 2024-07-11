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
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import numpy as np


# Image directory contains 90 morphs
# In each morph, there is enriched_tail and uniform
img_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
          "face_morph_v4_5_sets_processed"

save_embed_dir = "/project01/cvrl/jhuang24/face_morph_v4_5_sets_features/vgg_vgg16"
model_path = "/project01/cvrl/jhuang24/vgg_models/VGG_FACE.t7"


class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

    def load_weights(self, path=model_path):
        """ Function to load luatorch pretrained

        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        return self.fc8(x)


if __name__ == "__main__":
    model = VGG_16().double()
    model.load_weights()

    device = torch.device('cuda:0')
    model.to(device)
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
                    img = img.to(device)

                    output = model(img.double())  # N C H W torch.Size([1, 1, 401, 600])
                    # print(output.shape)
                    # print(F.softmax(output, dim=1).shape)
                    # sys.exit()
                    output = output.view(output.size(0), -1)
                    output = output.data.cpu().numpy()
                    output = np.squeeze(output)
                    # print(output.shape)

                    # sys.exit()

                    final_save_path = os.path.join(save_embed_dir, one_type, one_morph + "_" + one_img + ".npy")
                    np.save(final_save_path, output)
                    # sys.exit()
    #
    # im = cv2.imread("images/ak.png")
    # im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224).double()
    #
    #
    #
    # im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)
    # preds = F.softmax(model(im), dim=1)
    # values, indices = preds.max(-1)
    # print(indices)
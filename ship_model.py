
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import os
import torch
from torch.autograd import Variable
from transforms import transforms
from models.LoadModel import MainModel
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# desired size of the output image, need to be 224,244 to use pretrained vgg16
imgsize = 448,448
num_cls = 40
warship_id_name = {0: '051B', 1: '051C', 2: '051G1G2', 3: '052', 4: '052B', 5: '052C', 6: '052D', 7: '053H1(053H1G)', 8: '053H3', 9: '054', 10: '054A', 11: '055', 12: '056(056A)', 13: '1135', 14: '11356', 15: '1143', 16: '1144', 17: '11540', 18: '1155', 19: '1164', 20: '11661', 21: '20380(20385)', 22: '21630(20631)', 23: '22160', 24: '22350', 25: '956E(EM)', 26: 'CG', 27: 'CVN', 28: 'DDG', 29: 'FFG', 30: 'HM', 31: 'JHSV', 32: 'LCC', 33: 'LCS', 34: 'LHA', 35: 'LHD', 36: 'LPD', 37: 'LSD', 38: 'MLP', 39: 'T-AOE'}
#transform PIL image to tensor
def image_to_tensor(pil_image):
    #resize image
    resize_reso = 512
    crop_reso = 448
    resized=ImageOps.fit(pil_image, imgsize, Image.ANTIALIAS)
    # transform it into a torch tensor
    loader = transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return loader(resized).unsqueeze(0) #need to add one dimension, need to be 4D to pass into the network

#load model
def load_model():
   resume = './net_model/weights_95_735_0.9818_0.9994.pth'
   model = MainModel()
   model_dict = model.state_dict()
   pretrained_dict = torch.load(resume)
   pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
   model_dict.update(pretrained_dict)
   model.load_state_dict(model_dict)
   return model

#get label for image at path "path"
def Warship(path):
    one_image = load_image(path)
    image_tensor = image_to_tensor(one_image)
    image_as_variable = Variable(image_tensor)
    model = load_model()
    model.eval()
    probabilities = model.forward(image_as_variable)
    outputs = model(image_as_variable)
    outputs_pred = outputs[0] + outputs[1][:, 0:num_cls] + outputs[1][:, num_cls:2 * num_cls]
    top3_val, top3_pos = torch.topk(outputs_pred, 3)
    #print(top3_pos)
    img_id = top3_pos[0][0].item()
    img_name = warship_id_name[img_id]
    #print(img_name)
    #print("该图片是：", img_name)
    return img_id,img_name

#load image from path as PIL image
def load_image(path):
    return Image.open(path)

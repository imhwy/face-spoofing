# import libaries
import cv2
import torch
from torchvision import transforms
from source.Model import DeePixBiS
from facenet_pytorch import MTCNN


# face spoofing detector
def spoofing_detector(faceRegion, tfms, model):
    faceRegion = cv2.cvtColor(faceRegion, cv2.COLOR_BGR2RGB)
    faceRegion = tfms(faceRegion)
    faceRegion = faceRegion.unsqueeze(0)
    mask, _ = model.forward(faceRegion)
    res = torch.mean(mask).item()
    return "Real" if res >= 0.5 else "False"

# demo


def main():
    # loading model
    model = DeePixBiS()
    model.load_state_dict(torch.load('./source/DeePixBiS.pth'))
    model.eval()
    # preprocessing
    tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ##############################################################################
    # TODO: implement code ở đây
    # code here
    # TODO: call function spoofing_detector và trả về kết quả
    # code here
    ##############################################################################


if __name__ == "__main__":
    main()

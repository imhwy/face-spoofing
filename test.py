# import libaries
import cv2
import torch
from torchvision import transforms
from source.Model import DeePixBiS
from facenet_pytorch import MTCNN


# face spoofing detector
def spoofing_detector(faceRegion, tfms, model):
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
    # applying mtcnn
    mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True)
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        isSuccess, frame = camera.read()
        if isSuccess:
            boxes, _, _ = mtcnn.detect(frame, landmarks=True)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int, box.tolist()))
                    faceRegion = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    faceRegion = cv2.cvtColor(
                        faceRegion, cv2.COLOR_BGR2RGB)
                    result = spoofing_detector(faceRegion, tfms, model)
                    frame = cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    frame = cv2.putText(
                        frame, result, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2, cv2.LINE_8)
        cv2.imshow("face spoofing", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import torch
import torchvision
import cv2

device = True if torch.cuda.is_available() else False

model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name='celebAHQ-512',
                            pretrained=True, useGPU=device)

noise, _ = model.buildNoiseData(1)

with torch.no_grad() :
        images = model.test(noise)

image = torchvision.utils.make_grid(images.clamp(min=-1, max=1), scale_each=True, normalize=True)


image = image.permute(1, 2, 0).cpu().numpy ()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cv2.imshow ('Image:', image)
cv2.waitKey (0)

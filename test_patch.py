import data
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from models import imagebind_model
from models.imagebind_model import ModalityType


import torch.nn as nn

def imshow_save(img, pt):
    pth = 'images_patch'+'_' + pt + '_' '.png'
    img_pth = os.path.join('/home/pbz/ImageBind/adv_images', pth)
    timg = torch.clone(img).cpu()
    timg[0] = img[0] * 0.26862954 + 0.48145466
    timg[1] = img[1] * 0.26130258 + 0.4578275
    timg[2] = img[2] * 0.27577711 + 0.40821073
    npimg = timg.detach().numpy()  # convert from tensor
    # print(npimg.min())
    # print(npimg.max())
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(img_pth)

def sticker_mask(images, patch_width, patch_height):
    image_height, image_width = images.shape[2:4]
    height_to_pad = int(image_height - patch_height)
    width_to_pad = int(image_width - patch_width)


    height_offsets = torch.randint(
        low=0, high=height_to_pad - 1, size=[images.shape[0]], dtype=torch.int32
    )

    width_offsets = torch.randint(
        low=0, high=width_to_pad - 1, size=[images.shape[0]], dtype=torch.int32
    )
    # sticker = torch.zeros(images.shape)
    sticker = torch.zeros(images.shape, requires_grad=True)

    print (height_offsets, width_offsets)

    with torch.no_grad():
        for i in range(images.shape[0]):
            sticker[i, :, width_offsets[i]: width_offsets[i] + patch_width,
            height_offsets[i]: height_offsets[i] + patch_width] = 1



    return  sticker




def pgd_attack(model, images, labels, patch_width = 50, patch_height = 50, alpha=2 / 255, iters=60, target = False):
    loss = nn.CrossEntropyLoss()

    # ori_images = images['vision']

    # images['vision'].requires_grad = True
    min_pix = images['vision'].min()
    max_pix = images['vision'].max()

    delta = torch.rand_like(images['vision'], requires_grad=True).to(device)

    sticker = sticker_mask(images['vision'], patch_width, patch_height).to(device)

    images['vision'] = images['vision'].detach() * (1 - sticker) + ((delta.detach()) * sticker)


    for i in range(iters):
        images['vision'] = images['vision'].detach().clone()
        images['vision'].requires_grad = True

        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs['vision'], labels).to(device)
        cost.backward()

        print(cost)

        print(images['vision'].grad.shape)


        if target:
            images['vision'] = images['vision'] - alpha * images['vision'].grad.sign() * sticker
        else:
            images['vision'] = images['vision'] + alpha * images['vision'].grad.sign() * sticker



        # images['vision'] = images['vision'] + alpha * images['vision'].grad.sign() * sticker
        # eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images['vision'] = torch.clamp(images['vision'], min=min_pix, max=max_pix).detach_()
        print (i)


    return images['vision']

# text_list=["A dog.", "A car", "A bird"]
# image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
# audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

text_list=["A dog.", "A car", "A bird", "A cat", "A keyboard", "A rain", "Wave", "A train"]
# image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
image_paths=[".assets/wave_image.jpg"]
target_image = [".assets/train_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav", ".assets/cat_audio.wav", ".assets/keyboard_audio.wav", ".assets/rain_audio.wav", ".assets/wave_audio.wav", ".assets/train_audio.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

images = {
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device)
}

target_images = {
    ModalityType.VISION: data.load_and_transform_vision_data(target_image, device)
}

texts = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device)
}

audios = {
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device)
}



with torch.no_grad():
    labels = model(target_images)['vision']
    # labels = model(texts)['text']

# print("Inputs Size of Text: ",inputs['text'].shape)
# print("Inputs Size of Vision: ",inputs['vision'].shape)
# print("Inputs Size of Audio: ",inputs['audio'].shape)
#
# print("Embeddings Size of Text: ",embeddings['text'].shape)
# print("Embeddings Size of Vision: ",embeddings['vision'].shape)
# print("Embeddings Size of Audio: ",embeddings['audio'].shape)
imshow_save(inputs['vision'][0], 'clean')

inputs['vision'] = pgd_attack(model, images, labels, alpha=2 / 255, iters=1000, target= True)

imshow_save(inputs['vision'][0], 'adv')

with torch.no_grad():
    embeddings = model(inputs)
# print(embeddings)

# print(embeddings['vision'].shape)

print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)
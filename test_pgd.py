import data
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from models import imagebind_model
from models.imagebind_model import ModalityType




import torch.nn as nn


# mean=(0.48145466, 0.4578275, 0.40821073),
# std=(0.26862954, 0.26130258, 0.27577711),

def imshow_save(img, pt):
    pth = 'images'+'_' + pt + '_' '.png'
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

def pgd_attack(model, images, labels, eps=0.2, alpha=2 / 255, iters=60, target = False):
    loss = nn.CrossEntropyLoss()

    ori_images = images['vision']

    # images['vision'].requires_grad = True
    min_pix = images['vision'].min()
    max_pix = images['vision'].max()


    for i in range(iters):
        # images['vision'] = images['vision'].detach().clone()
        images['vision'].requires_grad = True

        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs['vision'], labels).to(device)
        cost.backward()

        # print (cost)
        #
        # print(images['vision'].grad.shape)

        if target:
            adv_images = images['vision'] - alpha * images['vision'].grad.sign()
        else:
            adv_images = images['vision'] + alpha * images['vision'].grad.sign()

        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images['vision'] = torch.clamp(ori_images + eta, min=min_pix, max=max_pix).detach_()
        # print (i)


    return images['vision']




# text_list=["A dog.", "A car", "A bird"]
# # image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
# image_paths=[".assets/dog_image.jpg"]
# audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

# text_list=["A dog.", "A car", "A bird", "A cat", "A keyboard", "A rain", "Wave", "A train"]
# # image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
# image_paths=[".assets/Sea_waves.jpg"]
# target_image = [".assets/train_image.jpg"]
# audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav", ".assets/cat_audio.wav", ".assets/keyboard_audio.wav", ".assets/rain_audio.wav", ".assets/wave_audio.wav", ".assets/train_audio.wav"]


text_list=["Dog", 	"Rain", 	"Crying baby", 	"Door wood knock", 	"Helicopter",
"Rooster",	"Sea waves",	"Sneezing",	"Mouse click",	"Chainsaw",
"Pig",	"Crackling fire",	"Clapping",	"Keyboard typing",	"Siren",
"Cow",	"Crickets",	"Breathing",	"Door, wood creaks",	"Car horn",
"Frog",	"Chirping birds",	"Coughing",	"Can opening",	"Engine",
"Cat",	"Water drops",	"Footsteps",	"Washing machine",	"Train",
"Hen",	"Wind",	"Laughing",	"Vacuum cleaner",	"Church bells",
"Insects (flying)",	"Pouring water",	"Brushing teeth",	"Clock alarm",	"Airplane",
"Sheep",	"Toilet flush",	"Snoring",	"Clock tick",	"Fireworks",
"Crow",	"Thunderstorm",	"Drinking, sipping",	"Glass breaking",	"Hand saw"]
# image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
image_paths=[".assets/bird_image.jpg"]
target_image = [".assets/train_image.jpg"]
audio_paths=[".assets/audio/dog_audio.wav", ".assets/audio/Rain.wav", ".assets/audio/Crying_baby.wav", ".assets/audio/door_wood_knock.wav", ".assets/audio/Helicopter.wav",
             ".assets/audio/Rooster.wav", ".assets/audio/wave_audio.wav", ".assets/audio/Sneezing.wav", ".assets/audio/mouse_click.wav", ".assets/audio/Chainsaw.wav",
".assets/audio/Pig.wav", ".assets/audio/Crackling_fire.wav", ".assets/audio/Clapping.wav", ".assets/audio/keyboard_audio.wav", ".assets/audio/Siren.wav",
".assets/audio/Cow.wav", ".assets/audio/Crickets.wav", ".assets/audio/Breathing.wav", ".assets/audio/door_wood_creaks.wav", ".assets/audio/car_horn.wav",
".assets/audio/Frog.wav", ".assets/audio/bird_audio.wav", ".assets/audio/Coughing.wav", ".assets/audio/can_opening.wav", ".assets/audio/Engine.wav",
".assets/audio/cat_audio.wav", ".assets/audio/Water_drops.wav", ".assets/audio/Footsteps.wav", ".assets/audio/washing_machine.wav", ".assets/train_audio.wav",
".assets/audio/Hen.wav", ".assets/audio/Wind.wav", ".assets/audio/Laughing.wav", ".assets/audio/vacuum_cleaner.wav", ".assets/audio/church_bells.wav",
".assets/audio/Insects.wav", ".assets/audio/Pouring_water.wav", ".assets/audio/Brushing_teeth.wav", ".assets/audio/clock_alarm.wav", ".assets/audio/Airplane.wav",
".assets/audio/Sheep.wav", ".assets/audio/Toilet_flush.wav", ".assets/audio/Snoring.wav", ".assets/audio/clock_tick.wav", ".assets/audio/Fireworks.wav",
".assets/audio/Crow.wav", ".assets/audio/Thunderstorm.wav", ".assets/audio/drinking_sipping.wav", ".assets/audio/glass_breaking.wav", ".assets/audio/hand_saw.wav"]



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

imshow_save(inputs['vision'][0], 'clean')

with torch.no_grad():
    labels = model(target_images)['vision']
    # labels = model(audios)['audio']
    # labels = model(texts)['text']

# print("Inputs Size of Text: ",inputs['text'].shape)
# print("Inputs Size of Vision: ",inputs['vision'].shape)
# print("Inputs Size of Audio: ",inputs['audio'].shape)
#
# print("Embeddings Size of Text: ",embeddings['text'].shape)
# print("Embeddings Size of Vision: ",embeddings['vision'].shape)
# print("Embeddings Size of Audio: ",embeddings['audio'].shape)

inputs['vision'] = pgd_attack(model, images, labels, iters=100, target = True)

imshow_save(inputs['vision'][0], 'adv')

with torch.no_grad():
    embeddings = model(inputs)
# print(embeddings)

# print(embeddings['vision'].shape)

print (embeddings[ModalityType.VISION])

print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1).argmax() + 1,
)
print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1).argmax(1) + 1,
)
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1).argmax() + 1,
)
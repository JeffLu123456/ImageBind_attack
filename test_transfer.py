import data
import torch
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
from models import imagebind_model
from models.imagebind_model import ModalityType
from CLIP import clip
from test_pgd import imshow_save


import torch.nn as nn



# def pgd_attack(model, images, labels, eps=0.5, alpha=2 / 255, iters=60):
#     loss = nn.CrossEntropyLoss()
#
#     ori_images = images['vision']
#
#     # images['vision'].requires_grad = True
#     min_pix = images['vision'].min()
#     max_pix = images['vision'].max()
#
#
#     for i in range(iters):
#         # images['vision'] = images['vision'].detach().clone()
#         images['vision'].requires_grad = True
#
#         outputs = model(images)
#
#         model.zero_grad()
#         cost = loss(outputs['vision'], labels).to(device)
#         cost.backward()
#
#         print (cost)
#
#         print(images['vision'].grad.shape)
#
#
#         adv_images = images['vision'] + alpha * images['vision'].grad.sign()
#         eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
#         images['vision'] = torch.clamp(ori_images + eta, min=min_pix, max=max_pix).detach_()
#         print (i)
#
#
#     return images['vision']

def pgd_attack(model, images, labels, eps=0.2, alpha=2 / 255, iters=60, target = False):
    loss = nn.CrossEntropyLoss()
    # loss = nn.MSELoss()

    ori_images = images

    # images['vision'].requires_grad = True
    min_pix = images.min()
    max_pix = images.max()

    labels = torch.softmax(labels, dim=-1)


    for i in range(iters):
        # images['vision'] = images['vision'].detach().clone()
        images.requires_grad = True

        # outputs = model(images)
        outputs = model.encode_image(images)

        outputs = torch.softmax(outputs, dim=-1)


        model.zero_grad()
        # cost = loss(outputs, labels).to(device)
        cost = torch.nn.functional.kl_div(outputs.log(), labels, reduction='sum')
        cost.backward()

        # print (cost)
        #
        # print(images['vision'].grad.shape)

        if target:
            adv_images = images- alpha * images.grad.sign()
        else:
            adv_images = images + alpha * images.grad.sign()

        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=min_pix, max=max_pix).detach_()
        # print (i)


    return images

# text_list=["A dog", "A car", "A bird", "A cat", "A keyboard", "A rain", "Wave", "A train"]
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
# target_image = [".assets/dog_image.jpg", "./ESC50/Dog/1.jpg", "./ESC50/Dog/2.jpg", "./ESC50/Dog/3.jpg", "./ESC50/Dog/1684399870997.jpg",
#                 "./ESC50/Dog/1684399891637.jpg", "./ESC50/Dog/1684399988599.jpg"]
# target_image = [".assets/train_image.jpg", "./ESC50/Train/1.jpg", "./ESC50/Train/1684399924533.jpg"]
target_image = ["./ESC50/Chirping birds/1.jpg", "./ESC50/Chirping birds/2.jpg", "./ESC50/Chirping birds/3.jpg"]
target_txt = ["Airplane"]
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

texts = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device)
}

audios = {
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device)
}

target_images = {
    ModalityType.VISION: data.load_and_transform_vision_data(target_image, device)
}

target_txts = {
    ModalityType.TEXT: data.load_and_transform_text(target_txt, device)
}



# with torch.no_grad():
#     labels = model(images)['vision']
    # labels = model(audios)['audio']
    # labels = model(texts)['text']

# print("Inputs Size of Text: ",inputs['text'].shape)
# print("Inputs Size of Vision: ",inputs['vision'].shape)
# print("Inputs Size of Audio: ",inputs['audio'].shape)
#
# print("Embeddings Size of Text: ",embeddings['text'].shape)
# print("Embeddings Size of Vision: ",embeddings['vision'].shape)
# print("Embeddings Size of Audio: ",embeddings['audio'].shape)

# inputs['vision'] = pgd_attack(model, images, labels, iters=1000)

with torch.no_grad():
    labels = model(target_images)['vision']

# with torch.no_grad():
#     labels = model(target_txts)['text']
    labels = torch.mean(labels, dim=0, keepdim=True)

with torch.no_grad():
    embeddings = model(inputs)
# print(embeddings)

# print(embeddings['vision'].shape)

# print (embeddings[ModalityType.VISION])

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


clip_model, preprocess = clip.load("ViT-B/32", device=device)

t_image = preprocess(Image.open(".assets/dog_image.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(text_list).to(device)
t_text = clip.tokenize(target_txt).to(device)

with torch.no_grad():
    timage_features = clip_model.encode_image(t_image)
    ttext_features = clip_model.encode_text(t_text)

    text_features = clip_model.encode_text(text)

#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

paths = glob('/home/pbz/ImageBind/ESC50/*')

for i in tqdm(range(len(paths))):
    path = paths[i]
    print(path)
    path = os.path.join(path, '*')
    images_paths = glob(path)
    for j in range(len(images_paths)):
        inputs_images = preprocess(Image.open(images_paths[j])).unsqueeze(0).to(device)

        print (inputs_images.shape)

        # imshow_save(inputs_images[0], 'clean')
        print (images_paths[j])
        # with torch.no_grad():
        #     label = clip_model.encode_image(inputs_images)

        # print ("_____________________new data__________________________________")

        # inputs = {
        #     ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        #     ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
        #     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
        # }

        adv_images = {
            ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
        }


        # print ("---------------------------------------------")
        # print (inputs_images.shape)
        # print ("---------------------------------------------")

        # adv_images['vision'] = pgd_attack(clip_model, inputs_images, timage_features, iters=200, target=True)
        adv_images['vision'] = pgd_attack(clip_model, inputs_images, ttext_features, iters=150, target=False)

        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(adv_images['vision'], text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy().argmax() + 1


        print("Label probs:", probs)
        # imshow_save(adv_images['vision'][0], 'adv')

        # images = {
        #     ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device)
        # }

        with torch.no_grad():
            embeddings_images = model(adv_images)

        print(
            "Vision x Text: ",
            torch.softmax(embeddings_images[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1).argmax() + 1,
        )
        # print(
        #     "Audio x Text: ",
        #     torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1).argmax(1) + 1,
        # )
        print(
            "Vision x Audio: ",
            torch.softmax(embeddings_images[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1).argmax() + 1,
        )


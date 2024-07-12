# import clip
# import torch
# import numpy as np
# from PIL import Image
# from transformers import CLIPFeatureExtractor, CLIPVisionModel
# from data.utils import load_txt
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# encoder_name = '/home/liuyuanqiu/code/ckpts/openai/clip-vit-base-patch32'
# feature_extractor1 = CLIPFeatureExtractor.from_pretrained(encoder_name) 
# print(feature_extractor1)
# # clip_encoder = CLIPVisionModel.from_pretrained(encoder_name).to(device)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # x = torch.randn(500000, 1024).to(device)
# # y = torch.randn(64, 1024).to(device)
# # c = torch.matmul(y, x.T)
# clip_model, feature_extractor = clip.load("RN50x64", device=device)
# print(feature_extractor)
# data_dir = "/home/liuyuanqiu/code/mscoco/feature/coco2014/train2014/"
# file_names = ["COCO_train2014_000000340964.jpg", "COCO_train2014_000000405520.jpg"]
# images = [Image.open(data_dir + file_name).convert("RGB") for file_name in file_names]

# vocab = load_txt("/home/liuyuanqiu/code/mscoco/txt/coco_vocabulary.txt")
# vocab = ['A bead and chair in a long abandoned room.', 'A beach scene with people carrying their surfboards.']
# with torch.no_grad():
#     image_features = []
#     image_input = [feature_extractor(Image.open(data_dir + file_name).convert("RGB")) for file_name in file_names]
#     # print(torch.tensor(np.stack(image_input)).to(device).shape)
#     image_features.append(clip_model.encode_image(torch.tensor(np.stack(image_input)).to(device)).cpu().numpy())
#     image_features = np.concatenate(image_features)
#     print(image_features.shape)
#     encoded_captions = []
#     input_ids = clip.tokenize(vocab).to(device)
#     encoded_captions.append(clip_model.encode_text(input_ids).cpu().numpy())
#     # input_ids = clip.tokenize(vocab[5000:]).to(device)
#     # encoded_captions.append(clip_model.encode_text(input_ids).cpu().numpy())
#     encoded_captions = np.concatenate(encoded_captions)

#     scores = image_features.dot(encoded_captions.T)
#     print(image_features.shape, encoded_captions.shape)
#     print(scores.shape)
#     print(scores)
#     # s = torch.from_numpy(scores).to(device)
#     # print(s.topk(20))
import numpy as np
swin_dert_region_path = "/home/liuyuanqiu/code/mscoco/feature/up_down_36/9.npz"
with np.load(swin_dert_region_path, allow_pickle=True) as data:
    region = data['feat']
    region = np.array(region).astype('float32')
    print(region.shape)
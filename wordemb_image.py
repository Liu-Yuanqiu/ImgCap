import os
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# fname = os.path.join('./ckpts', 'transformer', 'transformer', '%s_best.pth' % 'transformer')
# data = torch.load(fname)
# weights = data['state_dict']
# # for k,v in weights.items():
# #     print(k)

# word_emb = weights['decoder.word_emb.weight']
# fc = weights['decoder.fc.weight']

# # 然后，应用t-SNE
# word_emb_2d = TSNE(n_components=2, random_state=42).fit_transform(word_emb.cpu().numpy())
# fc_2d = TSNE(n_components=2, random_state=42).fit_transform(fc.cpu().numpy())

# plt.figure(figsize=(10, 8), dpi=150)
# plt.scatter(word_emb_2d[:, 0], word_emb_2d[:, 1], s=2)
# plt.title('t-SNE Projection of High-Dimensional Vectors')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.savefig('word_emb.jpg')
# plt.show()
# plt.close()

# plt.figure(figsize=(10, 8), dpi=150)
# plt.scatter(fc_2d[:, 0], fc_2d[:, 1], s=2)
# plt.title('t-SNE Projection of High-Dimensional Vectors')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.savefig('fc.jpg')
# plt.show()
# plt.close()

fname = os.path.join('./ckpts', 's2s', 'eed_pe_gen20_sorted_mse_newwordemb', '%s_best.pth' % 's2s')
data = torch.load(fname)
weights = data['state_dict']
for k,v in weights.items():
    print(k)
new_word_emb = weights['word_emb.weight']
new_word_emb_2d = TSNE(n_components=2, random_state=42).fit_transform(new_word_emb.cpu().numpy())
plt.figure(figsize=(10, 8), dpi=150)
plt.scatter(new_word_emb_2d[:, 0], new_word_emb_2d[:, 1], s=2)
plt.title('eed_pe_gen20_sorted_mse_newwordemb_word_emb')
plt.savefig('eed_pe_gen20_sorted_mse_newwordemb_word_emb.jpg')
plt.show()
plt.close()

# # 绘制散点图
# plt.figure(figsize=(10, 8), dpi=150)
# plt.scatter(word_emb_2d[:, 0], word_emb_2d[:, 1], s=2)
# plt.scatter(fc_2d[:, 0], fc_2d[:, 1], s=2)
# plt.scatter(new_word_emb_2d[:, 0], new_word_emb_2d[:, 1], s=2)
# plt.title('t-SNE Projection of High-Dimensional Vectors')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.savefig('all.jpg')
# plt.show()
# plt.close()

# Llama word_emb fc 可视化
# base_model = "/home/liuyuanqiu/code/ckpts/llama-2-7b-hf"
# model1 = torch.load(os.path.join(base_model, 'pytorch_model-00001-of-00002.bin'))
# embed_tokens = model1['model.embed_tokens.weight']
# model2 = torch.load(os.path.join(base_model, 'pytorch_model-00002-of-00002.bin'))
# lm_head = model2['lm_head.weight']

# lm_head_2d= TSNE(n_components=2, random_state=42).fit_transform(lm_head.detach().numpy())
# embed_tokens_2d= TSNE(n_components=2, random_state=42).fit_transform(embed_tokens.detach().numpy())
# plt.figure(figsize=(10, 8), dpi=150)
# plt.scatter(lm_head_2d[:, 0], lm_head_2d[:, 1], s=2)
# plt.scatter(embed_tokens_2d[:, 0], embed_tokens_2d[:, 1], s=2)
# plt.title('Llama word_emb and fc')
# plt.savefig('llama_word_emb_fc.jpg')
# plt.show()
# plt.close()

# plt.figure(figsize=(10, 8), dpi=150)
# plt.scatter(lm_head_2d[:, 0], lm_head_2d[:, 1], s=2)
# plt.title('Llama fc')
# plt.savefig('llama_fc.jpg')
# plt.show()
# plt.close()

# plt.figure(figsize=(10, 8), dpi=150)
# plt.scatter(embed_tokens_2d[:, 0], embed_tokens_2d[:, 1], s=2)
# plt.title('Llama word_emb')
# plt.savefig('llama_word_emb.jpg')
# plt.show()
# plt.close()
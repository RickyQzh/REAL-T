# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''ResNet in PyTorch.

Some modifications from the original architecture:
1. Smaller kernel size for the input layer
2. Smaller number of Channels
3. No max_pooling involved

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import pooling_layers as pooling_layers
import torchaudio.compliance.kaldi as kaldi


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 m_channels=32,
                 feat_dim=40,
                 embed_dim=128,
                 pooling_func='TSTP',
                 two_emb_layer=True):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2)

        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim,
                               embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        stats = self.pool(out)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a



def ResNet18(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet34(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet50(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet101(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet152(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [3, 8, 36, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet221(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [6, 16, 48, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet293(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [10, 20, 64, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def compute_fbank(waveform):
    waveform = waveform * (1 << 15)
    waveform = waveform.unsqueeze(1)
    result = None
    
    for i in range(waveform.shape[0]):
        if result is None:
            result = kaldi.fbank(waveform[i],
                                num_mel_bins=80,
                                frame_length=25,
                                frame_shift=10,
                                dither=1.0,
                                sample_frequency=16000,
                                window_type='hamming',
                                use_energy=False).unsqueeze(0)
        else:
            result = torch.cat([result, kaldi.fbank(waveform[i],
                                num_mel_bins=80,
                                frame_length=25,
                                frame_shift=10,
                                dither=1.0,
                                sample_frequency=16000,
                                window_type='hamming',
                                use_energy=False).unsqueeze(0)], 0)
    return result


def apply_cmvn(sample, norm_mean=True, norm_var=False):
    for i in range(sample.shape[0]):
        mat = sample[i]
        if norm_mean:
            mat = mat - torch.mean(mat, dim=0)
        if norm_var:
            mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
        sample[i] = mat
    return sample


class ModelWrapper(nn.Module):
    def __init__(self,
                 embedding_model=ResNet34(feat_dim=80, embed_dim=256, pooling_func='TSTP', two_emb_layer=False),
                 ):
        super(ModelWrapper, self).__init__()
        self.embedding_model = embedding_model

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # compute fbank
        fbank = compute_fbank(x)
        # apply cmvn
        fbank = apply_cmvn(fbank)
        embedding = self.embedding_model(fbank)

        return embedding[-1]


if __name__ == '__main__':
    model = ModelWrapper()
    cpt = torch.load("/mnt/yanxiaopeng/wespeaker/examples/voxceleb/v2/exp/voxceleb_resnet34_LM.pt", map_location='cpu')
    
    for i in list(cpt):
        if i=='projection.weight':
            del cpt[i]

    model.embedding_model.load_state_dict(cpt)
    
    try:
        device = torch.device("cuda")
        model = model.to(device)
    except Exception as e:
        print(e)
        device = torch.device("cpu")
    model.eval()

    # x = torch.randn(16, 16000*3).to(device=device)
    # y = model(x)
    # print(y.shape)

    # num_params = sum(p.numel() for p in model.parameters())
    # print("{} M".format(num_params / 1e6))
    
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model, (48000,), as_strings=True, print_per_layer_stat=True, verbose=True)
    # print(macs, params)
    
    import soundfile as sf
    import librosa
    import numpy as np
    
    import warnings 
    warnings.filterwarnings("ignore", category=FutureWarning) 
    
    audio_1,sr = sf.read("/mnt/data_set/yanxiaopeng/DNS-Challenge/raw/clean/english/librivox/read_speech_complete_reader_00003_0.wav")
    if sr != 16000:
        audio_1 = librosa.resample(audio_1, sr, 16000)
    audio_1 = audio_1.astype(np.float32)
    audio_1 = torch.tensor(audio_1).to(device=device)

    audio_2,sr = sf.read("/mnt/data_set/yanxiaopeng/DNS-Challenge/raw/clean/english/librivox/read_speech_complete_reader_00003_1.wav")
    if sr != 16000:
        audio_2 = librosa.resample(audio_2, sr, 16000)
    audio_2 = audio_2.astype(np.float32)
    audio_2 = torch.tensor(audio_2).to(device=device)
    
    # audio_2,sr = sf.read("/mnt/data_set/yanxiaopeng/DNS-Challenge/raw/clean/english/librivox/read_speech_complete_reader_00007_0.wav")
    # if sr != 16000:
    #     audio_2 = librosa.resample(audio_2, sr, 16000)
    # audio_2 = audio_2.astype(np.float32)
    # audio_2 = torch.tensor(audio_2).to(device=device)
    
    def cosine_loss(train_embedding, test_embedding):
        score = torch.cosine_similarity(train_embedding, test_embedding, dim=1)
        return score
    
    print(cosine_loss(model(audio_1), model(audio_2)))
    
    from speechbrain.pretrained import EncoderClassifier 
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    embeddings_1 = classifier.encode_batch(audio_1).squeeze(1)
    embeddings_2 = classifier.encode_batch(audio_2).squeeze(1)
    print(cosine_loss(embeddings_1, embeddings_2))
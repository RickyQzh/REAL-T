from __future__ import print_function
from typing import Tuple
import math

import numpy as np
import torch, torchaudio, os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from wesep.utils.funcs import compute_fbank,apply_cmvn

from wesep.modules.common.spkadapt import SpeakerFuseLayer, CrossFuseLayer
from wesep.modules.common.spkadapt import SpeakerTransform
from wesep.modules.speaker.get_speaker import PreEmphasis, get_speaker_model_cross, LDEPooling, ASTP
from wesep.modules.speaker.loadwhisper import load_init_whisper
# from wespeaker.models.speaker_model import get_speaker_model


# from thop import profile, clever_format

class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps

        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * 2, input_size)  # hidden_size = feature_dim * 2

    def forward(self, input):
        # input shape: batch, dim, seq

        rnn_output, _ = self.rnn(self.norm(input).transpose(1, 2).contiguous())
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(input.shape[0],
                                                                                           input.shape[2],
                                                                                           input.shape[1])

        return input + rnn_output.transpose(1, 2).contiguous()


"""
TODO : attach the speaker embedding to each input
Input shape:(B,feature_dim + spk_emb_dim , T)
"""

class BSNet(nn.Module):
    def __init__(self, in_channel, nband=7, bidirectional=True):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband
        self.band_rnn = ResRNN(self.feature_dim, self.feature_dim * 2, bidirectional=bidirectional)
        self.band_comm = ResRNN(self.feature_dim, self.feature_dim * 2, bidirectional=bidirectional)

    '''
    permute() is meant to rearranges the dimension according to the provided order
    e.g. (A,B,C,D).permute(0,3,2,1) = (A,D,C,B)

    contiguous() is used to ensure that the memory layout of the tensor is contiguous
    '''

    def forward(self, input):
        # input shape: B, nband*N, T
        B, N, T = input.shape

        band_output = self.band_rnn(
            input.view(B * self.nband, self.feature_dim, -1)).view(B, self.nband, -1, T)

        # band comm
        band_output = band_output.permute(0, 3, 2, 1).contiguous().view(B * T, -1, self.nband)
        output = self.band_comm(band_output).view(B, T, -1, self.nband).permute(0, 3, 2, 1).contiguous()

        return output.view(B, N, T)

class CrossAtt(nn.MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super(CrossAtt, self).__init__(*args, **kwargs)

    def forward(self, query, key, value):
        if query.dim() == 4:
            spk_embeddings = []
            for i in range(query.shape[1]):
                x = query[:,i,:,:].squeeze(dim=1)
                x, _ = super().forward(x.transpose(1,2), key.transpose(1,2), value.transpose(1,2))
                spk_embeddings.append(x.transpose(1,2))
            spk_embeddings = torch.stack(spk_embeddings, 1)
        elif query.dim() == 3:
            x, _ = super().forward(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2))
            spk_embeddings = x.transpose(1,2)
        return spk_embeddings
        
class FuseSeparation(nn.Module):
    def __init__(self, nband=7, num_repeat=6, feature_dim=128, spk_emb_dim=256, emb_feat=True, spk_fuse_type='concat', multi_fuse=False, spk_model = None, mode='simple', frame_type='last'):
        """

        :param nband : len(self.band_width)
        """
        super(FuseSeparation, self).__init__()
        self.multi_fuse = multi_fuse
        self.nband = nband 
        self.feature_dim = feature_dim
        self.spk_fuse_type = spk_fuse_type
        self.mode = mode
        self.emb_feat = emb_feat

        atten_dim = feature_dim # feature_dim Or spk_emb_dim.  Not sure about this. 
        if spk_fuse_type.startswith('cross') and emb_feat:
            if spk_model.startswith('ECAPA_TDNN_GLOB_c512'):
                if frame_type=='global':
                    spk_emb_frame_dim = 1536
                elif frame_type=='fbank':
                    spk_emb_frame_dim = 80
                else: #frame_type=='last':
                    spk_emb_frame_dim = 512
            elif spk_model.startswith('ResNet'):
                if frame_type=='last':
                    spk_emb_frame_dim = 256
                elif frame_type=='fbank':
                    spk_emb_frame_dim = 80
                elif frame_type=='last_3':
                    spk_emb_frame_dim = 128
                elif frame_type=='last_2':
                    spk_emb_frame_dim = 64
                elif frame_type=='last_1':
                    spk_emb_frame_dim = 32
                elif frame_type=='last_0':
                    spk_emb_frame_dim = 32
                else:
                    spk_emb_frame_dim = 256
            elif spk_model.startswith('CAMPPlus'):
                spk_emb_frame_dim = 512
            elif spk_model.startswith('Part_Whisper'):
                spk_emb_frame_dim = 1280 * 8            # Should adjust based on (l2-l1+1) * 1280
            self.attenFuse = nn.ModuleList([])
            if self.multi_fuse:
                num_repeat_fuse = num_repeat
            else:
                num_repeat_fuse = 1
            for i in range(num_repeat_fuse):
                if mode=='concat_before':
                    self.attenFuse.append(nn.Linear(spk_emb_frame_dim + spk_emb_dim, feature_dim))
                elif mode=='concat_after':
                    self.attenFuse.append(nn.Linear(spk_emb_frame_dim, feature_dim))
                elif mode=='simple':
                    self.attenFuse.append(nn.Linear(spk_emb_frame_dim, feature_dim))
                self.attenFuse.append(CrossAtt(embed_dim=atten_dim, num_heads=2, batch_first=True))

        utt_feat = False
        if emb_feat:
            if not spk_fuse_type.startswith('cross'):
                FuseLayer = SpeakerFuseLayer
            else:
                if spk_fuse_type.startswith('cross_both'):
                    FuseLayer_utt = SpeakerFuseLayer
                    utt_feat = True
                    spk_split = spk_fuse_type.split('_')
                    spk_fuse_type = spk_split[0] + '_' + spk_split[-1]
                    spk_utt = spk_split[-1]
                    spk_emb_dim_utt = spk_emb_dim
                FuseLayer = CrossFuseLayer
                if mode=='concat_after':
                    spk_emb_dim += atten_dim
                else:
                    spk_emb_dim = atten_dim

        self.utt_feat = utt_feat
        self.separation = nn.ModuleList([])
        if self.multi_fuse and emb_feat:
            for i in range(num_repeat):
                self.separation.append(FuseLayer(embed_dim=spk_emb_dim, feat_dim=feature_dim, fuse_type=spk_fuse_type))
                self.separation.append(BSNet(nband * feature_dim, nband))
        else:
            if emb_feat:
                self.separation.append(FuseLayer(embed_dim=spk_emb_dim, feat_dim=feature_dim, fuse_type=spk_fuse_type))
                if utt_feat:
                    self.separation.append(FuseLayer_utt(embed_dim=spk_emb_dim_utt, feat_dim=feature_dim, fuse_type=spk_utt))
            for i in range(num_repeat):
                self.separation.append(BSNet(nband * feature_dim, nband))

    def forward(self, x, spk_embedding, spk_emb_frame=None, nch=1):
        '''
           x: [B, nband, feature_dim, T]
           out: [B, nband, feature_dim, T]
        '''
        batch_size = x.shape[0]

        if self.mode == 'concat_before' and self.spk_fuse_type.startswith('cross') and self.emb_feat:
            spk_emb_frame = torch.cat([spk_emb_frame, torch.unsqueeze(spk_embedding, -1).repeat(1,1,spk_emb_frame.shape[-1])], 1)

        if self.multi_fuse and self.emb_feat:
            for i in range(len(self.separation)):
                if i % 2 == 0:
                    if self.spk_fuse_type.startswith('cross'):
                        spk_embedding_feed = torch.transpose(self.attenFuse[i](torch.transpose(spk_emb_frame,1,2)),1,2)
                        spk_embedding_feed = self.attenFuse[i+1](x, spk_embedding_feed, spk_embedding_feed)
                        if self.mode == 'concat_after':
                            spk_embedding_un = spk_embedding.unsqueeze(1).unsqueeze(3)
                            spk_embedding_feed = torch.cat([spk_embedding_feed, spk_embedding_un.expand(-1, spk_embedding_feed.size(1), -1, spk_embedding_feed.size(3))], 2)
                    else:
                        spk_embedding_feed = spk_embedding
                    x = self.separation[i](x,spk_embedding_feed)
                    x = x.view(batch_size * nch, self.nband * self.feature_dim, -1)
                else:
                    x = self.separation[i](x)
                    x = x.view(batch_size * nch, self.nband, self.feature_dim, -1)      
        else:
            if self.emb_feat:
                if self.spk_fuse_type.startswith('cross'):
                    spk_embedding_feed = torch.transpose(self.attenFuse[0](torch.transpose(spk_emb_frame,1,2)),1,2)
                    spk_embedding_feed = self.attenFuse[1](x, spk_embedding_feed, spk_embedding_feed)
                    if self.utt_feat:
                        spk_embedding_un = spk_embedding.unsqueeze(1).unsqueeze(3)
                    if self.mode == 'concat_after':
                        spk_embedding_un = spk_embedding.unsqueeze(1).unsqueeze(3)
                        spk_embedding_feed = torch.cat([spk_embedding_feed, spk_embedding_un.expand(-1, spk_embedding_feed.size(1), -1, spk_embedding_feed.size(3))], 2)
                else:
                    spk_embedding_feed = spk_embedding
            if self.emb_feat:
                x = self.separation[0](x,spk_embedding_feed)
                if self.utt_feat:
                    x = self.separation[1](x,spk_embedding_un)
            x = x.view(batch_size * nch, self.nband * self.feature_dim, -1)
            if self.emb_feat:
                if self.utt_feat:
                    for i in range(2, len(self.separation)):
                        x = self.separation[i](x)
                else:
                    for i in range(1, len(self.separation)):
                        x = self.separation[i](x)
            else:
                for i in range(len(self.separation)):
                    x = self.separation[i](x)
            x = x.view(batch_size * nch, self.nband, self.feature_dim, -1)    
        return x

class BSRNN_HR(nn.Module):
    # self, sr=16000, win=512, stride=128, feature_dim=128, num_repeat=6, use_bidirectional=True
    def __init__(self,
                 spk_emb_dim=256,
                 sr=16000,
                 win=512,
                 stride=128,
                 feature_dim=128,
                 num_repeat=6,
                 use_spk_transform=True,
                 use_bidirectional=True,
                 emb_score=False,
                 time_feat=False,
                 tf_feat=False,
                 map_norm = True,
                 emb_feat=False,
                 utt_feat=False,
                 spk_fuse_type='concat',
                 multi_fuse = False,
                 joint_training=False,
                 multi_task=False,
                 spksInTrain=251,
                 spk_model=None,
                 spk_model_init=None,
                 spk_model_freeze=False,
                 spk_args=None,
                 spk_feat=False,
                 feat_type='consistent',
                 mode='simple',   #simple, concat_before, concat_after
                 spk_norm=False,
                 pooling=False,
                ):
        super(BSRNN_HR, self).__init__()

        self.sr = sr
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps
        self.spk_emb_dim = spk_emb_dim
        self.joint_training = joint_training
        self.spk_feat = spk_feat
        self.feat_type = feat_type
        self.spk_model_freeze = spk_model_freeze
        self.multi_task = multi_task
        self.pooling = pooling
        self.time_feat = time_feat
        self.tf_feat = tf_feat
        self.emb_feat = emb_feat
        self.emb_score = emb_score
        self.utt_feat = utt_feat
        self.map_norm = map_norm

        # # # 0-1k (100 hop), 1k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf

        # # 0-8k (1k hop), 8k-16k (2k hop), 16k
        # bandwidth_100 = int(np.floor(100 / (sr / 2.) * self.enc_dim))
        # bandwidth_200 = int(np.floor(200 / (sr / 2.) * self.enc_dim))
        # bandwidth_500 = int(np.floor(500 / (sr / 2.) * self.enc_dim))
        # bandwidth_2k = int(np.floor(2000 / (sr / 2.) * self.enc_dim))

        # # add up to 8k
        # self.band_width = [bandwidth_100] * 15
        # self.band_width += [bandwidth_200] * 10
        # self.band_width += [bandwidth_500] * 5
        # self.band_width += [bandwidth_2k] * 1

        # self.band_width.append(self.enc_dim - np.sum(self.band_width))

        self.band_width = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 16, 16, 16, 16, 16, 64]

        self.nband = len(self.band_width)
        self.spk_fuse_type = spk_fuse_type
        self.mode = mode
        self.spk_norm = spk_norm

        if self.spk_norm:
            self.scale = nn.Parameter(torch.ones(1))

        if use_spk_transform:
            self.spk_transform = SpeakerTransform()
        else:
            self.spk_transform = nn.Identity()

        if joint_training and (self.emb_feat or self.emb_score or self.utt_feat):
            # if not spk_fuse_type.startswith('cross'):
            #     self.spk_model = get_speaker_model(spk_model)(**spk_args)
            # else:
            self.spk_model = get_speaker_model_cross(spk_model)(**spk_args)
            if spk_model_init:
                if spk_model.startswith('Part_Whisper'):
                    load_init_whisper(self.spk_model, spk_model_init)
                else:
                    pretrained_model = torch.load(spk_model_init)
                    state =self.spk_model.state_dict()
                    for key in state.keys():
                        if key in pretrained_model.keys():
                            state[key] = pretrained_model[key]
                            # print(key)
                        else:
                            print("not %s loaded" % key)
                    self.spk_model.load_state_dict(state)
            if spk_model_freeze:
                for param in self.spk_model.parameters():
                    param.requires_grad = False
            if not spk_feat:
                if feat_type=='consistent':
                    self.preEmphasis = PreEmphasis()
                    self.spk_encoder= torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=win, win_length=win, hop_length=stride, \
                                                  f_min = 20, window_fn=torch.hamming_window, n_mels=spk_args['feat_dim'])
            if multi_task:
                self.pred_linear = nn.Linear(spk_emb_dim, spksInTrain)


        channel_map = 2
        if self.time_feat:
            channel_map += 1
        if self.tf_feat:
            channel_map += 1

        self.channel_map = channel_map

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(nn.Sequential(nn.GroupNorm(1, self.band_width[i] * channel_map, self.eps),
                                         nn.Conv1d(self.band_width[i] * channel_map, self.feature_dim, 1)
                                         )
                           )
            
        if spk_args and 'frame_type' in spk_args:
            frame_type = spk_args['frame_type']
        else:
            frame_type = None

        if frame_type=='global':
            spk_emb_frame_dim = 1536
        elif frame_type=='fbank':
            spk_emb_frame_dim = 80
        else: #frame_type=='last':
            spk_emb_frame_dim = 512

        if self.pooling:
            if self.pooling=='ASTP':
                self.poolayer = ASTP(in_dim=spk_emb_frame_dim, global_context_att=True)
            else: 
                self.poolayer = LDEPooling(input_dim=spk_emb_frame_dim, c_num=self.pooling)
            self.pool_out_dim = self.poolayer.get_out_dim()
            self.bn_pooling = nn.BatchNorm1d(self.pool_out_dim)
            self.linear_pooling = nn.Linear(self.pool_out_dim, spk_emb_dim)

        self.separator = FuseSeparation(nband=self.nband, num_repeat=num_repeat, feature_dim=feature_dim, spk_emb_dim=spk_emb_dim, spk_fuse_type=spk_fuse_type, multi_fuse=multi_fuse, spk_model=spk_model, mode=mode, frame_type=frame_type, emb_feat=(emb_feat or utt_feat))

        # self.proj =  nn.Linear(hidden_size*2, input_size)

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(nn.GroupNorm(1, self.feature_dim, torch.finfo(torch.float32).eps),
                              nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                              nn.Tanh(),
                              nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                              nn.Tanh(),
                              nn.Conv1d(self.feature_dim * 4, self.band_width[i] * 4, 1)
                              )
            )

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input, embeddings, mode=False):
        # input shape: (B, C, T)

        wav_input = input
        spk_emb_input = embeddings
        batch_size, nsample = wav_input.shape
        nch = 1

        # frequency-domain separation
        spec = torch.stft(wav_input, n_fft=self.win, hop_length=self.stride,
                          window=torch.hann_window(self.win).to(wav_input.device).type(wav_input.type()),
                          return_complex=True)

        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T

        if self.joint_training and (self.emb_feat or self.emb_score or self.utt_feat):
            if not self.spk_feat:
                if self.feat_type=='consistent':
                    with torch.no_grad():
                        spk_emb_input = self.preEmphasis(spk_emb_input)
                        spk_emb_input = self.spk_encoder(spk_emb_input)+1e-8
                        spk_emb_input = spk_emb_input.log()
                        spk_emb_input = spk_emb_input - torch.mean(spk_emb_input, dim=-1, keepdim=True)
                        spk_emb_input = spk_emb_input.permute(0, 2, 1)
                
                if self.feat_type=='wespeaker':
                    with torch.no_grad():
                        if self.emb_score:
                            if True:    # center in torch.stft is True, pad the mixture
                                signal_dim = wav_input.dim()
                                extended_shape = [1] * (3 - signal_dim) + list(wav_input.size())
                                pad = int(self.win // 2)
                                wav_input_pad = F.pad(wav_input.view(extended_shape), [pad, pad], mode='reflect')
                                wav_input_pad = wav_input_pad.view(wav_input_pad.shape[-signal_dim:])

                                signal_dim = spk_emb_input.dim()
                                extended_shape = [1] * (3 - signal_dim) + list(spk_emb_input.size())
                                pad = int(self.win // 2)
                                spk_emb_input = F.pad(spk_emb_input.view(extended_shape), [pad, pad], mode='reflect')
                                spk_emb_input = spk_emb_input.view(spk_emb_input.shape[-signal_dim:])

                            spk_emb_input = compute_fbank(spk_emb_input, frame_length=self.win*1e3/self.sr, frame_shift=self.stride*1e3/self.sr, dither=.0, sample_rate=self.sr)
                            mix_emb_input = compute_fbank(wav_input_pad, frame_length=self.win*1e3/self.sr, frame_shift=self.stride*1e3/self.sr, dither=.0, sample_rate=self.sr)
                            mix_emb_input = apply_cmvn(mix_emb_input)
                        else:
                            spk_emb_input = compute_fbank(spk_emb_input, dither=.0, sample_rate=self.sr)
                        spk_emb_input = apply_cmvn(spk_emb_input)

            spk_emb_input = self.spk_model(spk_emb_input)
            if self.emb_score:
                mix_emb_input = self.spk_model(mix_emb_input)
                if isinstance(mix_emb_input,tuple):
                    mix_emb_frame = mix_emb_input[0]

            if mode=='SV':
                return spk_emb_input

            if isinstance(spk_emb_input,tuple):
                if self.spk_fuse_type.startswith('cross') or self.emb_score:
                    spk_emb_frame = spk_emb_input[0]
                if self.pooling:
                    spk_emb_input = self.poolayer(spk_emb_input[0])
                    spk_emb_input = self.bn_pooling(spk_emb_input)
                    spk_emb_input = self.linear_pooling(spk_emb_input)
                else:
                    spk_emb_input = spk_emb_input[-1]
            if self.multi_task:
                predict_speaker_lable = self.pred_linear(spk_emb_input)

        if (self.time_feat or self.tf_feat) and not self.spk_feat:
            spec_enroll = torch.stft(embeddings, n_fft=self.win, hop_length=self.stride,
                              window=torch.hann_window(self.win).to(embeddings.device).type(embeddings.type()),
                              return_complex=True)
            magnitude_spec = torch.abs(spec)
            magnitude_enroll = torch.abs(spec_enroll)

            if self.emb_score:
                mix_emb_frame = F.normalize(mix_emb_frame, p=2, dim=1)
                spk_emb_frame_norm = F.normalize(spk_emb_frame, p=2, dim=1)
                # magnitude_enroll_ = F.normalize(magnitude_enroll, p=2, dim=1)
                magnitude_enroll_ = magnitude_enroll
                att_scores = torch.matmul(mix_emb_frame.transpose(1,2), spk_emb_frame_norm)
            else:
                # magnitude_spec_ = torch.pow(magnitude_spec, 0.5)     # dynamic range compression
                magnitude_spec_ = magnitude_spec
                magnitude_enroll_ = F.normalize(magnitude_enroll, p=2, dim=1)
                # magnitude_enroll_pow = torch.pow(magnitude_enroll_, 0.5)
                magnitude_enroll_pow = magnitude_enroll_
                att_scores = torch.matmul(magnitude_spec_.transpose(1,2), magnitude_enroll_pow)
            
            att_weights = F.softmax(att_scores, dim=-1)

            if self.time_feat:
                time_mask = torch.sum(att_scores, dim=2)
                time_mask = time_mask / torch.sum(time_mask, dim=1, keepdim=True)
                time_mask = time_mask.unsqueeze(1).expand(-1, magnitude_spec.shape[1], -1)
                # print('time_mask:', time_mask.shape)
                # print('magnitude_spec:', magnitude_spec.shape)
                # os._exit()
                time_map = time_mask * magnitude_spec
                spec_RI = torch.cat((spec_RI, time_map.unsqueeze(1)), dim=1)

            if self.tf_feat:
                # print('att_weights:', att_weights.shape)
                # print('magnitude_enroll:', magnitude_enroll.shape)
                # os._exit()
                if self.tf_feat=='concat':
                    time1_length = magnitude_spec.size(-1)
                    time2_length = magnitude_enroll.size(-1)
                    if time2_length >= time1_length:
                        tf_map = magnitude_enroll[:, :, :time1_length]
                    else:
                        repeat_times = (time1_length + time2_length - 1) // time2_length
                        tf_map = magnitude_enroll.repeat(1, 1, repeat_times)[:, :, :time1_length]
                else:
                    tf_map = torch.matmul(att_weights, magnitude_enroll_.transpose(1,2)).transpose(1,2)
                    if self.map_norm:
                        tf_map = tf_map / tf_map.norm(dim=1, keepdim=True)
                        tf_map = torch.sum(magnitude_spec * tf_map, dim=1, keepdim=True) * tf_map
                    else:
                        tf_map = tf_map / tf_map.norm(dim=1, keepdim=True) * magnitude_spec.norm(dim=1, keepdim=True)

                if mode=='SV_tfmap':
                    return tf_map, magnitude_spec
                
                spec_RI = torch.cat((spec_RI, tf_map.unsqueeze(1)), dim=1)

        # concat real and imag, split to subbands
        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(spec_RI[:, :, band_idx:band_idx + self.band_width[i]].contiguous())
            subband_mix_spec.append(spec[:, band_idx:band_idx + self.band_width[i]])  # B*nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(self.BN[i](subband_spec[i].view(batch_size * nch, self.band_width[i] * self.channel_map, -1)))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T
        # print(subband_feature.size(), spk_emb_input.size())

        # print('spk_emb_input.shape: ', spk_emb_input.shape) torch.Size([24, 398, 80])

        # print('spk_emb_frame.shape: ',spk_emb_frame.shape) torch.Size([24, 512, 436])
        # os._exit()
        if self.emb_feat or self.utt_feat:
            spk_embedding = self.spk_transform(spk_emb_input)
            if self.spk_norm:
                spk_embedding = F.normalize(spk_embedding, p=2, dim=-1)
                spk_embedding = spk_embedding * self.scale
                # norm_factor = torch.norm(spk_embedding,p=2,dim=-1,keepdim=True)
                # spk_embedding = spk_embedding / (norm_factor + 1e-8)  #* math.sqrt(spk_embedding.shape[-1])

            if not self.spk_fuse_type.startswith('cross'): 
                spk_embedding = spk_embedding.unsqueeze(1).unsqueeze(3)
                spk_emb_frame = None
        else:
            spk_embedding = None
            spk_emb_frame = None
            
        sep_output = self.separator(subband_feature, spk_embedding, spk_emb_frame=spk_emb_frame, nch=nch)

        sep_subband_spec = []
        for i in range(len(self.band_width)):
            this_output = self.mask[i](sep_output[:, i]).view(batch_size * nch, 2, 2, self.band_width[i], -1)
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:, 0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:, 1]  # B*nch, K, BW, T
            est_spec_real = subband_mix_spec[i].real * this_mask_real - subband_mix_spec[
                i].imag * this_mask_imag  # B*nch, BW, T
            est_spec_imag = subband_mix_spec[i].real * this_mask_imag + subband_mix_spec[
                i].imag * this_mask_real  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T
        output = torch.istft(est_spec.view(batch_size * nch, self.enc_dim, -1),
                             n_fft=self.win, hop_length=self.stride,
                             window=torch.hann_window(self.win).to(wav_input.device).type(wav_input.type()),
                             length=nsample)

        output = output.view(batch_size, nch, -1)
        s = torch.squeeze(output, dim=1)
        if self.joint_training and self.multi_task:
            if not isinstance(s, list):
                s = [s,]
            s.append(predict_speaker_lable)
        return s


if __name__ == '__main__':
    from thop import profile, clever_format

    model = BSRNN(spk_emb_dim=256, sr=16000, win=512, stride=128,
                  feature_dim=128, num_repeat=6, spk_fuse_type='additive')

    s = 0
    for param in model.parameters():
        s += np.product(param.size())
    print('# of parameters: ' + str(s / 1024.0 / 1024.0))
    x = torch.randn(4, 32000)
    spk_embeddings = torch.randn(4, 256)
    output = model(x, spk_embeddings)
    print(output.shape)

    macs, params = profile(model, inputs=(x, spk_embeddings))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)

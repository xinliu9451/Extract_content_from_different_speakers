
import os
import numpy as np
from scipy import optimize
import json

import torch
from pyannote.audio import Inference

from .utils.fileio import load_audio
from .utils.builder import build
from .utils.config import Config
from .utils.utils import circle_pad


class Diarization3Dspeaker:

    def __init__(self,
                vad_model =None,
                embedding_model =None,
                embedding_feature_extractor =None,
                segmentation_model =None,
                device=None):
        
        self.device = self.normalize_device(device)
        self.vad_model = vad_model
        self.embedding_model = embedding_model
        self.feature_extractor = embedding_feature_extractor
        self.segmentation_model = segmentation_model
        self.cluster = self.get_cluster_backend()

        self.batchsize = 64
        self.fs = self.feature_extractor.sample_rate
        self.output_field_labels = None

    def __call__(self, wav, wav_fs=None, include_overlap = True,speaker_num=None):
        wav_data = load_audio(wav, wav_fs, self.fs)
        # stage 1-1: do vad
        vad_time = self.do_vad(wav_data)
        if include_overlap:
            # stage 1-2: do segmentation
            segmentations, count = self.do_segmentation(wav_data)
            valid_field = self.get_valid_field(count)
            vad_time = self.merge_vad(vad_time, valid_field)
        # stage 2: prepare subseg
        # 每一个vad段 按照1.5s窗长和0.75s窗移切分
        chunks = [c for (st, ed) in vad_time for c in self.chunk(st, ed,dur=1.5,step=0.75)]
        # stage 3: extract embeddings
        embeddings = self.do_emb_extraction(chunks, wav_data)
        # stage 4: clustering
        speaker_num, output_field_labels = self.do_clustering(chunks, embeddings, speaker_num)
        if include_overlap:
            # stage 5: include overlap results
            binary = self.post_process(output_field_labels, speaker_num, segmentations, count)
            timestamps = [count.sliding_window[i].middle for i in range(binary.shape[0])]
            output_field_labels = self.binary_to_segs(binary, timestamps,threshold=0.5)
        self.output_field_labels = output_field_labels
        return output_field_labels
    
    
    def do_vad(self, wav):
        # wav: [1, T]
        vad_results = self.vad_model(wav[0])[0]
        vad_time = [[vad_t[0]/1000, vad_t[1]/1000] for vad_t in vad_results['value']]
        return vad_time
    
    @staticmethod
    def merge_vad(vad1: list, vad2: list):
        intervals = vad1 + vad2
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged
    def do_segmentation(self, wav):
        segmentations = self.segmentation_model({'waveform':wav, 'sample_rate': self.fs})
        frame_windows = self.segmentation_model.model.receptive_field

        count = Inference.aggregate(
            np.sum(segmentations, axis=-1, keepdims=True),
            frame_windows,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count.data = np.rint(count.data).astype(np.uint8)
        return segmentations, count

    def chunk(self, st, ed, dur=1.5, step=0.75):
        chunks = []
        subseg_st = st
        while subseg_st + dur < ed + step:
            subseg_ed = min(subseg_st + dur, ed)
            chunks.append([subseg_st, subseg_ed])
            subseg_st += step
        return chunks

    def do_emb_extraction(self, chunks, wav):
        # chunks: [[st1, ed1]...]
        # wav: [1, T]
        wavs = [wav[0, int(st*self.fs):int(ed*self.fs)] for st, ed in chunks]
        max_len = max([x.shape[0] for x in wavs])
        wavs = [circle_pad(x, max_len) for x in wavs]
        wavs = torch.stack(wavs).unsqueeze(1)

        embeddings = []
        batch_st = 0
        with torch.no_grad():
            while batch_st < len(chunks):
                wavs_batch = wavs[batch_st: batch_st+self.batchsize].to(self.device)
                feats_batch = torch.vmap(self.feature_extractor)(wavs_batch)
                embeddings_batch = self.embedding_model(feats_batch).cpu()
                embeddings.append(embeddings_batch)
                batch_st += self.batchsize
        embeddings = torch.cat(embeddings, dim=0).numpy()
        return embeddings

    @staticmethod
    def compressed_seg(seg_list):
        new_seg_list = []
        for i, seg in enumerate(seg_list):
            seg_st, seg_ed, cluster_id = seg
            if i == 0:
                new_seg_list.append([seg_st, seg_ed, cluster_id])
            elif cluster_id == new_seg_list[-1][2]:
                if seg_st > new_seg_list[-1][1]:
                    new_seg_list.append([seg_st, seg_ed, cluster_id])
                else:
                    new_seg_list[-1][1] = seg_ed
            else:
                if seg_st < new_seg_list[-1][1]:
                    p = (new_seg_list[-1][1]+seg_st) / 2
                    new_seg_list[-1][1] = p
                    seg_st = p
                new_seg_list.append([seg_st, seg_ed, cluster_id])
        return new_seg_list

    @staticmethod
    def get_cluster_backend():
        conf = {
            'cluster':{
                'obj': 'speakerlab.process.cluster.CommonClustering',
                'args':{
                    'cluster_type': 'spectral',
                    'mer_cos': 0.8,
                    'min_num_spks': 1,
                    'max_num_spks': 10,# 15
                    'min_cluster_size': 2,# 4
                    'oracle_num': None,
                    'pval': 0.012,
                }
            }
        }
        config = Config(conf)
        return build('cluster', config)
    
    def do_clustering(self, chunks, embeddings, speaker_num=None):
        cluster_labels = self.cluster(embeddings, speaker_num = speaker_num)

        speaker_num = cluster_labels.max()+1
        output_field_labels = [[i[0], i[1], int(j)] for i, j in zip(chunks, cluster_labels)]
        output_field_labels = self.compressed_seg(output_field_labels)
        return speaker_num, output_field_labels

    def post_process(self, output_field_labels, speaker_num, segmentations, count):
        num_frames = len(count)
        cluster_frames = np.zeros((num_frames, speaker_num))
        frame_windows = count.sliding_window
        for i in output_field_labels:
            cluster_frames[frame_windows.closest_frame(i[0]+frame_windows.duration/2) :\
                            frame_windows.closest_frame(i[1]+frame_windows.duration/2), i[2]] = 1.0

        activations = np.zeros((num_frames, speaker_num))
        num_chunks, num_frames_per_chunk, num_classes = segmentations.data.shape
        for i, (c, data) in enumerate(segmentations):
            # data: [num_frames_per_chunk, num_classes]
            # chunk_cluster_frames: [num_frames_per_chunk, speaker_num]
            start_frame = frame_windows.closest_frame(c.start+frame_windows.duration/2)
            end_frame = start_frame + num_frames_per_chunk
            chunk_cluster_frames = cluster_frames[start_frame:end_frame]
            align_chunk_cluster_frames = np.zeros((num_frames_per_chunk, speaker_num))

            # assign label to each dimension of "data" according to number of 
            # overlap frames between "data" and "chunk_cluster_frames"
            cost_matrix = []
            for j in range(num_classes):
                if sum(data[:, j])>0:
                    num_of_overlap_frames = [(data[:, j].astype('int') & d.astype('int')).sum() \
                        for d in chunk_cluster_frames.T]
                else:
                    num_of_overlap_frames = [-1]*speaker_num
                cost_matrix.append(num_of_overlap_frames)
            cost_matrix = np.array(cost_matrix) # (num_classes, speaker_num)
            row_index, col_index = optimize.linear_sum_assignment(-cost_matrix)
            for j in range(len(row_index)):
                r = row_index[j]
                c = col_index[j]
                if cost_matrix[r, c] > 0:
                    align_chunk_cluster_frames[:, c] = np.maximum(
                            data[:, r], align_chunk_cluster_frames[:, c]
                            )
            activations[start_frame:end_frame] += align_chunk_cluster_frames

        # correct activations according to count_data
        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations)
        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
            cur_max_spk_num = min(speaker_num, c.item())
            for i in range(cur_max_spk_num):
                if activations[t, speakers[i]] > 0:
                    binary[t, speakers[i]] = 1.0

        supplement_field = (binary.sum(-1)==0) & (cluster_frames.sum(-1)!=0)
        binary[supplement_field] = cluster_frames[supplement_field]
        return binary

    def binary_to_segs(self, binary, timestamps, threshold=0.5):
        output_field_labels = []
        # binary: [num_frames, num_classes]
        # timestamps: [T_1, ..., T_num_frames]        
        for k, k_scores in enumerate(binary.T):
            start = timestamps[0]
            is_active = k_scores[0] > threshold

            for t, y in zip(timestamps[1:], k_scores[1:]):
                if is_active:
                    if y < threshold:
                        output_field_labels.append([round(start, 3), round(t, 3), k])
                        start = t
                        is_active = False
                else:
                    if y > threshold:
                        start = t
                        is_active = True

            if is_active:
                output_field_labels.append([round(start, 3), round(t, 3), k])
        return sorted(output_field_labels, key=lambda x : x[0])

    def diarization_output_json(self, wav_id=None, output_field_labels=None):
        if output_field_labels is None and self.output_field_labels is None:
            raise ValueError('No results can be saved.')
        if output_field_labels is None:
            output_field_labels = self.output_field_labels

        out_json = {}
        for seg in output_field_labels:
            seg_st, seg_ed, cluster_id = seg
            item = { 'start': seg_st, 'stop': seg_ed, 'speaker': cluster_id}
            segid = wav_id+'_'+str(round(seg_st, 3)) + '_'+str(round(seg_ed, 3))
            out_json[segid] = item

        return out_json

        


    def normalize_device(self, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        else:
            assert isinstance(device, torch.device)
        return device
    
    @staticmethod
    def get_valid_field(count):
        valid_field = []
        start = None
        for i, (c, data) in enumerate(count):
            if data.item()==0 or i==len(count)-1:
                if start is not None:
                    end = c.middle
                    valid_field.append([start, end])
                    start = None
            else:
                if start is None:
                    start = c.middle
        return valid_field

import numpy as np
import os
import torch
import json
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import ast


class IcreLoader(Dataset):
    def __init__(self, args, mode='train', modality='audio-visual', incremental_task=0, incremental_step=0):
        self.mode = mode
        self.args = args
        self.modality = modality

        self.incremental_task = incremental_task
        self.incremental_step = incremental_step   # ke neng hui you wen ti

        self.ques_word_to_ix = {}
        self.label_to_ix = json.load(open('../../../data/split_avqa/json/label_dict_6_20.json', 'r'))
        self.task_partition = json.load(open('../../../data/split_avqa/json/task_partition_6_20.json', 'r'))
        self.all_task = ['Come_From', 'Happening', 'Where', 'Which']
        self.all_ans_len = {}
        self.max_len = 16

        self.audio_train_dir = args.audio_train_dir
        self.audio_test_dir = args.audio_test_dir
        self.visual_train_dir = args.visual_train_dir
        self.visual_test_dir = args.visual_test_dir

        if not os.path.exists(f'./encoder'):
            os.mkdir(f'./encoder')
        self.all_ans_type = []
        self.all_que_type = []
        self.all_que_num = 0
        self.all_ans_num = 0
        self.all_current_data_vids = []
        self.num_current_step_que = []
        self.num_current_step_ans = []
        self.last_step_out_ans_num = []
        self.last_step_out_que_num = []
        self.exemplar_class_vids = None
    def num_current_step_qa(self):

        if self.mode == 'train':

            current_ques_vocab = ['<pad>', '<unk>']
            current_ans_vocab = ['<unk>']
            i = 0
            for sample in self.all_current_data_vids:
                i += 1

                question = sample['question_text'].rstrip().split(' ')
                question[-1] = question[-1][:-1]

                p = 0
                for pos in range(len(question)):
                    if '<' in question[pos]:
                        question[pos] = ast.literal_eval(sample['templ_values'])[p]
                        p += 1

                for wd in question:
                    if wd not in current_ques_vocab:
                        current_ques_vocab.append(wd)
                if sample['answer'] not in current_ans_vocab:
                    current_ans_vocab.append(sample['answer'])
            temp_que_vocab = []
            for i, item in enumerate(current_ques_vocab):
                if item not in self.all_que_type:
                    temp_que_vocab.append(item)
            self.all_que_type += temp_que_vocab
            self.all_que_num = len(self.all_que_type)
            self.ques_word_to_ix = {word: i for i, word in enumerate(self.all_que_type)}
            with open(f'./encoder/{self.incremental_task}_{self.incremental_step}_ques_word_to_ix.json', 'w') as f:
                json.dump(self.ques_word_to_ix, f, indent=4)
            temp_ans_vocab = []
            for i, item in enumerate(current_ans_vocab):
                if item not in self.all_ans_type:
                    temp_ans_vocab.append(item)
            self.all_ans_type += temp_ans_vocab
            self.all_ans_num = len(self.all_ans_type)
            self.ans_word_to_ix = {word: i for i, word in enumerate(self.all_ans_type)}
            with open(f'./encoder/{self.incremental_task}_{self.incremental_step}_ans_word_to_ix.json', 'w') as f:
                json.dump(self.ans_word_to_ix, f, indent=4)
            num_current_step_que = self.all_que_num
            num_current_step_ans = self.all_ans_num
        else:
            self.ques_word_to_ix = json.load(open(f'./encoder/{self.incremental_task}_{self.incremental_step}_ques_word_to_ix.json', 'r'))
            self.ans_word_to_ix = json.load(open(f'./encoder/{self.incremental_task}_{self.incremental_step}_ans_word_to_ix.json', 'r'))

            num_current_step_que = len(self.ques_word_to_ix)

            num_current_step_ans = len(self.ans_word_to_ix)

        return num_current_step_que, num_current_step_ans

    def current_step_data(self):
        if self.mode == 'train':
            self.all_current_data_vids = []
            data = json.load(open(
                f'../../../data/split_avqa/json/{self.mode}_' + f'{self.all_task[self.incremental_task]}.json',
                'r'))
            for sample in data:
                if sample['label'] in self.task_partition[self.incremental_step]:
                    sample['label'] = self.all_task[self.incremental_task] + '_' + sample['label']
                    self.all_current_data_vids.append(sample)
        else:
            data = json.load(open(
                f'../../../data/split_avqa/json/{self.mode}_' + f'{self.all_task[self.incremental_task]}.json', 'r'))
            for sample in data:
                if sample['label'] in self.task_partition[self.incremental_step]:
                    sample['label'] = self.all_task[self.incremental_task] + '_' + sample['label']
                    self.all_current_data_vids.append(sample)

        return self.all_current_data_vids

    def set_incremental_step(self, task, step):
        self.incremental_task = task
        self.incremental_step = step
        self.all_current_data_vids = self.current_step_data()
        self.num_current_step_que, self.num_current_step_ans = self.num_current_step_qa()

        self.last_step_out_ans_num.append(self.num_current_step_ans)
        self.last_step_out_que_num.append(self.num_current_step_que)

    def __getitem__(self, index):
        sample = self.all_current_data_vids[index]
        name = sample['video_name']
        que_id = sample['id']
        label = sample['label']
        label = self.label_to_ix.get(label, self.label_to_ix['<unk>'])
        label = torch.from_numpy(np.array(label)).long()
        name = name[:-4]
        if self.mode == 'train' or self.mode == 'gen_exemplar':
            audio = np.load(os.path.join(self.audio_train_dir, name + '.npy'))
            audio = torch.from_numpy(audio).float()

            visual = np.load(os.path.join(self.visual_train_dir, name + '.npy'))
            visual = torch.from_numpy(visual).float()
        else:
            audio = np.load(os.path.join(self.audio_test_dir, name + '.npy'))
            audio = torch.from_numpy(audio).float()

            visual = np.load(os.path.join(self.visual_test_dir, name + '.npy'))
            visual = torch.from_numpy(visual).float()

        question = sample['question_text'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')

        idxs = [self.ques_word_to_ix.get(w, self.ques_word_to_ix['<unk>']) for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)

        answer = sample['answer']
        anser = self.ans_word_to_ix.get(answer, self.ans_word_to_ix['<unk>'])
        anser = torch.from_numpy(np.array(anser)).long()

        return audio, visual, ques, anser, que_id, label
    def __len__(self):
        return len(self.all_current_data_vids)

class exemplarLoader(Dataset):
    def __init__(self, args, modality='audio-visual', incremental_task =0, incremental_step=0):
        self.args = args
        self.modality = modality
        self.incremental_task = incremental_task
        self.incremental_step = incremental_step

        self.ques_word_to_ix = {}
        self.label_to_ix = json.load(open('../../../data/split_avqa/json/label_dict_6_20.json', 'r'))
        self.task_partition = json.load(open('../../../data/split_avqa/json/task_partition_6_20.json', 'r'))
        self.all_task = ['Come_From', 'Happening', 'Where', 'Which']
        self.all_ans_len = {}
        self.max_len = 16

        self.audio_train_dir = args.audio_train_dir
        self.audio_test_dir = args.audio_test_dir
        self.visual_train_dir = args.visual_train_dir
        self.visual_test_dir = args.visual_test_dir

        self.cur_all_ans_num = 0
        self.cur_ans_num = 0
        self.exemplar_class_vids_set = []
        self.exemplar_vids_set = []

    def _set_incremental_step_(self, task, step, classes_per_step, cur_all_ans_num):
        self.last_all_ans_num = cur_all_ans_num
        self.last_ans_num = classes_per_step

        self.incremental_task = task
        self.incremental_step = step
        self.ques_word_to_ix = json.load(
            open(f'./encoder/{self.incremental_task}_{self.incremental_step}_ques_word_to_ix.json', 'r'))
        self.ans_word_to_ix = json.load(
            open(f'./encoder/{self.incremental_task}_{self.incremental_step}_ans_word_to_ix.json', 'r'))
        self._update_exemplars_()

    def _update_exemplars_(self):

        if self.incremental_task == 0 and self.incremental_step == 0:
            return
        new_memory_classes = range(self.last_ans_num * (self.incremental_step-1), self.last_ans_num * self.incremental_step )
        exemplar_num_per_class = self.args.memory_size // (self.last_all_ans_num)  #
        new_memory_class_exemplars = self._init_new_memory_class_exemplars_(new_memory_classes, exemplar_num_per_class)

        if self.incremental_task ==0 and self.incremental_step == 1:
            self.exemplar_class_vids_set += new_memory_class_exemplars
        else:
            for i in range(len(self.exemplar_class_vids_set)):
                self.exemplar_class_vids_set[i] = self.exemplar_class_vids_set[i][:exemplar_num_per_class]

            self.exemplar_class_vids_set += new_memory_class_exemplars
        self.exemplar_vids_set = self.exemplar_class_vids_set
        self.exemplar_vids_set = np.array(self.exemplar_class_vids_set).reshape(-1).tolist()
        self.exemplar_vids_set = [vid for vid in self.exemplar_vids_set if vid is not None]

    def _init_new_memory_class_exemplars_(self, new_memory_classes, exemplar_num_per_class):
        new_memory_class_exemplars = []
        last_step_data = []
        data = json.load(open(
            f'../../../data/split_avqa/json/train_' + f'{self.all_task[self.incremental_task]}.json',
            'r'))
        for sample in data:
            if sample['label'] in self.task_partition[self.incremental_step-1]:
                sample['label'] = self.all_task[self.incremental_task] + '_' + sample['label']
                last_step_data.append(sample)
        data_current = []
        for c in new_memory_classes:
            for i, item in enumerate(last_step_data):

                if self.label_to_ix.get(item['label'], self.label_to_ix['<unk>']) == c:
                    data_current.append(item)
            class_exemplar = random.sample(data_current, min(len(data_current), exemplar_num_per_class))
            if len(data_current) < exemplar_num_per_class:
                class_exemplar += [None for i in range(exemplar_num_per_class - len(data_current))]
            new_memory_class_exemplars.append(class_exemplar)
        return new_memory_class_exemplars

    def __getitem__(self, index):
        i = index % len(self.exemplar_vids_set)
        sample = self.exemplar_vids_set[i]
        name = sample['video_name']
        que_id = sample['id']
        label = sample['label']
        label = self.label_to_ix.get(label, self.label_to_ix['<unk>'])
        label = torch.from_numpy(np.array(label)).long()
        name = name[:-4]
        audio = np.load(os.path.join(self.audio_train_dir, name + '.npy'))
        audio = torch.from_numpy(audio).float()
        visual = np.load(os.path.join(self.visual_train_dir, name + '.npy'))
        visual = torch.from_numpy(visual).float()
        question = sample['question_text'].rstrip().split(' ')
        question[-1] = question[-1][:-1]
        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')

        idxs = [self.ques_word_to_ix.get(w, self.ques_word_to_ix['<unk>']) for w in question]  ##
        ques = torch.tensor(idxs, dtype=torch.long)
        answer = sample['answer']
        anser = self.ans_word_to_ix.get(answer, self.ans_word_to_ix['<unk>'])
        anser = torch.from_numpy(np.array(anser)).long()

        return audio, visual, ques, anser, que_id, label

    def __len__(self):
        return len(self.exemplar_vids_set)

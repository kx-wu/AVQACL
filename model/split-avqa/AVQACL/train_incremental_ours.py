import json
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import warnings
warnings.filterwarnings("ignore")
from dataloader_ours import IcreLoader, exemplarLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from audio_visual_model_incremental import IncreAudioVisualNet
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from datetime import datetime
import random
from itertools import cycle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def CE_loss(step_out_ans_num, logits, label):
    targets = F.one_hot(label, num_classes=step_out_ans_num)

    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

    return loss

def que_loss(new_features, old_features):

    cos_sim = F.cosine_similarity(new_features, old_features, dim=-1)
    que_loss = 1 - cos_sim.mean()

    return que_loss

def top_1_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()

def adjust_learning_rate(args, optimizer, epoch):
    miles_list = np.array(args.milestones) - 1
    if epoch in miles_list:
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.1
        print('Reduce lr from {} to {}'.format(current_lr, new_lr))
        for param_group in optimizer.param_groups: 
            param_group['lr'] = new_lr

def spatial_temporal_dis_loss(new_features, old_features, dim):

    new_features_diff = new_features.diff(dim=dim)
    old_features_diff = old_features.diff(dim=dim)
    spatial_temporal_dis_loss = F.mse_loss(new_features_diff, old_features_diff)

    return spatial_temporal_dis_loss



def train(args, task, step, train_data_set, val_data_set, exemplar_set):
    T = 2

    train_loader = DataLoader(train_data_set, batch_size=args.train_batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False)

    step_out_ans_num = train_data_set.num_current_step_ans
    vocab_size = train_data_set.num_current_step_que
    step_per_task = 6

    if task == 0 and step == 0:
        model = IncreAudioVisualNet(args, step_out_ans_num, vocab_size)
    elif task > 0 and step == 0:
        model = torch.load('./save/{}/{}/task_{}_step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, task-1, step_per_task-1, args.modality))
        model.incremental_classifier(step_out_ans_num)
        model.question_encoder.incremental_vocab(vocab_size)
        print('actual size of exemplar set: {}'.format(exemplar_set.__len__()))
        exemplar_loader = DataLoader(exemplar_set, batch_size=min(args.exemplar_batch_size, exemplar_set.__len__()), num_workers=args.num_workers,
                                     pin_memory=True, drop_last=True, shuffle=True)
        old_model = torch.load('./save/{}/{}/task_{}_step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, task-1, step_per_task-1, args.modality))   #加载旧模型用于知识蒸馏
        last_step_out_class_num = train_data_set.last_step_out_ans_num[-2]

    else:
        model = torch.load('./save/{}/{}/task_{}_step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, task, step-1, args.modality))
        model.incremental_classifier(step_out_ans_num)
        model.question_encoder.incremental_vocab(vocab_size)
        print('actual size of exemplar set: {}'.format(exemplar_set.__len__()))
        exemplar_loader = DataLoader(exemplar_set, batch_size=min(args.exemplar_batch_size, exemplar_set.__len__()), num_workers=args.num_workers,
                                     pin_memory=True, drop_last=True, shuffle=True)
        old_model = torch.load('./save/{}/{}/task_{}_step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, task, step-1, args.modality))   #加载旧模型用于知识蒸馏
        last_step_out_class_num = train_data_set.last_step_out_ans_num[-2]
    model = model.to(device)
    if task!= 0 or step != 0:
        old_model = old_model.to(device)
        old_model.question_encoder.word2vec.weight = model.question_encoder.word2vec.weight
        old_model.classifier.weight = model.classifier.weight
        old_model.classifier.bias = model.classifier.bias
        old_model.eval()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0
    for epoch in range(args.max_epoches):
        train_loss = 0.0
        num_steps = 0
        model.train()
        if task == 0 and step == 0:
            iterator = tqdm(train_loader)
        else:
            iterator = tzip(train_loader, cycle(exemplar_loader))

        for samples in iterator:
            if task == 0 and step == 0:
                audio, visual, ques, labels, *_ = samples
                labels = labels.to(device)
                if args.modality == 'visual':
                    visual = data
                    visual = visual.to(device)
                    out = model(visual=visual)
                elif args.modality == 'audio':
                    audio = data
                    audio = audio.to(device)
                    out = model(audio=audio)
                else:
                    visual = visual.to(device)
                    audio = audio.to(device)
                    ques = ques.to(device)
                    out, audio_features, visual_features, qst_feature = model(audio=audio, visual=visual, question=ques, out_feature_before_fusion=True, que_feature=True)

                loss = CE_loss(step_out_ans_num, out, labels)
            else:
                curr, prev = samples
                audio, visual, ques, labels, *_ = curr
                labels = labels % (step_out_ans_num - last_step_out_class_num)   # 当前问题数量
                labels = labels.to(device)

                exemplar_audio, exemplar_visual, exemplar_ques, exemplar_labels, *_ = prev
                exemplar_labels = exemplar_labels.to(device)

                data_batch_size = labels.shape[0]
                exemplar_data_batch_size = exemplar_labels.shape[0]

                if args.modality == 'visual':
                    visual = data
                    exemplar_visual = exemplar_data
                    total_visual = torch.cat((visual, exemplar_visual))
                    total_visual = total_visual.to(device)
                    out = model(visual=total_visual)
                    with torch.no_grad():
                        old_out = old_model(visual=total_visual).detach()
                elif args.modality == 'audio':
                    audio = data
                    exemplar_audio = exemplar_data
                    total_audio = torch.cat((audio, exemplar_audio))
                    total_audio = total_audio.to(device)
                    out = model(audio=total_audio)
                    with torch.no_grad():
                        old_out = old_model(audio=total_audio).detach()
                else:
                    total_visual = torch.cat((visual, exemplar_visual))
                    total_audio = torch.cat((audio, exemplar_audio))
                    total_ques = torch.cat((ques, exemplar_ques))
                    total_ques = total_ques.to(device)
                    total_visual = total_visual.to(device)
                    total_audio = total_audio.to(device)
                    out, audio_features, visual_features, qst_feature = model(audio=total_audio, visual=total_visual, question=total_ques, out_feature_before_fusion=True, que_feature=True)
                    with torch.no_grad():

                        old_out, old_audio_features, old_visual_features, old_qst_feature = old_model(audio=total_audio, visual=total_visual, question=total_ques, out_feature_before_fusion=True, que_feature=True)
                        old_out = old_out.detach()

                old_out = old_out[:, :last_step_out_class_num]

                curr_out = out[:data_batch_size, last_step_out_class_num:]
                loss_curr = CE_loss(step_out_ans_num - last_step_out_class_num, curr_out, labels)

                prev_out = out[data_batch_size:data_batch_size + exemplar_data_batch_size, :last_step_out_class_num]
                loss_prev = CE_loss(last_step_out_class_num, prev_out, exemplar_labels)

                loss_CE = (loss_curr * data_batch_size + loss_prev * exemplar_data_batch_size) / (
                            data_batch_size + exemplar_data_batch_size)

                loss_que = que_loss(qst_feature, old_qst_feature)
                loss_CE = loss_CE + args.que_weight * loss_que
                loss_KD = torch.zeros(task*step_per_task + step).to(device)

                start = 0
                for t in range(task*step_per_task + step):
                    end = train_data_set.last_step_out_ans_num[t]
                    soft_target = F.softmax(old_out[:, start:end] / T, dim=1)
                    output_log = F.log_softmax(out[:, start:end] / T, dim=1)
                    loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T ** 2)
                    start = end
                loss_KD = loss_KD.sum()
                temporal_audio_dis_loss = spatial_temporal_dis_loss(audio_features, old_audio_features, dim=0)
                temporal_visual_dis_loss = spatial_temporal_dis_loss(visual_features, old_visual_features, dim=0)
                spatial_visual_dis_loss = spatial_temporal_dis_loss(visual_features, old_visual_features, dim=-1)

                total_spatia_temporal_loss = args.spatial_temporal_weight * (temporal_audio_dis_loss + temporal_visual_dis_loss + spatial_visual_dis_loss)

                loss = loss_CE + loss_KD + total_spatia_temporal_loss

            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1
        train_loss /= num_steps
        train_loss_list.append(train_loss)
        print('Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss), flush=True)

#**********************************************************************************************************

        all_val_out = torch.Tensor([])
        all_val_labels = torch.Tensor([])
        model.eval()
        with torch.no_grad():
            for val_samples in tqdm(val_loader):
                # val_labels = val_labels.to(device)
                if args.modality == 'visual':

                    val_visual = val_visual.to(device)
                    if torch.cuda.device_count() > 1:
                        val_out_logits = model.module.forward(visual=val_visual)
                    else:
                        val_out_logits = model(visual=val_visual)
                elif args.modality == 'audio':

                    val_audio = val_audio.to(device)
                    if torch.cuda.device_count() > 1:
                        val_out_logits = model.module.forward(audio=val_audio)
                    else:
                        val_out_logits = model(audio=val_audio)
                else:
                    val_audio, val_visual, val_ques, val_labels, *_ = val_samples
                    val_visual = val_visual.to(device)
                    val_audio = val_audio.to(device)
                    val_ques = val_ques.to(device)
                    val_out= model(audio=val_audio, visual=val_visual, question=val_ques)

                val_out = F.softmax(val_out, dim=-1).detach().cpu()
                all_val_out = torch.cat((all_val_out, val_out), dim=0)
                all_val_labels = torch.cat((all_val_labels, val_labels), dim=0)

        val_res = top_1_acc(all_val_out, all_val_labels)
        val_acc_list.append(val_res)
        print('Epoch:{} val_res:{:.6f} '.format(epoch, val_res), flush=True)

        if val_res > best_val_res:
            best_val_res = val_res
            print('Saving best model at Epoch {}'.format(epoch), flush=True)
            if torch.cuda.device_count() > 1:
                torch.save(model.module,
                           './save/{}/{}/task_{}_step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, task, step, args.modality))
            else:
                torch.save(model, './save/{}/{}/task_{}_step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, task, step, args.modality))


        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig('./save/fig/{}/{}/train_loss_task_{}_step_{}.png'.format(args.dataset, args.modality, task, step))
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label='val_acc')
        plt.legend()
        plt.savefig('./save/fig/{}/{}/val_acc_task_{}_step_{}.png'.format(args.dataset, args.modality, task, step))
        plt.close()

        if args.lr_decay:
            adjust_learning_rate(args, opt, epoch)

def detailed_test(args, task, step, test_data_set, task_best_acc_list):
    print("=====================================")
    print("Start testing...")
    print("=====================================")

    model = torch.load('./save/{}/{}/task_{}_step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, task, step, args.modality))
    model.to(device)

    test_loader = DataLoader(test_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)

    all_test_out_logits = torch.Tensor([])
    all_test_ans = torch.Tensor([])
    all_test_labels = torch.Tensor([])
    model.eval()
    with torch.no_grad():
        for test_samples in tqdm(test_loader):
            # test_labels = test_labels.to(device)
            if args.modality == 'visual':
               pass
            elif args.modality == 'audio':
               pass
            else:
                test_audio, test_visual, test_ques, test_ans, test_id, test_labels = test_samples
                test_visual = test_visual.to(device)
                test_audio = test_audio.to(device)
                test_ques = test_ques.to(device)
                test_out_logits= model(audio=test_audio, visual=test_visual, question=test_ques)
            test_out_logits = F.softmax(test_out_logits, dim=-1).detach().cpu()
            all_test_out_logits = torch.cat((all_test_out_logits, test_out_logits), dim=0)
            all_test_ans = torch.cat((all_test_ans, test_ans), dim=0)
            all_test_labels = torch.cat((all_test_labels, test_labels), dim=0)
    test_top1 = top_1_acc(all_test_out_logits, all_test_ans)
    print("Incremental task_{}_step_{} Testing res: {:.6f}".format(task, step, test_top1))

    if args.upper_bound:
        return test_top1, None

    old_task_acc_list = []
    class_num_per_step = 20

    for i in range(task*total_incremental_steps+step+1):

        test_start = i*class_num_per_step
        test_end = (i+1)*class_num_per_step + test_start
        step_class_list = range(test_start, test_end)
        step_class_idxs = []
        for c in step_class_list:
            idxs = np.where(all_test_labels.numpy() == c)[0].tolist()
            step_class_idxs += idxs
        step_class_idxs = np.array(step_class_idxs)
        i_labels = torch.Tensor(all_test_ans.numpy()[step_class_idxs])
        i_logits = torch.Tensor(all_test_out_logits.numpy()[step_class_idxs])
        i_acc = top_1_acc(i_logits, i_labels)
        if i == task*total_incremental_steps+step:
            curren_step_acc = i_acc
        else:
            old_task_acc_list.append(i_acc)

    if task == 0 and step == 0:
        forgetting = None
    else:
        forgetting = np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list))
        print('forgetting: {:.6f}'.format(forgetting))
        for i in range(len(task_best_acc_list)):
            task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])
    task_best_acc_list.append(curren_step_acc)

    return test_top1, forgetting

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='split_avqa', choices=['split_music-avqa', 'split_avqa'])
    parser.add_argument('--modality', type=str, default='audio-visual', choices=['visual', 'audio', 'audio-visual'])
    parser.add_argument('--audio_train_dir', type=str,
                        default='../../../features/split_avqa/train/vggish', help='audio feats dir')
    parser.add_argument('--visual_train_dir', type=str,
                        default='../../../features/split_avqa/train//resnet18', help='visual feats dir')
    parser.add_argument('--audio_test_dir', type=str,
                        default='../../../features/split_avqa/test/vggish', help='audio feats dir')
    parser.add_argument('--visual_test_dir', type=str,
                        default='../../../features/split_avqa/test/resnet18', help='visual feats dir')
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--infer_batch_size', type=int, default=16)
    parser.add_argument('--exemplar_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epoches', type=int, default=1)
    # parser.add_argument('--num_classes', type=int, default=28)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument("--milestones", type=int, default=[80], nargs='+', help="")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--memory_size', type=int, default=5000)
    parser.add_argument('--upper_bound', type=bool, default=False)
    parser.add_argument('--spatial_temporal_weight', type=float, default=0.5)
    parser.add_argument('--que_weight', type=float, default=0.7)

    args = parser.parse_args()
    print(args)

    total_incremental_tasks = 4
    total_incremental_steps = 6
    classes_per_step = 20
    all_class_num = 120
    setup_seed(args.seed)

    print('Training start time: {}'.format(datetime.now()))

    train_set = IcreLoader(args=args, mode='train', modality=args.modality)
    val_set = IcreLoader(args=args, mode='val', modality=args.modality)
    test_set = IcreLoader(args=args, mode='test', modality=args.modality)
    exemplar_set = exemplarLoader(args=args, modality=args.modality)

    task_best_acc_list = []

    step_forgetting_list = []
    task_top1_list = []

    ckpts_root = './save/{}/{}/'.format(args.dataset, args.modality)
    figs_root = './save/fig/{}/{}/'.format(args.dataset, args.modality)

    if not os.path.exists(ckpts_root):
        os.makedirs(ckpts_root)
    if not os.path.exists(figs_root):
        os.makedirs(figs_root)
    exemplar_class_vids = None
    performance_data_list = {}

    for task in range(total_incremental_tasks):
        for step in range(total_incremental_steps):
            train_set.set_incremental_step(task, step)
            val_set.set_incremental_step(task, step)
            test_set.set_incremental_step(task, step)

            exemplar_set._set_incremental_step_(task, step, classes_per_step, task*all_class_num + classes_per_step * step)

            print('Incremental step: {}_{}'.format(task, step))

            train(args, task, step, train_set, val_set, exemplar_set)
            test_top1, step_forgetting = detailed_test(args, task, step, test_set, task_best_acc_list)
            task_top1_list.append(test_top1)
            performance_data_list['step:' + f'{task}_' + f'{step}'] = test_top1, step_forgetting
            if step_forgetting is not None:
                step_forgetting_list.append(step_forgetting)
        Mean_accuracy = np.mean(task_top1_list)
        performance_data_list[f'task_{task}_Mean_accuracy'] = Mean_accuracy
        print('Average Accuracy: {:.6f}'.format(Mean_accuracy))
        if not args.upper_bound:
            Mean_forgetting = np.mean(step_forgetting_list)
            performance_data_list[f'task_{task}_Mean_forgetting'] = Mean_forgetting
            with open(figs_root + 'performance_data.json', 'w') as f:
                json.dump(performance_data_list, f, indent=4)
            print('Average Forgetting: {:.6f}'.format(Mean_forgetting))
        else:
            with open(figs_root + 'performance_data.json', 'w') as f:
                json.dump(performance_data_list, f, indent=4)
    



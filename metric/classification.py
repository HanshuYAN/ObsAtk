import os
import sys
import pathlib
import numpy as np

import torch
import torch.nn.functional as F

import advertorch
# from advertorch.attacks.utils import multiple_mini_batch_attack
# from advertorch.utils import predict_from_logits

from tqdm import tqdm
####################

def topk_dataset_accuracy(predict, test_loader, num_batch=None, device='cuda', topk=1):

    clncorrect = 0
    idx_batch = 0
    num_examples = 0
    
    lst_label = []
    lst_pred = []
    
    for clndata, target in tqdm(test_loader):
        clndata, target = clndata.to(device), target.to(device)
        # with torch.no_grad():
        output = predict(clndata)
        pred = predict_from_logits_topk(output, topk=topk)
        
        lst_label.append(target)
        lst_pred.append(pred)
        
        if topk == 1:
            clncorrect += pred.eq(target.view_as(pred)).sum().item()
        
        num_examples += clndata.shape[0]
        idx_batch += 1
        if idx_batch == num_batch:
            break
    
    label = torch.cat(lst_label).view(-1, 1)
    pred = torch.cat(lst_pred).view(-1, topk)
    num = label.size(0)
    accuracy = (label == pred).sum().item() / num

    message = '***** Test set acc: {}/{} ({:.2f}%)'.format(
                clncorrect, 
                num_examples,
                100. * accuracy)
    return accuracy, message



def topk_defense_success_rate(predict_for_test, predict_for_atk, loader, attack_class, attack_kwargs,
                    device=torch.device("cuda:0"), num_batch=None, topk=1, 
                    class_wise_results=False, verbose=False):
    
    adversary = attack_class(predict_for_atk, **attack_kwargs)
    accuracy, defense_success_rate, dist, num, label_pred_advpred = attack_mini_batches(predict_for_test, adversary, loader, device=device, norm=None, num_batch=num_batch, topk=topk)
    
    # message returned
    if class_wise_results == False:
        message = '***** Test set acc: {:.2f}%, adv: {:.2f}%.'.format(
                accuracy * 100., defense_success_rate * 100.)
        rval = _generate_basic_benchmark_str(attack_class, attack_kwargs, num, accuracy, defense_success_rate)
        return accuracy, defense_success_rate, message, rval
    else:
        label = label_pred_advpred[0]
        pred = label_pred_advpred[1]
        advpred = label_pred_advpred[2]
        nat_acc = []
        adv_acc = []
        for i in range(label.max()+1):
            idx = (label == i).nonzero()
            label_i = label[idx[:,0]]
            pred_i = pred[idx[:,0]]
            advpred_i = advpred[idx[:,0]]
            num_i = len(label_i)
            nat_acc.append( (pred_i == label_i).sum().item() / num_i )
            adv_acc.append( (advpred_i == label_i).sum().item() / num_i )
        nat_std = np.std(nat_acc)
        adv_std = np.std(adv_acc)
        
        message = '***** Test set acc: {:.2f}%, std: {:.2f}.\t'.format(accuracy * 100., nat_std*100.)
        if verbose:
            message += '\n'
            for i in range(label.max()+1):
                message += ' {:2d} - {:.2f}%,'.format(i, nat_acc[i]*100.)
            message += ';\n'
        message += '***** adv: {:.2f}%, std: {:.2f}.'.format(defense_success_rate * 100., adv_std*100.)
        if verbose:
            message += '\n'
            for i in range(label.max()+1):
                message += ' {:2d} - {:.2f}%,'.format(i, adv_acc[i]*100.)
            
        rval = _generate_basic_benchmark_str(attack_class, attack_kwargs, num, accuracy, defense_success_rate)
        
        return accuracy, defense_success_rate, nat_acc, adv_acc ,message, rval



def predict_from_logits_topk(logits, dim=1, topk=1):
    return logits.topk(topk, dim)[1]

def attack_mini_batches(myPredict, 
                adversary, loader, device="cuda",
                norm=None, num_batch=None, topk=1):
    lst_label = []
    lst_pred = []
    lst_advpred = []
    lst_dist = []

    _norm_convert_dict = {"Linf": "inf", "L2": 2, "L1": 1}
    if norm in _norm_convert_dict:
        norm = _norm_convert_dict[norm]

    if norm == "inf":
        def dist_func(x, y):
            return (x - y).view(x.size(0), -1).max(dim=1)[0]
    elif norm == 1 or norm == 2:
        from advertorch.utils import _get_norm_batch
        def dist_func(x, y):
            return _get_norm_batch(x - y, norm)
    else:
        assert norm is None

    idx_batch = 0
    from tqdm import tqdm
    for data, label in tqdm(loader):
        data, label = data.to(device), label.to(device)
        adv = adversary.perturb(data, label)
        
        adv_logits = myPredict(adv); advpred = predict_from_logits_topk(adv_logits, topk=topk)
        nat_logits = myPredict(data); pred = predict_from_logits_topk(nat_logits, topk=topk)

        lst_label.append(label)
        lst_pred.append(pred)
        lst_advpred.append(advpred)
        
        if norm is not None:
            lst_dist.append(dist_func(data, adv))
            
        idx_batch += 1
        if idx_batch == num_batch:
            break

    # return torch.cat(lst_label), torch.cat(lst_pred), torch.cat(lst_advpred), \
    #     torch.cat(lst_dist) if norm is not None else None
    
    label = torch.cat(lst_label).view(-1, 1)
    pred = torch.cat(lst_pred).view(-1, topk)
    advpred = torch.cat(lst_advpred).view(-1, topk)
    dist = torch.cat(lst_dist) if norm is not None else None
    
    num = label.size(0)
    accuracy = (label == pred).sum().item() / num
    defense_success_rate = (label == advpred).sum().item() / num
    dist = None if dist is None else dist[(label != advpred) & (label == pred)]
    
    return accuracy, defense_success_rate, dist, num, [label, pred, advpred]




def _generate_basic_benchmark_str(attack_class, attack_kwargs, num, accuracy,
        defense_success_rate):
    rval = ""
    rval += "# attack type: {}\n".format(attack_class.__name__)
    prefix = "# attack kwargs: "
        
    rval += prefix
    for key in attack_kwargs:
        rval += "{}={}, ".format(key, attack_kwargs[key])
    rval += '\n#\n'
    
    rval += "# accuracy: {:.2f}%\n".format(accuracy * 100.)
    rval += "# defending rate: {:.2f}%\n".format(defense_success_rate * 100.)
    return rval



class CW_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, logits,  target):        
        return self._cw_loss(logits, target)
    
    def _cw_loss(self, output, target, confidence=50, num_classes=10):
        # Compute the probability of the label class versus the maximum other
        # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
        target = target.data
        target_onehot = torch.zeros(target.size() + (num_classes,))
        target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = Variable(target_onehot, requires_grad=False)
        real = (target_var * output).sum(1)
        other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
        loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
        loss = torch.sum(loss)
        return loss
    






















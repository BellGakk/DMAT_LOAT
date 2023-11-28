import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.metrics import accuracy


def mart_loss(model, x_natural, y, epoch, reg_ep, more_reg, optimizer, step_size=0.007, epsilon=0.031, perturb_steps=10, beta=6.0, 
              attack='linf-pgd'):
    """
    MART training (Wang et al, 2020).
    """

    fluc_logit_correct = torch.tensor(0, dtype=torch.float32)
    fluc_logit_correct = fluc_logit_correct.cuda()
    fluc_logit_correct_prob = torch.tensor(0, dtype=torch.float32)
    fluc_logit_correct_prob = fluc_logit_correct.cuda()
        
    fluc_logit_wrong = torch.tensor(0, dtype=torch.float32)
    fluc_logit_wrong = fluc_logit_wrong.cuda()
    fluc_logit_wrong_prob = torch.tensor(0, dtype=torch.float32)
    fluc_logit_wrong_prob = fluc_logit_wrong_prob.cuda()
    correct = 0
    wrong= 0

    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise ValueError(f'Attack={attack} not supported for MART training!')
    model.train()
    
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    logits_adv = model(x_adv)
    pred_natural = torch.max(logits, dim=1)[1]

    for idx_l in range(y.size(0)):
        logits_softmax = torch.softmax(logits[idx_l], dim=0)
        logits_softmax_max = torch.tensor(0, dtype=torch.float32)
        mean_softmax = ((1 - logits_softmax[y[idx_l].cpu().numpy()])/(10-1))
    
        for j in range(len(logits_softmax)):
            if j != y[idx_l].cpu().numpy():
                logits_softmax_max = torch.max(logits_softmax[j], logits_softmax_max)

        if pred_natural[idx_l].cpu().numpy() == y[idx_l].cpu().numpy():
            correct += 1
            fluc_logit_correct_prob -= torch.log(1 - logits_softmax_max)
            for idx in range(len(logits_softmax)):
                if idx != y[idx_l].cpu().numpy():
                    fluc_logit_correct += ((logits_softmax[idx] - mean_softmax)**2)
                else:
                    pass
        else:
            wrong += 1
            fluc_logit_wrong_prob -= torch.log(1 - logits_softmax_max)
            for idx in range(len(logits_softmax)):
                if idx != y[idx_l].cpu().numpy():
                    fluc_logit_wrong += ((logits_softmax[idx] - mean_softmax)**2)
                else:
                    pass
                
    fluc_logit_correct /= correct
    fluc_logit_wrong /= wrong
    fluc_logit_correct_prob /= correct
    fluc_logit_wrong_prob /= wrong
     
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

    if more_reg == 'None':
        pass
    elif more_reg == 'kl':
        if epoch <= 1:
            reg = fluc_logit_correct_prob -  fluc_logit_wrong_prob
        elif epoch >= (reg_ep-10):
            reg = fluc_logit_wrong_prob - fluc_logit_correct_prob
    elif more_reg == 'mse':
        if epoch <= 1:
            reg = fluc_logit_correct -  fluc_logit_wrong
        elif epoch >= (reg_ep-10):
            reg = fluc_logit_wrong - fluc_logit_correct
    else:
        pass
    
    if more_reg != 'None':
        if epoch <= 1 or epoch >= (reg_ep-10):
            loss = loss_adv + float(beta) * loss_robust + reg
        else:
            loss = loss_adv + float(beta) * loss_robust
    else:
        loss = loss_adv + float(beta) * loss_robust
    
    

    batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits.detach()), 
                     'adversarial_acc': accuracy(y, logits_adv.detach())}
        
    return loss, batch_metrics

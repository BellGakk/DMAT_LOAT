import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from core.metrics import accuracy
from core.utils import SmoothCrossEntropyLoss
from core.utils import track_bn_stats



def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def _kl_div(logit1, logit2):
    return F.kl_div(F.log_softmax(logit1, dim=1), F.softmax(logit2, dim=1), reduction='batchmean')


def _jensen_shannon_div(logit1, logit2, T=1.):
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)

    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
    jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
    return jsd * 0.5


def trades_loss(model, x_natural, y, epoch, reg_ep, more_reg, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, 
                attack='linf-pgd', label_smoothing=0.1, use_cutmix=False, use_consistency=False, cons_lambda=0.0, cons_tem=0.0):
    """
    TRADES training (Zhang et al, 2019).
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
    
    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)
    
    x_adv = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix: # CutMix
        p_natural = y
    else:
        p_natural = F.softmax(model(x_natural), dim=1)
        p_natural = p_natural.detach()
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()
    track_bn_stats(model, True)
  
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()
    # calculate robust loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)
    pred_natural = torch.max(logits_natural, dim=1)[1]
    
    for idx_l in range(y.size(0)):
        logits_softmax = torch.softmax(logits_natural[idx_l], dim=0)
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
    
    if use_cutmix: # CutMix
        loss_natural = criterion_kl(F.log_softmax(logits_natural, dim=1), y)
    else:
        loss_natural = criterion_ce(logits_natural, y)

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
            
    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
    if more_reg != 'None':
        if epoch <= 1 or epoch >= (reg_ep-10):
            loss = loss_natural + reg + beta * loss_robust
        else:
            loss = loss_natural + beta * loss_robust
    else:
        loss = loss_natural + beta * loss_robust
        
    if use_consistency:
        logits_adv1, logits_adv2 = logits_adv.chunk(2)
        if more_reg != 'None':
            if epoch <= 1 or epoch >= (reg_ep-10):
                loss = loss + reg + cons_lambda * _jensen_shannon_div(logits_adv1, logits_adv2, cons_tem)
            else:
                loss = loss + cons_lambda * _jensen_shannon_div(logits_adv1, logits_adv2, cons_tem)
        else:
            loss = loss + cons_lambda * _jensen_shannon_div(logits_adv1, logits_adv2, cons_tem)

    if use_cutmix: # CutMix
        batch_metrics = {'loss': loss.item(), 'clean_acc': -1, 
                     'adversarial_acc': -1}
    else:
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                     'adversarial_acc': accuracy(y, logits_adv.detach())}
        
    return loss, batch_metrics



def trades_loss_LSE(model, x_natural, y, epoch, reg_ep, more_reg, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, 
                attack='linf-pgd', label_smoothing=0.1, clip_value=0, use_cutmix=False, num_classes=10):
    """
    SCORE training (Ours).
    """
    # criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    # criterion_kl = nn.KLDivLoss(reduction='sum')

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
    
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)
    
    x_adv = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix: # CutMix
        p_natural = y
    else:
        p_natural = F.softmax(model(x_natural), dim=1)
        p_natural = p_natural.detach()
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            output_adv = F.softmax(model(x_adv), dim=1)
            with torch.enable_grad():
                loss_lse = torch.sum((output_adv - p_natural) ** 2)
            grad = torch.autograd.grad(loss_lse, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                output_adv = F.softmax(model(adv), dim=1)
                loss = (-1) * torch.sum((output_adv - p_natural) ** 2)
            loss.backward()

            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()
    track_bn_stats(model, True)
  
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)
    pred_natural = torch.max(logits_natural, dim=1)[1]

    for idx_l in range(y.size(0)):
        logits_softmax = torch.softmax(logits_natural[idx_l], dim=0)
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
    
    if use_cutmix: # CutMix
        y_onehot = y
    else:
        y_onehot = (1 - num_classes * label_smoothing / (num_classes-1)) * F.one_hot(y, num_classes=num_classes) + label_smoothing / (num_classes-1)
    
    logits_natural = F.softmax(model(x_natural), dim=1)
    logits_adv = F.softmax(model(x_adv), dim=1)
    loss_natural = torch.sum((logits_natural - y_onehot) ** 2, dim=-1)
    loss_robust = torch.sum((logits_adv - logits_natural) ** 2, dim=-1)

    loss_robust = F.relu(loss_robust - clip_value) # clip loss value

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
            loss = loss_natural.mean() + beta * loss_robust.mean() + reg
        else:
            loss = loss_natural.mean() + beta * loss_robust.mean()
    else:
        loss = loss_natural.mean() + beta * loss_robust.mean()
        
    if use_cutmix: # CutMix
        batch_metrics = {'loss': loss.item(), 'clean_acc': -1, 
                     'adversarial_acc': -1}
    else:
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                     'adversarial_acc': accuracy(y, logits_adv.detach())}

    return loss, batch_metrics

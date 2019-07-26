import torch

from decimal import Decimal

from models.nets import MLP, CNN, ResNet, WideResNet
from util.schedules import PiecewiseSchedule

MAX_LR_SCHEDULE = 5


def get_input_size(args):
    return {
        'mnist': (28, 28, 1),
        'cifar10': (32, 32, 3),
        'cifar100': (32, 32, 3),
    }[args.dataset]


def create_model(args, input_size, num_classes, sample_batch=None):
    if args.nn == 'mlp':
        model = MLP(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers,
                    weight_norm=args.weight_norm, num_classes=num_classes, init=args.init,
                    sample_batch=sample_batch)
    elif args.nn == 'cnn':
        model = CNN(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers,
                    weight_norm=args.weight_norm, num_classes=num_classes, init=args.init,
                    sample_batch=sample_batch)
    elif args.nn == 'resnet':
        model = ResNet(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes,
                       weight_norm=args.weight_norm, batch_norm=args.batch_norm, init=args.init,
                       num_blocks=args.num_blocks, sample_batch=sample_batch,
                       init_extra_param=args.init_extra_param)
    elif args.nn == 'wrn':
        model = WideResNet(input_size=input_size, num_blocks=args.wrn_n, num_classes=num_classes,
                           weight_norm=args.weight_norm, batch_norm=args.batch_norm, init=args.init,
                           sample_batch=sample_batch, k=args.wrn_k, reduced_memory=args.wrn_reduced_memory,
                           init_extra_param=args.init_extra_param)
    else:
        raise NotImplementedError
    return model


def create_optimizer(model, args):
    optim_cls = {'sgd': torch.optim.SGD}[args.optimizer]
    extra_kwargs = {'momentum': args.momentum} if args.optimizer in ['sgd'] else {}
    optimizer = optim_cls(model.optimizer_parameters(), lr=args.lr, weight_decay=args.weight_decay, **extra_kwargs)
    return optimizer


def compute_correct_predictions(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == labels).sum().item()


def train_epoch(model, optimizer, dataloader, criterion, batches_per_update=1, log_fn=None,
                lr_list=None):
    running_loss, num_batches, num_samples, correct_preds = 0., 0, 0, 0
    model.train()
    optimizer.zero_grad()
    batches_per_iter = len(dataloader)

    if lr_list is not None:
        assert len(lr_list) == batches_per_iter

    for iter_idx, (x, y) in enumerate(dataloader):
        # Update LR for warmup
        if lr_list is not None:
            lr = lr_list[iter_idx]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * model.get_lr_multiplier(param_group)

        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()

        # Forward + backward pass
        y_ = model(x)
        loss = criterion(y_, y)
        assert not torch.isnan(loss), "loss is NaN after forward pass in iter %d" % iter_idx
        loss /= batches_per_update
        loss.backward()
        if (iter_idx + 1) % batches_per_update == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Accuracy
        correct_preds += compute_correct_predictions(y_, y)

        # Update monitors
        running_loss += loss.item()
        num_batches += 1 / batches_per_update
        num_samples += y.size(0)

        if (iter_idx + 1) % batches_per_update == 0 and log_fn is not None:
            log_fn("[Iteration %d/%d] running_train_loss=%f, lr=%.1E" %
                   (iter_idx + 1,
                    batches_per_iter, running_loss / num_batches,
                    Decimal(optimizer.param_groups[0]['lr'])))

    avg_acc = correct_preds / num_samples
    avg_loss = running_loss / num_batches
    return avg_loss, avg_acc


def evaluate_model(model, dataloader, criterion=None):
    running_loss, num_batches, num_samples, correct_preds = 0., 0, 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            y_ = model(x)
            running_loss += criterion(y_, y).item() if criterion is not None else 0.

            # Accuracy
            correct_preds += compute_correct_predictions(y_, y)

            # Update monitors
            num_batches += 1
            num_samples += y.size(0)

        avg_acc = correct_preds / num_samples
        avg_loss = running_loss / num_batches
    return avg_loss, avg_acc


def create_lr_schedule(args):
    if args.lr_annealing_type == 0:
        lr_scheduler = PiecewiseSchedule([(0, args.lr), (args.num_epochs, args.lr)], outside_value=args.lr)
    elif args.lr_annealing_type == 1:
        lr_scheduler = PiecewiseSchedule([(0, args.lr),
                                          (args.num_epochs / 3, args.lr * 0.1),
                                          (args.num_epochs * 2 / 3, args.lr * 0.01)],
                                         outside_value=args.lr * 0.01)
    elif args.lr_annealing_type == 2:
        lr_scheduler = PiecewiseSchedule([(0, args.lr),
                                          (args.num_epochs / 2, args.lr * 0.1),
                                          (args.num_epochs * 3 / 4, args.lr * 0.01)],
                                         outside_value=args.lr * 0.01)
    elif args.lr_annealing_type == 3:  # schedule in WRN paper
        max_epochs = 200  # number of epochs in WRN paper
        lr_scheduler = PiecewiseSchedule([(0, args.lr),
                                          ((60. / max_epochs) * args.num_epochs, args.lr * 0.2),
                                          ((120. / max_epochs) * args.num_epochs, args.lr * (0.2 ** 2)),
                                          ((160. / max_epochs) * args.num_epochs, args.lr * (0.2 ** 3))],
                                         outside_value=args.lr * (0.2 ** 3))
    elif args.lr_annealing_type == 4:  # schedule for ResNet from He et al 2015
        lr_scheduler = PiecewiseSchedule([(0, args.lr),
                                          (91, 0.1*args.lr),
                                          (136, 0.01*args.lr)],
                                         outside_value=0.01*args.lr)
    elif args.lr_annealing_type == 5:
        lr_scheduler = PiecewiseSchedule([(0, args.lr), (91, 0.1 * args.lr), (136, 0.01 * args.lr)],
                                         outside_value=0.01 * args.lr)
    else:
        raise ValueError("Unsupported --lr_annealing_type:", args.lr_annealing_type)
    return lr_scheduler

import argparse

import torch.nn as nn

from util.misc import *
from util.graph_def import *
from models.nets import ARCHITECTURES
from data.loaders import load_data, DATASETS
from util.schedules import linear_interpolation
from util.hessian import hessian_spectral_norm_approx


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# Experiment params
parser.add_argument('--dataset', default=None, choices=DATASETS,
                    help='dataset to train on')

# Architecture and init
parser.add_argument('--nn', default=None, choices=ARCHITECTURES,
                    help='neural network architecture')
parser.add_argument('--hidden_size', default=128, type=int,
                    help='number of hidden units for MLP')
parser.add_argument('--num_layers', default=2, type=int,
                    help='number of hidden layers for MLP/CNN')
parser.add_argument('--num_blocks', default=2, type=int,
                    help='number of residual blocks for ResNet')
parser.add_argument('--wrn_n', default=6, type=int,
                    help='N for WRN (number of blocks per stage, with num_layers=6N+4)')
parser.add_argument('--wrn_k', default=2, type=int,
                    help='k for WRN (widening factor)')
parser.add_argument('--wrn_reduced_memory', default=False, action='store_true',
                    help='Use stride=2 in WRN\'s conv1 to reduce memory footprint in very deep nets')
parser.add_argument('--init', default='orthogonal_proposed',
                    choices=['he', 'orthogonal',
                             'he_datadep', 'orthogonal_datadep',
                             'he_proposed', 'orthogonal_proposed'],
                    help='Initialization scheme.\n'
                         'he/orthogonal: pytorch default WN init with He/orthogonal init for weights\n'
                         '{he/orthogonal}_datadep: data-dependent WN init with He/orthogonal init for weights\n'
                         '{he/orthogonal}_proposed: proposed WN init with He/orthogonal init for weights')
parser.add_argument('--init_extra_param', default=None, choices=[None, 'hanin'],
                    help='extra param for WRN init; used for baselines in the 10k layer experiments mostly')
parser.add_argument('--weight_norm', default=False, action='store_true',
                    help='whether to use Weight Normalization')
parser.add_argument('--batch_norm', default=False, action='store_true',
                    help='whether to use Batch Normalization')

# Hyperparameters
parser.add_argument('--seed', default=1, type=int,
                    help='random seed')
parser.add_argument('--num_epochs', default=200, type=int,
                    help='number of optimization epochs')
parser.add_argument('--optimizer', default='sgd', choices=['sgd'],
                    help='optimizer type')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size')
parser.add_argument('--mini_batch_size', default=None, type=int,
                    help='for very large models, the batch size will be split into several batches of this size')
parser.add_argument('--lr', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_annealing_type', default=0, type=int, choices=list(range(MAX_LR_SCHEDULE + 1)),
                    help='lr annealing type')
parser.add_argument('--momentum', default=0, type=float,
                    help='momentum for SGD')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='weight decay rate')
parser.add_argument('--cutout', default=False, action='store_true',
                    help='whether to use cutout')
parser.add_argument('--warmup_epochs', default=0, type=float,
                    help='duration of lr warmup period in epochs')
parser.add_argument('--warmup_lr', default=0, type=float,
                    help='initial lr for warmup')

# Validation set
parser.add_argument('--val_fraction', default=0.1, type=float,
                    help='fraction of withheld validation data')

# Logging
parser.add_argument('--save_interval', default=1, type=int,
                    help='number of epochs between checkpoints')
parser.add_argument('--model', default=None,
                    help='model name (will log to log/model_prefix/model_name)')
parser.add_argument('--model_prefix', default=None,
                    help='prefix for the model name (will log to log/model_prefix/model_name)')
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")

# Other
parser.add_argument('--cudnn_deterministic', default=False, action='store_true',
                    help='disable stochastic cuDNN operations (enable them for faster execution)')
parser.add_argument('--log_every_iter', default=False, action='store_true',
                    help='whether to log train loss after every SGD update')
parser.add_argument('--hessian', default=False, action='store_true',
                    help='whether to compute spectral norm instead of training')


args = parser.parse_args()

# Make sure that we chose a valid init scheme
if 'proposed' in args.init or 'datadep' in args.init:
    assert args.weight_norm, "'{}' init will only work with Weight Normalized networks".format(args.init)

if args.mini_batch_size is None:
    args.mini_batch_size = args.batch_size
assert args.batch_size % args.mini_batch_size == 0

# Set random seed
set_seed(args.seed)

if args.cudnn_deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set up directories
model_name = get_model_name(args)
save_dir = get_model_dir(model_name)

# Load training status
try:
    status = load_status(save_dir)
except OSError:
    status = {"num_epochs": 0, "sgd_steps": 0, "best_val_acc": -1., "test_acc": -1., "best_model_epoch": -1}

# Save config and status files
save_config(args, save_dir)
save_status(status, save_dir)

# Set up loggers
logger = get_loggers(save_dir)
logger.info("\nLogging to %s\n" % os.path.abspath(save_dir))

# Create data loaders
trainloader, validloader, testloader, num_classes = load_data(args)
input_size = get_input_size(args)

# Create model
sample_batch, _ = iter(trainloader).__next__()
model = create_model(args, input_size, num_classes, sample_batch=sample_batch)
logger.info(model)

# Create optimizer
optimizer = create_optimizer(model, args)

try:
    load_model(model, save_dir, 'model')
    load_model(optimizer, save_dir, 'optimizer')
    loaded_model = True
except:
    loaded_model = False
if torch.cuda.is_available():
    model.cuda()
    # Ugly fix to a bug in PyTorch with optimizer loading when CUDA is available
    #  https://github.com/pytorch/pytorch/issues/2830
    if loaded_model:
        optimizer = create_optimizer(model, args)
        load_model(optimizer, save_dir, 'optimizer')
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()


# CSV logger
csv_file, csv_writer = get_csv_writer(save_dir)
csv_header = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"]
if not loaded_model:  # avoid writing a header in the middle of the file
    csv_writer.writerow(csv_header)
    csv_file.flush()

# Tensorboard
if args.tb:
    from tensorboardX import SummaryWriter
    tb_writer = SummaryWriter(save_dir)

num_epochs = status["num_epochs"]
sgd_steps = status["sgd_steps"]
best_val_acc = status["best_val_acc"]

logger.info("Parameter count: %s" % pretty_number(sum(p.numel() for p in model.parameters() if p.requires_grad)))
if loaded_model:
    logger.info("Model loaded successfully\n")
else:
    logger.info("Training from scratch\n")


# Train
try:
    batches_per_update = args.batch_size // args.mini_batch_size
    lr_scheduler = create_lr_schedule(args)
    criterion = nn.CrossEntropyLoss()

    if args.hessian:
        spectral_norm = hessian_spectral_norm_approx(model, trainloader, criterion, M=40, seed=args.seed, logger=logger)
        exit(0)

    while num_epochs < args.num_epochs:
        # Update learning rate
        lr = lr_scheduler.value(num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * model.get_lr_multiplier(param_group)

        if num_epochs < args.warmup_epochs:
            num_batches = len(trainloader)
            offset = int(args.warmup_lr == 0)
            alphas = [(num_epochs + i / num_batches) / args.warmup_epochs for i in range(offset, num_batches+offset)]
            lr_list = [linear_interpolation(args.warmup_lr, lr, alpha ** 2) for alpha in alphas]
        else:
            lr_list = None

        train_loss, train_acc = train_epoch(model, optimizer, trainloader, criterion,
                                            batches_per_update=batches_per_update,
                                            log_fn=logger.info if args.log_every_iter else None,
                                            lr_list=lr_list)
        val_loss, val_acc = evaluate_model(model, validloader, criterion)

        # Log to file and stdout
        logger.info('[Epoch %d] train_loss = %.5f    val_loss = %.5f    train_acc = %.5f    val_acc = %.5f' %
                    (num_epochs, train_loss, val_loss, train_acc, val_acc))

        # Log to CSV file
        csv_data = [num_epochs, train_loss, val_loss, train_acc, val_acc, lr]
        csv_writer.writerow(csv_data)
        csv_file.flush()

        # Log to TensorBoard
        if args.tb:
            for field, value in zip(csv_header, csv_data):
                tb_writer.add_scalar(field, value, csv_data[0])  # csv_data[0] = num_epochs

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            status["best_model_epoch"] = num_epochs
            save_model(model, save_dir, 'model_best')

        # Save last model
        save_model(model, save_dir, 'model')
        save_model(model, save_dir, 'optimizer')

        # Update and save status
        num_epochs += 1
        sgd_steps += len(trainloader)
        status["num_epochs"] = num_epochs
        status["sgd_steps"] = sgd_steps
        status["best_val_acc"] = best_val_acc
        save_status(status, save_dir)

    logger.info("Finished training!")
except KeyboardInterrupt:
    logger.info("\nCTRL+C received. Stopping training...")
except Exception as e:
    logger.info("Something went wrong:")
    logger.info(e)
    logger.info("Stopping training...")
finally:
    if model_exists(save_dir, 'model_best'):
        if testloader is not None:
            logger.info("Evaluating best model (epoch %d)..." % status["best_model_epoch"])
            load_model(model, save_dir, 'model_best')
            _, test_acc = evaluate_model(model, testloader)
            status["test_acc"] = test_acc
            logger.info("Test accuracy: %.5f" % test_acc)
            save_status(status, save_dir)
        else:
            logger.info('No test data was provided. Skipping evaluation...')
            status["test_acc"] = status["best_val_acc"]
            save_status(status, save_dir)
            logger.info("Validation accuracy: %.5f" % status["best_val_acc"])
    else:
        logger.info('No checkpoints found. Skipping evaluation...')

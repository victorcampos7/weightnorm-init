import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm


def hessian_spectral_norm_approx(model, loader, criterion, M=20, seed=777, logger=None):
    model.train()

    def get_Hv(v):
        flat_grad_loss = None
        flat_Hv = None
        ind = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(loader), total=int(0.1*len(loader))):
            ind += 1
            if ind > 0.1 * len(loader):
                break
            model.zero_grad()

            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            flat_grad_loss = torch.cat([grad.view(-1) for grad in grads])

            grad_dot_v = (flat_grad_loss * v).sum()

            Hv = torch.autograd.grad(grad_dot_v, model.parameters())
            if flat_Hv is None:
                flat_Hv = torch.cat([grad.contiguous().view(-1) for grad in Hv])
            else:
                flat_Hv.data.add_(torch.cat([grad.contiguous().view(-1) for grad in Hv]).data)

        flat_Hv.data.mul_(1./ind)

        return flat_Hv

    p_order = [p[0] for p in model.named_parameters()]
    params = model.state_dict()
    init_w = np.concatenate([params[w].cpu().numpy().reshape(-1,) for w in p_order])

    rng = np.random.RandomState(seed)

    if torch.cuda.is_available():
        v = Variable(torch.from_numpy(rng.normal(0.0, scale=1.0, size=init_w.shape).astype("float32")).cuda())
    else:
        v = Variable(torch.from_numpy(rng.normal(0.0, scale=1.0, size=init_w.shape).astype("float32")))

    for i in range(M):
        Hv = get_Hv(v)
        pmax = torch.max(Hv.data).item()
        nmax = torch.min(Hv.data).item()
        if pmax < np.abs(nmax):
            spec_norm = nmax
        else:
            spec_norm = pmax
        v = Hv.detach()
        v.data.mul_(1./spec_norm)
        if logger is not None:
            logger.info('iter: {}/{}, spectral norm: {} '.format(i, M, spec_norm))

    return spec_norm

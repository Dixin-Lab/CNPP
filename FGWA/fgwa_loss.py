import torch


def mle_loss(data, model):
    """
    compute log-likelihood of seq under model
    """
    J = 100

    event_tensor, dtime_tensor, mask_tensor = data[0], data[1], data[2]
    batch_size, T = event_tensor.size()
    # Initial
    model.start(batch_size)
    loglik = 0.0
    for i in range(T):
        # real event & noise non-event
        event = event_tensor[:, i ].unsqueeze(-1)
        dtime = dtime_tensor[:, i ].unsqueeze(-1)
        mask = mask_tensor[:, i ].unsqueeze(-1)
        p_real = model.compute_intensities(event, dtime, mask)
        loglik += torch.sum(torch.log(p_real))
        # print(p_real.squeeze(), loglik)

        integral = 0.0
        for j in range(J):
            dtj = dtime * torch.empty_like(dtime).uniform_(0, 1)
            integral += torch.sum(model.compute_total_intensity(dtj, mask), dim=1)
        loglik -= torch.sum(dtime * integral / J)

        model.forward(event, dtime, mask)
    return loglik


def distance_tensor(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C, D] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.abs(x_col - y_row) ** p
    return distance
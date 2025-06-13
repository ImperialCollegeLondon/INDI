import torch
from tqdm.auto import tqdm

device = torch.device("mps")


class DTIModel(torch.nn.Module):
    def __init__(self, bvals, bvecs):
        super(DTIModel, self).__init__()

        self.register_buffer("bvals", bvals)
        self.register_buffer("bvecs", bvecs)

    def forward(self, S0, tensor):
        pred_images = torch.empty(S0.shape[0], self.bvals.shape[0], S0.shape[1], S0.shape[2], device=S0.device)
        for i in range(self.bvals.shape[0]):
            b = torch.linalg.matmul(tensor, self.bvecs[i])
            a = torch.linalg.vecdot(self.bvecs[i], b)
            pred_images[:, i] = S0 * torch.exp(-self.bvals[i] * a)
        return pred_images


class DTILoss(torch.nn.Module):
    def __init__(self, bvals, bvecs):
        super(DTILoss, self).__init__()

        self.model = DTIModel(bvals, bvecs)

    def forward(self, pred, images):
        S0 = pred[:, 0, :, :]

        # Predicts the Cholesky decomposition of the diffusion tensor, so that it is positive definite
        # D = R^T @ R
        R = torch.zeros(pred.shape[0], pred.shape[2], pred.shape[3], 3, 3, device=pred.device)
        R[:, :, :, 0, 0] = pred[:, 1, :, :]
        R[:, :, :, 0, 1] = pred[:, 2, :, :]
        R[:, :, :, 0, 2] = pred[:, 3, :, :]

        R[:, :, :, 1, 1] = pred[:, 4, :, :]
        R[:, :, :, 1, 2] = pred[:, 5, :, :]
        R[:, :, :, 2, 2] = pred[:, 6, :, :]
        tensor = torch.transpose(R, 3, 4) @ R

        # R[:, :, :, 1, 0] = R[:, :, :, 0, 1]
        # R[:, :, :, 2, 0] = R[:, :, :, 0, 2]
        # R[:, :, :, 2, 1] = R[:, :, :, 1, 2]
        # tensor = R
        pred_images = self.model(S0, tensor)
        return torch.mean((pred_images - images) ** 2)


def get_model(shape):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
        torch.nn.InstanceNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
        torch.nn.InstanceNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.Upsample(scale_factor=2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.InstanceNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.InstanceNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Upsample(scale_factor=2),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.InstanceNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.InstanceNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Upsample(scale_factor=2),
        torch.nn.Conv2d(64, 7, kernel_size=3, padding=1),
        torch.nn.Upsample(size=shape),
        # torch.nn.Tanh()
    ).to(device)
    return model


def nn_fit(data, bvals, bvecs, iterations=1000, learning_rate=1e-4):

    data = torch.tensor(data, dtype=torch.float32, device=device)
    data = data / data.max()  # Normalize the data
    model = get_model(data.shape[:2])
    data = data.permute(2, 0, 1).unsqueeze(0)  # Reshape to (1, C, H, W)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bvals = torch.tensor(bvals, dtype=torch.float32, device=device)
    bvecs = torch.tensor(bvecs, dtype=torch.float32, device=device)
    criterion = DTILoss(bvals, bvecs).to(device)

    z = torch.randn(1, 32, 16, 32).to(device)

    loss_hist = []
    pbar = tqdm(total=iterations, desc="Training Progress")
    pbar.set_postfix({"loss": 0.0})
    for epoch in range(iterations):
        model.train()
        optimizer.zero_grad()
        output = model(z)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        pbar.update(1)
    pbar.close()

    output = model(z)
    S0 = output[:, 0, :, :]

    # Predicts the Cholesky decomposition of the diffusion tensor, so that it is positive definite
    # D = R^T @ R
    R = torch.zeros(output.shape[0], output.shape[2], output.shape[3], 3, 3, device=output.device)
    R[:, :, :, 0, 0] = output[:, 1, :, :]
    R[:, :, :, 0, 1] = output[:, 2, :, :]
    R[:, :, :, 0, 2] = output[:, 3, :, :]

    R[:, :, :, 1, 1] = output[:, 4, :, :]
    R[:, :, :, 1, 2] = output[:, 5, :, :]
    R[:, :, :, 2, 2] = output[:, 6, :, :]
    tensor = torch.transpose(R, 3, 4) @ R

    # R[:, :, :, 1, 0] = R[:, :, :, 0, 1]
    # R[:, :, :, 2, 0] = R[:, :, :, 0, 2]
    # R[:, :, :, 2, 1] = R[:, :, :, 1, 2]
    # tensor = R

    return tensor.cpu().detach().numpy(), S0.cpu().detach().numpy(), loss_hist

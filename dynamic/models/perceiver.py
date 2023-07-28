import torch
from perceiver_pytorch import PerceiverIO
import numpy as np
import random, os
from nnfabrik import builder
from tqdm import tqdm
from torch import nn, optim
from training.measures import correlation
import torch.nn.functional as F

seed = 18

random.seed(seed)
np.random.seed(seed)
cuda = torch.cuda.is_available()
if cuda:
    device = "cuda"
else:
    device = "cpu"
torch.manual_seed(seed)


class ImgPerceiver(nn.Module):
    def __init__(self, perceiver):
        super(ImgPerceiver, self).__init__()
        self.perceiver = perceiver

    def forward(self, x):
        x = torch.flatten(x, start_dim=2)
        x = self.perceiver(x)
        x = torch.squeeze(x)
        x = F.softplus(x)
        return x


def perceiver_model(num_of_cells, num_of_channels, img_h, img_w):
    dim = img_h * img_w
    perceiver = PerceiverIO(
        depth=1,
        dim=dim,
        queries_dim=dim,
        logits_dim=num_of_cells,
        num_latents=78,
        latent_dim=1,
        cross_heads=1,
        latent_heads=4,
        cross_dim_head=32,
        latent_dim_head=32,
        weight_tie_layers=False,
        decoder_ff=False,
    )
    model = ImgPerceiver(perceiver)
    return model.double()


def train(model, dataloaders, optimizer, loss_function, retina_index):
    model.train()
    losses = []
    for images, responses in tqdm(dataloaders["train"][str(retina_index + 1).zfill(2)]):
        optimizer.zero_grad()
        output = model(images.double())
        loss = loss_function(output, responses)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def validate(model, dataloaders, loss_function, correlation_func, retina_index):
    model.eval()
    losses, corrs = [], []
    for images, responses in tqdm(
        dataloaders["validation"][str(retina_index + 1).zfill(2)]
    ):
        output = model(images.double())
        loss = loss_function(output, responses)
        corr = correlation_func(output, responses, eps=1e-12)

        losses.append(loss.item())
        corrs.append(corr.detach().numpy())
    return np.mean(losses), np.mean(np.mean(corrs, axis=1))


def create_perceiver():
    pass


if __name__ == "_main__":
    lr = 1e-4
    epochs = 36
    num_of_channels = 15
    num_of_trials = 40
    batch_size = 50
    img_w = 200
    img_h = 150
    basepath = "/Users/m_vys/Documents/doktorat/CRC1456/retinal_circuit_modeling/"
    num_of_neurons = 1
    # l1 = [0.1, 0.01, 0.001, 0.0001]
    l1 = 0.00001
    l2 = 0.00001
    max_coordinate = None
    cell_index = [10, 32, 9]

    os.listdir(f"{basepath}/data")

    neuronal_data_path = os.path.join(basepath, "data/responses/")
    # neuronal_data_path = os.path.join(basepath, 'data/dummy_data/')
    training_img_dir = os.path.join(basepath, "data/non_repeating_stimuli/")
    test_img_dir = os.path.join(basepath, "data/repeating_stimuli/")
    retina_index = 0

    dataset_fn = "datasets.white_noise_loader"
    dataset_config = dict(
        neuronal_data_dir=neuronal_data_path,
        train_image_path=training_img_dir,
        test_image_path=test_img_dir,
        batch_size=batch_size,
        seed=1000,
        num_of_trials_to_use=num_of_trials,
        use_cache=True,
        movie_like=True,
        num_of_channels=num_of_channels,
        cell_index=None,
        retina_index=retina_index,
    )

    dataloaders = builder.get_data(dataset_fn, dataset_config)
    print(dataloaders)

    first_session_ID = list((dataloaders["train"].keys()))[0]
    print(first_session_ID)
    a_dataloader = dataloaders["train"][first_session_ID]
    inputs, targets = next(iter(a_dataloader))

    model = create_perceiver(
        targets.shape[1], num_of_channels=num_of_channels, img_h=150, img_w=200
    )
    model.to(device)

    loss_function = nn.PoissonNLLLoss(log_input=False, reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        train_loss = train(model, dataloaders, optimizer, loss_function, 0)
        print(f"Train loss: {train_loss}")
        validate_loss, corr = validate(
            model, dataloaders, loss_function, correlation, retina_index
        )
        print(f"Validation loss: {validate_loss}")
        print(f"Correlation: {corr}")

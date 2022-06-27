import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import model_cvae

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

lr = 1e-4
num_epochs = 300
latent_dims = 250


def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    reconstruction_error_epoch = 0.0
    kl_error_epoch = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x in tqdm(dataloader):
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = vae(x)
        # Evaluate loss
        reconstruction_error = ((torch.abs(x - x_hat)) ** 2).sum()
        kl_error = vae.encoder.kl.sum()
        loss = reconstruction_error + kl_error
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reconstruction_error_epoch += reconstruction_error.item()
        kl_error_epoch += kl_error.item()
        train_loss += loss.item()

    len_dataset = len(dataloader.dataset)
    return train_loss / len_dataset, reconstruction_error_epoch / len_dataset, kl_error_epoch / len_dataset


def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad():  # No need to track the gradients
        for x in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            # encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            # loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            loss = ((torch.abs(x - x_hat)) ** 2).sum() + vae.encoder.kl.sum()
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)


def main(x_train, x_test, model_save_path):
    vae = model_cvae.CVAE(latent_dims=latent_dims)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    vae.to(device)

    train_loader = DataLoader(x_train, batch_size=8, shuffle=True)
    valid_loader = DataLoader(x_test, batch_size=8, shuffle=True)

    for epoch in range(num_epochs):
        train_loss, reconstruction_error, kl_error = train_epoch(vae, device, train_loader, optimizer)
        val_loss, global_validation_loss = test_epoch(vae, device, valid_loader)
        print('\n EPOCH {}/{} \t train loss {:.3f} (recon error {:.3f}, kl error {:.3f}) \t val loss {:.3f}'.format(
            epoch + 1,
            num_epochs,
            train_loss,
            reconstruction_error,
            kl_error,
            val_loss))

    torch.save(vae.state_dict(), model_save_path)

import torch
from utils import config
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pypianoroll
from pypianoroll import Multitrack, Track

# Variable to read the constant values
conf = config.Config

def get_dataloader(train_data):
    data = torch.as_tensor(train_data, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.BATCH_SIZE, drop_last=True, shuffle=True)
    return data_loader

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """
    Computes gradient penalty that helps stabilize the magnitude of the gradients that the
    discriminator provies to the generator, and thus help stabilize the training of the generator.
    """
    # Get random interpolations between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = interpolates.requires_grad_(True)

    # Get the discriminator output for the interpolations
    d_interpolates = discriminator(interpolates)

    # Get gradients w.r.t the interpolations
    fake = torch.ones(real_samples.size(0), 1)
    gradients = torch.autograd.grad(
        outputs = d_interpolates,
        inputs = interpolates,
        grad_outputs = fake,
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    # Compute gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()

    return gradient_penalty


def train_one_step(generator, discriminator, d_optimizer, g_optimizer, real_samples):
    """ Trains the network for one step"""

    # Samples from the lantent distribution
    latent = torch.randn(conf.BATCH_SIZE, conf.LATENT_DIM)

    # TRAIN THE DISCRIMINATOR
    # Reset cached gradients to zero
    d_optimizer.zero_grad()

    # Get discriminator outputs for the real samples
    pred_real = discriminator(real_samples)
    # Compute the loss function
    d_loss_real = -torch.mean(pred_real)
    # Backpropogate the gradients
    d_loss_real.backward()

    # Generate fake samples with the generator
    fake_samples = generator(latent)
    # Get discriminator outputs for the fake samples
    pred_fake_d = discriminator(fake_samples.detach())
    # Compute the loss
    d_loss_fake = torch.mean(pred_fake_d)
    # Backpropogate the gradients
    d_loss_fake.backward()

    # Compute gradient penalty
    gradient_penalty = 10.0 * compute_gradient_penalty(discriminator, real_samples.data, fake_samples.data)
    # Backpropogate the gradients
    gradient_penalty.backward()

    # Update the weights
    d_optimizer.step()


    # TRAIN THE GENERATOR
    # Reset cached gradients to zero
    g_optimizer.zero_grad()
    # Get discriminator outputs for the fake samples
    pred_fake_g = discriminator(fake_samples)
    # Compute the loss
    g_loss = -torch.mean(pred_fake_g)
    # Backpropogate the gradients
    g_loss.backward()
    # Update the weights
    g_optimizer.step()

    return d_loss_fake + d_loss_real, g_loss


def start_training(data_loader, generator, discriminator, d_optimizer, g_optimizer, sample_latent):
    
    history_samples = {}
    # Create a progress bar instance for monitoring
    progress_bar = tqdm(total=conf.N_STEPS, initial=step, ncols=80, mininterval=1)

    # Start iterations
    while step <  + 1:
        # Iterate over the dataset
        for real_samples in data_loader:
            # Train the neural networks
            generator.train()
            d_loss, g_loss = train_one_step(d_optimizer, g_optimizer, real_samples[0])

            # Record smoothened loss values to LiveLoss logger
            if step > 0:
                running_d_loss = 0.05 * d_loss + 0.95 * running_d_loss
                running_g_loss = 0.05 * g_loss + 0.95 * running_g_loss
            else:
                running_d_loss, running_g_loss = 0.0, 0.0
            # liveloss.update({'negative_critic_loss': -running_d_loss})
            # liveloss.update({'d_loss': running_d_loss, 'g_loss': running_g_loss})
            
            # Update losses to progress bar
            progress_bar.set_description_str(
                "(d_loss={: 8.6f}, g_loss={: 8.6f})".format(d_loss, g_loss))
            
            if step % conf.SAMPLE_INTERVAL == 0:
                # Get generated samples
                generator.eval()
                samples = generator(sample_latent).cpu().detach().numpy()
                history_samples[step] = samples

                # Display generated samples
                samples = samples.transpose(1, 0, 2, 3).reshape(conf.N_TRACKS, -1, conf.N_PITCHES)
                tracks = []
                for idx, (program, is_drum, track_name) in enumerate(
                    zip(conf.PROGRAMS, conf.IS_DRUMS, conf.TRACK_NAMES)
                ):
                    pianoroll = np.pad(
                        samples[idx] > 0.5,
                        ((0, 0), (conf.LOWEST_PITCH, 128 - conf.LOWEST_PITCH - conf.N_PITCHES))
                    )
                    tracks.append(
                        Track(
                            name=track_name,
                            program=program,
                            is_drum=is_drum,
                            pianoroll=pianoroll
                        )
                    )
                m = Multitrack(
                    tracks=tracks,
                    tempo=conf.TEMPO_ARRAY,
                    resolution=conf.BEAT_RESOLUTION
                )
                axs = m.plot()
                plt.gcf().set_size_inches((16, 8))
                for ax in axs:
                    for x in range(
                        conf.MEASURE_RESOLUTION,
                        4 * conf.MEASURE_RESOLUTION * conf.N_MEASURES,
                        conf.MEASURE_RESOLUTION
                    ):
                        if x % (conf.MEASURE_RESOLUTION * 4) == 0:
                            ax.axvline(x - 0.5, color='k')
                        else:
                            ax.axvline(x - 0.5, color='k', linestyle='-', linewidth=1)
                plt.show()
                
            step += 1
            progress_bar.update(1)
            if step >= conf.N_STEPS:
                break

    # Saving the model
    torch.save(generator,'../models/Generator1.pt')

    return history_samples
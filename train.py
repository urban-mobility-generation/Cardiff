import os
from datetime import datetime
import torch.utils.data
from torch import optim
from argparse import ArgumentParser
import yaml

import argparse
import json

from tqdm import tqdm
from tqdm import trange
import numpy as np

from accelerate import Accelerator, DistributedDataParallelKwargs

from src.cardiff import Cardiff
from src.dit import DiT

from src.sd_unet import Unet, Super
from src.training import (trajectory_dataset,
                          create_directory,
                          get_model_params,
                          get_model_size,
                          save_training_info,
                          get_default_args,
                          load_restart_training_parameters)


from auto_encoder.traj_compressed_ae import BARTLatentCompression

from transformers import (
    BertTokenizerFast,
    BartForConditionalGeneration,
    BartConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def CardiffTrain(timestamp, args, denoisers, cardiff, autoencoder, train_dataloader, valid_dataloader, training_dir, optimizer, accelerator):
    best_loss = [torch.tensor(9999999) for i in range(len(denoisers))]

    # trainable_denoiser_indices = [cardiff.only_train_unet_number] \
    #     if cardiff.only_train_unet_number is not None else range(len(denoisers))

    trainable_denoiser_indices = range(len(denoisers))

    for epoch in trange(args.EPOCHS, desc="Training Epochs"):
        timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")

        if accelerator.is_main_process:
            print(f'\n{"-" * 20} EPOCH {epoch + 1} {"-" * 10}-{timestamp}')
            with training_dir():
                with open('training_progress.txt', 'a') as f:
                    f.write(f'{"-" * 20} EPOCH {epoch + 1} {"-" * 10}--{timestamp}\n')
            print(f'\n{"-" * 10}Training...{"-" * 10}')

        cardiff.train(True)

        running_train_loss = [0. for i in range(len(trainable_denoiser_indices))]


        for batch_num, batch in enumerate(train_dataloader):
            trajs = batch['trajs']  # traj
            attrs = batch['attrs']  # attr
            attention_mask = batch['attention_mask']
            input_ids=batch['input_ids']
            labels = batch['labels']

            encoder_outputs = autoencoder.get_encoder()(input_ids=input_ids,
                                                        attention_mask=attention_mask)
            road_segment_embed = autoencoder.get_diffusion_latent(encoder_outputs=encoder_outputs,
                                                                    attention_mask = attention_mask,
                                                                  segment_coords=torch.stack([batch['lat'], batch['lon']], dim=-1))

            losses = [0. for i in range(len(trainable_denoiser_indices))]
            for denoiser_idx in range(len(trainable_denoiser_indices)):
                denoiser_id = trainable_denoiser_indices[denoiser_idx]
                loss = cardiff(trajs,
                               attr_embeds=attrs,
                               segment_embeds=road_segment_embed,
                               denoiser_number=denoiser_id + 1,
                               labels=labels,
                               use_p_loss=args.use_p_loss)
                losses[denoiser_idx] = loss.detach()
                running_train_loss[denoiser_idx] += loss.detach()

                optimizer.zero_grad()
                # loss.backward()
                accelerator.backward(loss)
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(cardiff.parameters(), 50)

        if accelerator.is_main_process:
            # Write and batch average training loss so far
            avg_loss = [i / batch_num for i in running_train_loss]
            with training_dir():
                with open('training_progress.txt', 'a') as f:
                    f.write(f'Denoisers Avg Train Losses Epoch {epoch+1}: '
                            f'{[round(i.item(), 3) for i in avg_loss]}\n')
                    f.write(f'Denoisers Train Losses Epoch {epoch+1}: '
                            f'{[round(i.item(), 3) for i in losses]}\n')

        # Save temporary state dicts
        if (epoch+1) % 10 == 0 and accelerator.is_main_process:
            with training_dir("tmp"):
                for idx in range(len(trainable_denoiser_indices)):
                    denoiser_id = trainable_denoiser_indices[idx]
                    model_path = f"denoisers_{denoiser_id}_{epoch}_tmp.pth"
                    if accelerator.num_processes > 1:
                        torch.save(cardiff.module.denoisers[denoiser_id].state_dict(), model_path)
                    else:
                        torch.save(cardiff.denoisers[denoiser_id].state_dict(), model_path)



def main(args):
    # Get device
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


    device = accelerator.device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestamp = args.timestamp

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create training directory
    dir_path = f"./training_{timestamp}"
    training_dir = create_directory(dir_path)

    # If loading from a parameters/training directory
    if args.RESTART_DIRECTORY is not None:
        args = load_restart_training_parameters(args)
    elif args.PARAMETERS is not None:
        args = load_restart_training_parameters(args, justparams=True)

    # dataset and model

    train_dataloader, valid_dataloader, adjacency_matrix, tokenizer_vocab = trajectory_dataset(args)

    # Create denoiser, cardiff
    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cardiff_params = dict(
        first_stage_dim = config['cardiff']['first_stage_dim'],
        first_stage_len = config['cardiff']['first_stage_len'],
        second_stage_len = config['cardiff']['second_stage_seq_len'],
        timesteps=args.TIMESTEPS,
        cond_drop_prob=0.15,
        # only_train_unet_number = args.only_train_unet_number
    )
    base_dit_params = get_default_args(DiT)
    super_unet_params = get_default_args(Super)
    denoiser_params = [base_dit_params, super_unet_params]

    base_dit = DiT(**base_dit_params).to(device)
    super_unet = Unet(**super_unet_params).to(device)
    denoisers = [base_dit, super_unet]

    cardiff = Cardiff(denoisers=denoisers, **cardiff_params).to(device)

    cardiff_params = {**get_default_args(Cardiff), **cardiff_params}

    # Get the size of the model in megabytes
    model_size_MB = get_model_size(cardiff)
    print(f'Model size MB: {model_size_MB}')
    print(f'Model size MB: first_stage: {get_model_size(base_dit)}, second_stage:{get_model_size(super_unet)}')
    # Save all training info (config files, model size, etc.)
    save_training_info(args, timestamp, denoiser_params, cardiff_params, model_size_MB, training_dir)

    latent_model_path = "pretrained_auto_encoder"

    # config for compressed-AE
    with open(os.path.join(latent_model_path, 'args.json'), 'rt') as f:
        latent_model_args = json.load(f)

    latent_argparse = argparse.Namespace(**latent_model_args)

    # config for ori-AE
    ae_config = BartConfig.from_json_file("pretrained_auto_encoder/ae_config.json")

    autoencoder = BARTLatentCompression(
        config=ae_config,
        num_encoder_latents=latent_argparse.num_encoder_latents,
        num_decoder_latents=latent_argparse.num_decoder_latents,
        dim_ae=latent_argparse.dim_ae,
        num_layers=3,
        l2_normalize_latents=latent_argparse.l2_normalize_latents,
        use_coords=latent_argparse.use_coords
    )

    ae_model = torch.load(os.path.join(latent_model_path, 'model.pt'), map_location=device)
    autoencoder.load_state_dict(ae_model['model'])

    for param in autoencoder.parameters():
        param.requires_grad = False

    cardiff.set_lm(autoencoder, adjacency_matrix, tokenizer_vocab,
                   structure_loss_percent=0.1, structure_loss_weight=args.structure_loss_weight)

    optimizer = optim.Adam(cardiff.parameters(), lr=args.OPTIM_LR)

    cardiff, optimizer, train_dataloader, valid_dataloader, autoencoder = accelerator.prepare(
        cardiff, optimizer, train_dataloader, valid_dataloader, autoencoder
    )
    CardiffTrain(timestamp, args, denoisers, cardiff, autoencoder, train_dataloader, valid_dataloader, training_dir, optimizer, accelerator)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--PARAMETERS", dest="PARAMETERS", help="Parameters directory to load Imagen from",
                        default=None, type=str)
    parser.add_argument("-n", "--NUM_WORKERS", dest="NUM_WORKERS", help="Number of workers for DataLoader", default=0,
                        type=int)
    parser.add_argument("-b", "--BATCH_SIZE", dest="BATCH_SIZE", help="Batch size", default=512, type=int)
    parser.add_argument("-s", "--f", dest="IMG_SIDE_LEN", help="Side length of square Imagen output images",
                        default=128, type=int)
    parser.add_argument("-e", "--EPOCHS", dest="EPOCHS", help="Number of training epochs", default=500, type=int)
    parser.add_argument("-f", "--TRAIN_VALID_FRAC", dest="TRAIN_VALID_FRAC",
                        help="Fraction of dataset to use for training (vs. validation)", default=0.95, type=float)
    parser.add_argument("-t", "--TIMESTEPS", dest="TIMESTEPS", help="Number of timesteps in Diffusion process",
                        default=1000, type=int)
    parser.add_argument("-lr", "--OPTIM_LR", dest="OPTIM_LR", help="Learning rate for Adam optimizer", default=0.0001,
                        type=float)
    parser.add_argument("-ai", "--ACCUM_ITER", dest="ACCUM_ITER", help="Number of batches for gradient accumulation",
                        default=1, type=int)
    parser.add_argument("-rd", "--RESTART_DIRECTORY", dest="RESTART_DIRECTORY",
                        help="Training directory to resume training from if restarting.", default=None, type=str)
    parser.add_argument("-ts", "--TIMESTAMP", dest="timestamp", help="Timestamp for training directory", type=str,
                        default=None)

    parser.add_argument("-config", "--CONFIG", dest="config", help="model config", type=str,
                        default="config.yml")

    # parser.add_argument("-only_train_unet_number", "--only_train_unet_number", dest="only_train_unet_number", help="only_train_unet_number", type=int,
    #                     default=0)
    parser.add_argument("-structure_loss_weight", "--structure_loss_weight",
                        dest="structure_loss_weight", help="structure_loss_weight", type=float,
                        default=0.01)

    parser.add_argument("-use_p_loss", "--use_p_loss",
                        dest="use_p_loss", help="use_p_loss", type=bool,
                        default=False)

    # parser.add_argument("-use_cond", "--use_cond",
    #                     dest="use_cond", help="use_cond", type=bool,
    #                     default=False)

    args = parser.parse_args()
    main(args)

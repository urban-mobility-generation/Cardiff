import inspect
import json
import os
import torch

from contextlib import contextmanager

import torch.utils.data

from src.dit import DiT
from src import sd_unet
from src.sd_unet import Unet
from src.helpers import exists
from src.cardiff import Cardiff




from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader

import pandas as pd
import numpy as np

from transformers import (
    BertTokenizerFast,
    BartForConditionalGeneration,
    BartConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


def parse_sequence(seq):
    if isinstance(seq, str):
        return [int(x) for x in seq.strip().split()]
    elif isinstance(seq, list):
        return seq
    else:
        raise ValueError("unknown format")


class TrajectoryDataset(Dataset):
    def __init__(self, sequences, tokenizer, segment_coord_map, max_length=128):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.segment_coord_map = segment_coord_map

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_length]
        text = " ".join(str(x) for x in seq)
        lat = [self.segment_coord_map.get(x, (0.0, 0.0))[0] for x in seq]
        lon = [self.segment_coord_map.get(x, (0.0, 0.0))[1] for x in seq]
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        item = {key: encoding[key].squeeze(0) for key in encoding}
        item["lat"] = torch.tensor(lat + [0.0] * (self.max_length - len(lat)))
        item["lon"] = torch.tensor(lon + [0.0] * (self.max_length - len(lon)))
        item["labels"] = item["input_ids"].clone()
        return item

class CombinedDataset(Dataset):
    def __init__(self, tensor_dataset, token_dataset):
        assert len(tensor_dataset) == len(token_dataset)
        self.tensor_dataset = tensor_dataset
        self.token_dataset = token_dataset

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, idx):
        trajs, attrs = self.tensor_dataset[idx]
        token_data = self.token_dataset[idx]
        combined_sample = {
            "trajs": trajs,
            "attrs": attrs,
            "input_ids": token_data["input_ids"],
            "labels": token_data["labels"],
            "lat": token_data["lat"],
            "lon": token_data["lon"],
            "attention_mask": token_data.get("attention_mask", None)
        }
        return combined_sample


def trajectory_dataset(args, testset=False):

    def custom_collate(batch):
        token_keys = {"input_ids", "labels", "attention_mask", "lat", "lon"}
        token_features = []
        other_features = {}

        for sample in batch:
            token_feat = {k: sample[k] for k in token_keys if k in sample}
            token_features.append(token_feat)
            for k, v in sample.items():
                if k not in token_keys:
                    if k not in other_features:
                        other_features[k] = []
                    other_features[k].append(v)

        for k in other_features:
            other_features[k] = torch.stack(other_features[k], dim=0)

        collated_tokens = data_collator(token_features)

        other_features.update(collated_tokens)
        return other_features


    segment_df = pd.read_csv("train_data/edge_map_feature_chengdu.csv")
    lat_min, lat_max = segment_df["lat"].min(), segment_df["lat"].max()
    lon_min, lon_max = segment_df["lon"].min(), segment_df["lon"].max()
    segment_df["norm_lat"] = 2 * (segment_df["lat"] - lat_min) / (lat_max - lat_min) - 1
    segment_df["norm_lon"] = 2 * (segment_df["lon"] - lon_min) / (lon_max - lon_min) - 1
    segment_coord_map = {
        int(row.edge_id): (row.norm_lat, row.norm_lon) for _, row in segment_df.iterrows()
    }

    trajs = np.load('train_data/all_traj_results.npy', allow_pickle=True)
    attrs = np.load('train_data/all_attr_results.npy', allow_pickle=True)


    trajs = trajs.transpose(0,2,1)
    trajs = torch.from_numpy(trajs).float()
    attrs = torch.from_numpy(attrs).float()

    tensor_dataset = TensorDataset(trajs, attrs)


    file_path = "train_data/final_segments_all_train_data.pkl"
    trajs_df = pd.read_pickle(file_path)

    all_sequences = trajs_df['unique_id_seq'].apply(parse_sequence).tolist()
    max_seq_length = 128

    # load vocab and initialize tokenizer
    vocab_file = "pretrained_auto_encoder/bart_vocab.txt"
    tokenizer = BertTokenizerFast(vocab_file=vocab_file, do_lower_case=False)
    tokenizer.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
        "unk_token": "<unk>"
    })
    # print(tokenizer.bos_token_id)
    # print(tokenizer.eos_token_id)
    # print(tokenizer.pad_token_id)
    # print(tokenizer.decode([tokenizer.eos_token_id]))

    # build adjacency_matrix
    edge_ids = [str(int(eid)) for eid in segment_df["edge_id"]]
    edge_id_to_token = {eid: tokenizer.convert_tokens_to_ids(eid) for eid in edge_ids}
    num_tokens = len(tokenizer)
    adjacency_matrix = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)

    edge_to_uv = {str(int(row.edge_id)): (int(row.u), int(row.v)) for _, row in segment_df.iterrows()}
    from collections import defaultdict
    node_out_edges = defaultdict(set)
    for eid, (u, v) in edge_to_uv.items():
        node_out_edges[u].add(eid)
    for eid, (u, v) in edge_to_uv.items():
        neighbors = node_out_edges[v]
        for nei in neighbors:
            if eid in edge_id_to_token and nei in edge_id_to_token:
                i = edge_id_to_token[eid]
                j = edge_id_to_token[nei]
                adjacency_matrix[i, j] = True

    # build tokenizer_vocab
    vocab_dict = tokenizer.get_vocab()  # {'11425': 0, '11426': 1, ...}
    tokenizer_vocab = [None] * len(vocab_dict)
    for token_str, token_id_ in vocab_dict.items():
        tokenizer_vocab[token_id_] = token_str


    token_dataset = TrajectoryDataset(all_sequences, tokenizer,
                                      segment_coord_map=segment_coord_map,
                                      max_length=max_seq_length)

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding="longest")

    dataset = CombinedDataset(tensor_dataset, token_dataset)
    total_size = len(dataset)

    train_valid_size = int(0.9 * total_size)
    test_size = total_size - train_valid_size

    # train / test split
    train_valid_dataset, test_dataset = random_split(dataset, [train_valid_size, test_size])

    if testset:
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=args.BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=1,
                                                       collate_fn=custom_collate)
        return test_dataloader
    else:
        # Torch train/valid dataset, Split into train/valid
        train_size = int(args.TRAIN_VALID_FRAC * train_valid_size)
        valid_size = train_valid_size - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_valid_dataset, [train_size, valid_size])

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=args.BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=1,
                                                       collate_fn=custom_collate)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=args.BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=1,
                                                       collate_fn=custom_collate)
        return train_dataloader, valid_dataloader, adjacency_matrix, tokenizer_vocab


def load_restart_training_parameters(args, justparams=False):
    """
    :param args: Arguments Namespace returned from parsing :func:`~.minimagen.training.get_minimagen_parser`.
    :param justparams: Whether loading from a parameters directory rather than a full training directory.
    """
    if justparams:
        params = args.PARAMETERS
    else:
        directory = args.RESTART_DIRECTORY
        # Get file to parse
        params = os.path.join(directory, "parameters")

    file = list(filter(lambda x: x.startswith("training_"), os.listdir(params)))[0]
    with open(os.path.join(params, file), 'r') as f:
        lines = f.readlines()

    # Parse relevant args into dict
    to_keep = ["MAX_NUM_WORDS", "IMG_SIDE_LEN", "T5_NAME", "TIMESTEPS"]
    lines = list(filter(lambda x: True if True in [x.startswith(f"--{i}") for i in to_keep] else False, lines))
    d = {}
    for line in lines:
        s = line.split("=")
        try:
            d[s[0][2:]] = int(s[1][:-1])
        except:
            d[s[0][2:]] = s[1][:-1]

    # Replace relevant values in arg dict
    args.__dict__ = {**args.__dict__, **d}
    return args




def create_directory(dir_path):
    """
    creates
    subdirectories "parameters", "state_dicts", and "tmp" under the parent directory which can be similarly
    :param dir_path: Path of directory to create
    """
    original_dir = os.getcwd()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        for i in ["parameters", "state_dicts", "tmp", "results"]:
            os.makedirs(os.path.join(dir_path, i))

    @contextmanager
    def cm(subpath=""):
        os.chdir(os.path.join(dir_path, subpath))
        yield
        os.chdir(original_dir)

    return cm


def get_model_size(gen_model):
    """Returns model size in MB"""
    param_size = 0
    for param in gen_model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in gen_model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024 ** 2


def save_training_info(args, timestamp, unets_params, imagen_params, model_size, training_dir):
    """
    Saves training info to training directory
    :param args: Arguments Namespace/dict from argparsing :func:`~.minimagen.training.get_minimagen_parser` parser.
    :param timestamp: Training timestamp
    :param unets_params: List of parameters of Unets to save.
    :param imagen_params: Imagen parameters to save
    :param training_dir: Context manager returned from :func:`~.minimagen.training.create_directory`
    :return:
    """
    # Save the training parameters
    with training_dir("parameters"):
        with open(f"training_parameters_{timestamp}.txt", "w") as f:
            for i in args.__dict__.keys():
                f.write(f'--{i}={getattr(args, i)}\n')

    with training_dir():
        with open('training_progress.txt', 'a') as f:
            if args.RESTART_DIRECTORY is not None:
                f.write(f"STARTED FROM CHECKPOINT {args.RESTART_DIRECTORY}\n")
            f.write(f'model size: {model_size:.3f}MB\n\n')

    # Save parameters
    with training_dir("parameters"):
        for idx, param in enumerate(unets_params):
            with open(f'denoiser_{idx}_params_{timestamp}.json', 'w') as f:
                json.dump(param, f, indent=4)
        with open(f'cardiff_params_{timestamp}.json', 'w') as f:
            json.dump(imagen_params, f, indent=4)


def get_model_params(parameters_dir):
    """
    Returns the U-Net parameters and Imagen parameters saved in a "parameters" subdirectory of a training folder.
    :param parameters_dir: "parameters" subdirectory from which to load.
    :return: (unets_params, im_params) where unets_params is a list where the parameters index corresponds to the
        Unet number in the Imagen instance.
    """
    im_params = None
    unets_params = []

    # Find appropriate files
    for file in os.listdir(parameters_dir):
        if file.startswith('cardiff'):
            im_params = file
        elif file.startswith('denoiser_'):
            unets_params.append(file)

    # Make sure UNets params are sorted properly
    unets_params = sorted(unets_params, key=lambda x: int(x.split('_')[1]))

    for idx, filepath in enumerate(unets_params):
        print(filepath)
        with open(os.path.join(parameters_dir, f'{filepath}'), 'r') as f:
            unets_params[idx] = json.loads(f.read())

    with open(os.path.join(parameters_dir, f'{im_params}'), 'r') as f:
        im_params = json.loads(f.read())

    return unets_params, im_params


def get_default_args(object):
    """Returns a dictionary of the default arguments of a function or class"""
    # For any subclass of Unet but not Unet itself
    if issubclass(object, sd_unet.Unet) and not object is sd_unet.Unet:
        return {**get_default_args(sd_unet.Unet), **object.defaults}

    signature = inspect.signature(object)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def _read_params(directory, filename):
    """Returns dictionary from JSON config file in the parameters folder of a training directory"""
    with open(os.path.join(directory, "parameters", filename), 'r') as _file:
        return json.loads(_file.read())


def load_params(directory):
    """
    Loads parameters from a training directory
    :param directory: Path of training directory generated by training
    :return: (unets_params, cardiff_params)
    """
    # Files in parameters directory
    files = os.listdir(os.path.join(directory, "parameters"))
    # Filter only param files for U-Nets
    unets_params_files = sorted(list(filter(lambda x: x.startswith("denoiser_", ), files)),
                                key=lambda x: int(x.split("_")[1]))

    # Load U-Nets / MinImagen parameters
    unets_params = [_read_params(directory, f) for f in unets_params_files]
    cardiff_params_files = _read_params(directory, list(filter(lambda x: x.startswith("cardiff_"), files))[0])
    return unets_params, cardiff_params_files


def _instatiate_minimagen(directory):
    # TODO: When restarted training, parameters folder only has the cmd line args, not the unet/imagen params.
    #   had to copy from training folder this one was restarted from. Fix this so it copies.
    """ Instantiate an Imagen model with given parameters """
    denoisers_params, cardiff_params_files = load_params(directory)

    return Cardiff(denoisers=[DiT(**denoisers_params[0]), Unet(**denoisers_params[1])], **cardiff_params_files)


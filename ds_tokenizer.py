from pathlib import Path
from typing import Optional


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, andom_split, random_split
from datasets import Dataset as DS



from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


###############################################################################
# Tokenizer

def yield_sentences(ds, key = 'sentence'):
    for item in ds:
        yield item[key]


def create_tokenizer(ds: DS, key: str, tokenizer_path: Optional[str] = None) -> Tokenizer:
    """
    
    """
    tkph = Path(tokenizer_path)
    if not tkph.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace() # split words based on whitespaces between
        
        trainer = WordLevelTrainer(special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency = 2) 
        
        tokenizer.train_from_iterator(yield_sentences(ds, key), trainer = trainer)

        tokenizer.save(str(tokenizer_path))
    else: 
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

# Dataset utilities
class CompressionDataset(Dataset):
    def __init__(self, ds: DS, tokenizer_src, tokenizer_tgt, src_seq_len, tgt_seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text, tgt_text = src_target_pair['sentence'], src_target_pair['compression']


        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        

        # padding
        enc_num_padding_tokens = self.src_seq_len - len(enc_input_tokens) - 2 # -2 : [SOS], [EOS]
        dec_num_padding_tokens = self.tgt_seq_len - len(dec_input_tokens) - 1 # we do not want to have [EOS]

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.src_seq_len
        assert decoder_input.size(0) == self.tgt_seq_len
        # assert label.size(0) == self.src_seq_len


        return {'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
                'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & decoder_mask(decoder_input.size(0)),
                'label': label,
                'src_text': src_text,
                'tgt_text': tgt_text
                }
    

def decoder_mask(size: int) -> torch.Tensor:
    """
    Generates a mask for the decoder to prevent attention to subsequent positions in the sequence.

    This mask is an upper triangular matrix of ones, which is then inverted to zeros and converted to a boolean mask.
    The resulting mask ensures that the decoder at a given position can only attend to earlier positions.

    Args:
        size (int): The size of the mask, typically the length of the target sequence.

    Returns:
        torch.Tensor: A boolean mask of shape (1, size, size), where False values indicate masked (unattendable) positions.
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0




def ds_from_parquet(filename: str, train_size: float, tokenizer_src_path: Optional[str] = None, tokenizer_tgt_path: Optional[str] = None, src_seq_len: int = None, tgt_seq_len: int = None, batch_size: int = 32) -> tuple:

    assert train_size > 0 and train_size < 1, "Train size can't be outside (0; 1) range "
    
    fpath = Path(filename)
    dataset = DS.from_parquet(fpath)
    
    # build tokenizers for source and targets
    tokenizer_src = create_tokenizer(dataset, 'sentence', tokenizer_src_path)
    tokenizer_tgt = create_tokenizer(dataset, 'compression', tokenizer_tgt_path)

    # train-val split
    train_ds_size = int(train_size * len(dataset))
    val_ds_size = len(dataset) - train_ds_size
    
    
    train_ds, val_ds = random_split(dataset, [train_ds_size, val_ds_size])

    if src_seq_len is None:
        src_len = 0

        for item in dataset:
            src_ids = tokenizer_src.encode(item['sentence']).ids 
            src_len = max(src_len, len(src_ids))

        print(f'Max length of source sentence:', src_len)
        src_len += 2 # [SOS] and [EOS] tokens
    else:
        src_len = src_seq_len

    if tgt_seq_len is None:
        tgt_len = 0

        for item in dataset:
            tgt_ids = tokenizer_tgt.encode(item['compression']).ids 
            tgt_len = max(tgt_len, len(tgt_ids))

        print(f'Max length of target sentence:', src_len)
        tgt_len += 1 # [SOS] token
    else:
        tgt_len = tgt_seq_len

    train_ds = CompressionDataset(train_ds, tokenizer_src, tokenizer_tgt, src_seq_len=src_len, tgt_seq_len=tgt_len)
    val_ds = CompressionDataset(val_ds, tokenizer_src, tokenizer_tgt, src_seq_len=src_len, tgt_seq_len=tgt_len)


    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

# TODO: zaimplementowac metode train do modelu i podpiac to
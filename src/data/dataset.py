from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


import numpy as np
import torch


def texts_tokenize(tokenizer, texts):
    inputs = []
    for txt in texts:
        input_info = tokenizer(txt,
                               add_special_tokens=True,
                               return_offsets_mapping=False,
                               truncation="only_first",
                               max_length=512,
                               return_tensors="pt"
                          )
        inputs.append(input_info)

    return inputs


def prepare_input(code_tokenizer, md_tokenizer, notebook):
    code_info = texts_tokenize(code_tokenizer, notebook.codes)
    md_info = texts_tokenize(md_tokenizer, notebook.markdowns)

    return {
            "code_inputs_id": [item["input_ids"] for item in code_info],
            "md_inputs_id": [item["input_ids"] for item in md_info],
            "labels": notebook.labels
            }


class TrainDataset(Dataset):
    def __init__(self, code_tokenizer, md_tokenizer, notebooks):
        self.code_tokenizer = code_tokenizer
        self.md_tokenizer = md_tokenizer
        self.notebooks = notebooks

    def __len__(self):
        return len(self.notebooks)

    def __getitem__(self, item):
        inputs = prepare_input(self.code_tokenizer, 
                self.md_tokenizer,
                self.notebooks[item])
        return inputs


def collate_helper(batch, key):
    sequence = []
    mask = []
    doc_lengths = [len(sample[key]) for sample in batch]
    for sample in batch:
        sequence.extend(sample[key])

    sent_lengths = [len(sent) for sent in sequence]
    max_length = max(sent_lengths)

    for sent in sent_lengths:
        ground = [1] * len(sent)
        pad = [0] * (max_length - len(sent))
        mask.append(ground.extend(pad))

    sequence = pad_sequence(sequence, batch_first=True)

    return {
            "sequence": sequence,
            "masks": mask,
            "doc_lengths": doc_lengths
            }


def attention_mask_generate(query_lengthes, key_lengthes):
    L = max(query_lengthes)
    S = max(key_lengthes)
    masks = []
    for l, s in zip(query_lengthes, key_lengthes):
        mask = torch.zeros((L, S))
        mask[:l, :s] = 1

    masks = torch.stack(masks)

    return masks


def collate_fn(batch):
    output = dict()

    code_data = collate_helper(batch, "code_inputs_id")
    md_data = collate_helper(batch, "md_inputs_id")
    
    output["code_inputs_id"] = code_data["sequence"]
    output["code_masks"] = code_data["masks"]
    output["code_doc_lengths"] = code_data["doc_lengths"]

    output["md_inputs_id"] = md_data["sequence"]
    output["md_masks"] = md_data["masks"]
    output["md_doc_lengths"] = md_data["doc_lengths"]

    labels = [sample["labels"] for sample in batch]
    output["labels"] = labels

    output["attention_masks"] = attention_mask_generate(code_data["doc_lengths"], md_data["doc_lengths"])
    
    return output

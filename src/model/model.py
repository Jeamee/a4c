import re

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.functional import f1_score
from transformers import AdamW, AutoConfig, AutoModel

import bitsandbytes as bnb
from loss.sce import SCELoss
from pytorchcrf import CRF
from utils import Freeze
from utils import GradualWarmupScheduler, ReduceLROnPlateau, span_decode


class OrderModel(pl.LightningModule):
    def __init__(
        self,
        model_name,
        num_train_steps,
        transformer_learning_rate,
        num_labels,
        span_num_labels,
        steps_per_epoch,
        dynamic_merge_layers,
        loss="ce",
        sce_alpha=1.0,
        sce_beta=1.0,
        label_smooth=0.0,
        decoder="softmax",
        max_len=4096,
        merge_layers_num=-2,
        warmup_ratio=0.05,
        lr_decay=1.,
        finetune=False,
        gradient_ckpt=False,
        max_position_embeddings=None,
        use_tpu=False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cur_step = 0
        self.max_len = max_len
        self.transformer_learning_rate = transformer_learning_rate
        self.dynamic_merge_layers = dynamic_merge_layers
        self.merge_layers_num = merge_layers_num
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.span_num_labels = span_num_labels
        self.label_smooth = label_smooth
        self.decoder = decoder
        self.warmup_ratio = warmup_ratio
        self.finetune = finetune
        self.lr_decay = lr_decay
        self.use_tpu = use_tpu

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7


        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
                
            }
        )
        self.num_layers = config.num_hidden_layers
        
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        if gradient_ckpt:
            self.transformer.gradient_checkpointing_enable()
            

        self.attention = nn.MultiheadAttention(config.hidden_size, 12, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        self.loss_layer = nn.CrossEntropyLoss()
            
            
    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]

        transformer_param_optimizer = [[] for _ in range(self.num_layers + 1)]
        other_param_optimizer = []

        for name, para in param_optimizer:
            space = name.split('.')
            if space[0] == 'transformer':
                prob_layer_id = re.findall("\d{1,2}", name)
                layer_num = int(prob_layer_id [0]) if prob_layer_id else 0
                transformer_param_optimizer[layer_num].append((name, para))
            else:
                other_param_optimizer.append((name, para))
                
        other_lr = self.transformer_learning_rate * 1
        
        self.optimizer_grouped_parameters = []
        for idx, layer in enumerate(transformer_param_optimizer):
            lr = self.lr_decay ** (self.num_layers - idx) * self.transformer_learning_rate if idx > 0 else self.transformer_learning_rate

            decay_param_dict = {"params": [p for n, p in layer if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.01, 'lr': lr}
            no_decay_param_dict = {"params": [p for n, p in layer if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0, 'lr': lr}
            self.optimizer_grouped_parameters.extend([decay_param_dict, no_decay_param_dict])

        self.optimizer_grouped_parameters.extend([
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.01, 'lr': other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0, 'lr': other_lr}
        ])
        

            
        opt = bnb.optim.AdamW8bit(self.optimizer_grouped_parameters, lr=self.transformer_learning_rate)

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )
                
        if not self.finetune:
            sch = GradualWarmupScheduler(
                opt,
                multiplier=1.1,
                warmup_epoch=int(self.warmup_ratio * self.num_train_steps) ,
                total_epoch=self.num_train_steps)
            
            return [opt], [sch]

        return opt
    
    def loss(self, outputs, targets, attention_mask):
        outputs = outputs.view(-1, self.num_labels)
        targets = targets.view(-1)
        loss = self.loss_layer(outputs, targets)
        return loss

    def monitor_metrics(self, outputs, targets, sequence_mask):
        outputs = torch.argmax(outputs, dim=-1)
        outputs = torch.masked_select(outputs, sequence_mask)
        targets = torch.masked_select(targets, sequence_mask)

        return {
                "outputs": outputs,
                "targets": targets
                }

    def forward(self, code_inputs_id=None, md_inputs_id=None, code_masks=None, md_masks=None, attention_masks=None, code_doc_lengths=None, md_doc_lengths=None):
        code_transformer_out = self.transformer(code_inputs_id, code_masks)
        md_transformer_out = self.transformer(md_inputs_id, md_masks)
        # shape: sequence length * sentence length * hidden

        # mean pooling
        code_transformer_out = torch.squeeze(code_transformer_out * code_masks / torch.sum(code_masks, -1), 1)
        md_transformer_out = torch.squeeze(md_transformer_out * md_masks / torch.sum(md_masks, -1), 1)
        # shape: sequence length * hidden

        codes = pad_sequence(torch.split(code_transformer_out, code_doc_lengths))
        mds = pad_sequence(torch.split(md_transformer_out, md_doc_lengths))
            
        sequence_output = self.dropout(sequence_output)
        
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        #probs = torch.sigmoid(logits)
        probs = torch.softmax(logits, dim=-1)
        loss = 0
        
        if self.training:
            loss1 = self.loss(torch.softmax(logits1, dim=-1), targets, attention_mask=attention_mask)
            loss2 = self.loss(torch.softmax(logits2, dim=-1), targets, attention_mask=attention_mask)
            loss3 = self.loss(torch.softmax(logits3, dim=-1), targets, attention_mask=attention_mask)
            loss4 = self.loss(torch.softmax(logits4, dim=-1), targets, attention_mask=attention_mask)
            loss5 = self.loss(torch.softmax(logits5, dim=-1), targets, attention_mask=attention_mask)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            metric = None
        else:
            metric = self.monitor_metrics(probs, targets, sequence_mask=sequence_mask)
        
        return {
            "preds": probs,
            "logits": logits,
            "loss": loss,
            "metric": metric
        }

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output["loss"]
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        outputs = output["metric"]["outputs"]
        targets = output["metric"]["targets"]
        
        return {
                "outputs": outputs,
                "targets": targets
                }

    def validation_epoch_end(self, outputs) -> None:
        preds = torch.cat([output["outputs"] for output in outputs])
        grounds = torch.cat([output["targets"] for output in outputs])
        preds = preds.long()
        grounds = grounds.long()
        preds[preds == 2] = 1
        grounds[grounds == 2] = 1
        f1 = f1_score(preds, grounds, average=None, num_classes=2)
        self.log('valid/f1', f1[1], on_epoch=True)

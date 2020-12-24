import torch.nn as nn
from Models.GPT2_Model.Model.GPT2Model import GPT2Model
from Models.GPT2_Model.Model.GPT2LMHead import GPT2LMHead


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)

    def set_tied(self):
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)

        if lm_labels is not None:
            cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
            loss = cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss

        return lm_logits, presents
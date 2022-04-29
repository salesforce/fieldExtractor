import logging
import torch
from torch import nn

from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_bert import BertLayerNorm

logger = logging.getLogger(__name__)

class LayoutlmConfig(BertConfig):
    model_type = "bert"

    def __init__(self, max_2d_position_embeddings=1024, **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings


class LayoutlmEmbeddings(nn.Module):
    def __init__(self, config):
        super(LayoutlmEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        bbox,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        h_position_embeddings = self.h_position_embeddings(
            bbox[:, :, 3] - bbox[:, :, 1]
        )
        w_position_embeddings = self.w_position_embeddings(
            bbox[:, :, 2] - bbox[:, :, 0]
        )
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            words_embeddings
            + position_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutlmModel(BertModel):

    config_class = LayoutlmConfig
    base_model_prefix = "bert"

    def __init__(self, config):
        super(LayoutlmModel, self).__init__(config)
        self.embeddings = LayoutlmEmbeddings(config)
        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, bbox, position_ids=position_ids, token_type_ids=token_type_ids
        )
        encoder_outputs = self.encoder(
            embedding_output, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class FieldExtractor(BertPreTrainedModel):
    config_class = LayoutlmConfig
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.branch_num = config.branch_num
        self.online_refine_weight = config.online_refine_weight
        self.bert = LayoutlmModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.relu = nn.ReLU()
        self.classifier= nn.Linear(config.hidden_size, config.num_labels)

        for i in range(2, self.branch_num+1):
            setattr(self, 'fc'+str(i), nn.Linear(config.hidden_size, config.hidden_size))
            setattr(self, 'classifier' + str(i), nn.Linear(config.hidden_size, config.num_labels))

        self.softmax = torch.nn.Softmax(dim=-1)
        self.init_weights()

    def guess_label(self, probs, score_thred):
        # get pseudo labels for each branch
        batch_size, num_words, num_labels = probs.shape
        score_max = torch.max(probs, dim=-1)[0]
        p_target = torch.argmax(probs, dim=-1)

        for b in range(0, batch_size):
            target = p_target[b]
            for i in range(0, num_labels):
                if i == 0:
                    continue
                inds = (target == i).nonzero(as_tuple=True)[0]
                if len(inds) == 0:
                    continue
                scores = probs[b, inds, i]
                max_ind = torch.argmax(scores)
                for j in range(0, len(inds)):
                    if j == max_ind:
                        continue
                    p_target[b][inds[j]] = 0
        p_target[score_max < score_thred] = 0

        return p_target

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        # get prediction from each branch
        logits = dict()
        logits['1'] = self.classifier(sequence_output)
        for i in range(2, self.branch_num+1):
            logits[str(i)] = getattr(self, 'classifier'+str(i))(getattr(self, 'fc'+str(i))(self.relu(sequence_output)))

        outputs = ()
        for i in range(self.branch_num, 0, -1):
            outputs += (logits[str(i)], )

        probs=dict()
        guess_labels = dict()
        # get pseudo label from each branch based on prediction
        for i in range(1, self.branch_num):
            probs[str(i)] = self.softmax(logits[str(i)])
            guess_labels[str(i)] = self.guess_label(probs[str(i)], 0.1)

        if labels is not None:

            loss_fct = CrossEntropyLoss()
            loss = 0.0

            labels_unlabeled = labels # initial pseudo labels, p0
            logits_unlabeled = dict()
            for i in range(1, self.branch_num+1):
                logits_unlabeled[str(i)] = logits[str(i)]

            attention_mask_unlabeled = attention_mask

            active_loss_unlabeled = attention_mask_unlabeled.view(-1) == 1

            active_p_labels = dict()

            active_p_labels['1'] = labels_unlabeled.view(-1)[active_loss_unlabeled]
            active_logits_unlabeled = dict()
            for i in range(1, self.branch_num + 1):
                active_logits_unlabeled[str(i)] = logits_unlabeled[str(i)].view(-1, self.num_labels)[active_loss_unlabeled]

            for i in range(2, self.branch_num + 1):
                active_p_labels[str(i)] = guess_labels[str(i-1)].view(-1)[active_loss_unlabeled]

            label_token_pad_id = CrossEntropyLoss().ignore_index
            ignored_indexes = (active_p_labels['1']==label_token_pad_id).nonzero(as_tuple=True)[0]

            for ign_ind in ignored_indexes:
                for i in range(2, self.branch_num+1):
                    if active_p_labels[str(i)][ign_ind] == 0:
                        active_p_labels[str(i)][ign_ind] = label_token_pad_id

            for i in range(1, self.branch_num + 1):
                if not torch.is_tensor(loss):
                    loss = loss_fct(active_logits_unlabeled[str(i)], active_p_labels[str(i)])
                else:
                    loss += loss_fct(active_logits_unlabeled[str(i)], active_p_labels[str(i)])
                for j in range(i-1, 0, -1):
                    if j == 1:
                        loss += self.online_refine_weight*loss_fct(active_logits_unlabeled[str(i)], active_p_labels[str(j)])
                    else:
                        loss += loss_fct(active_logits_unlabeled[str(i)], active_p_labels[str(j)])


            outputs = (loss,) + outputs

        return outputs

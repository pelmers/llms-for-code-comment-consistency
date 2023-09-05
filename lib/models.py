import torch
import torch.nn as nn

from transformers import PreTrainedModel

def get_model(model_type, model_size, freeze_base=False):
    '''
    model_type: "codebert" or "codegen"
    Return: tuple of (model, tokenizer, config)
    '''
    if model_type == 'codebert':
        print('Downloading CodeBERT config files')
        from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
        model_name = 'microsoft/codebert-base'
        config = RobertaConfig.from_pretrained(model_name)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        print('Downloading CodeGen config files')
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        model_name = f'Salesforce/codegen-{model_size}-multi'
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config.gradient_checkpointing = True
        config.use_cache = False

    class RobertaClassificationHead(nn.Module):
        """Head for sentence-level classification tasks."""

        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.out_proj = nn.Linear(config.hidden_size, 2)

        def forward(self, x, **kwargs):
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)
            x = torch.softmax(x, dim=1)
            return x


    class CodeBERTBasedModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.encoder = RobertaModel.from_pretrained(model_name, config=config)
            self.classifier = RobertaClassificationHead(config)
            self.loss_fn = nn.CrossEntropyLoss()
            if freeze_base:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        def forward(self, input_ids, attention_mask, labels=None, **kwargs):
            encoder_out = self.encoder(input_ids, attention_mask, **kwargs)
            cls_encoded = encoder_out[0][:, 0, :]  # take <s> token (equiv. to [CLS])
            prob = self.classifier(cls_encoded)
            if labels is not None:
                loss = self.loss_fn(prob, labels)
                return prob, loss, encoder_out.attentions
            else:
                return prob, 0, encoder_out.attentions


    class CodeGenBasedModel(PreTrainedModel):
        def __init__(self, config):
            super().__init__(config=config)
            codegen_model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
            codegen_model.gradient_checkpointing_enable()
            self.codegen_tf = codegen_model.transformer
            self.classifier = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, 2),
                nn.Softmax(dim=1)
            )
            self.loss_fn = nn.CrossEntropyLoss()
            self.generation_config = codegen_model.generation_config
            self.can_generate = lambda: False
            self.lm_head = nn.Sequential(*list(self.classifier.children())[:-1])
            if freeze_base:
                for param in self.codegen_tf.parameters():
                    param.requires_grad = False

        def forward(self, input_ids=None, attention_mask=None, labels=None, print_logits=False, inputs_embeds=None, **kwargs):
            # If return_dict is on, then also turn on output_hidden_states
            if kwargs.get('return_dict', False) or inputs_embeds is not None:
                kwargs['output_hidden_states'] = True
            if not self.can_generate():
                self.make_lm_head()
            codegen_out = self.codegen_tf(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
            next_tok_embed = codegen_out.last_hidden_state
            logits = self.lm_head(next_tok_embed[:, -1]).unsqueeze(1)
            prob = self.classifier(next_tok_embed[:, -1])
            if kwargs.get('return_dict', False) or print_logits:
                print(f'Note, Logit values: {logits}')
            # If return_dict, then return a CausalLMOutputWithPast
            if kwargs.get('return_dict', False) or inputs_embeds is not None:
                from transformers.modeling_outputs import CausalLMOutputWithPast
                return CausalLMOutputWithPast(
                    logits=logits,
                    past_key_values=codegen_out.past_key_values,
                    hidden_states=codegen_out.hidden_states,
                    attentions=codegen_out.attentions,
                    loss=self.loss_fn(prob, labels) if labels is not None else None
                )
            if labels is not None:
                loss = self.loss_fn(prob, labels)
                return prob, loss, codegen_out.attentions
            else:
                return prob, 0, codegen_out.attentions

        # TODO: a separate forwardmode that does contrastive learning with two inputs

        def make_lm_head(self):
            return
            # LM head is everything before the softmax on the classifier
            self.lm_head = nn.Sequential(*list(self.classifier.children())[:-1])
            self.can_generate = lambda: True

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            token_type_ids = kwargs.get("token_type_ids", None)
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)

            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                position_ids = None
            return {
                "input_ids": input_ids,
                "past_key_values": None,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }


    print('Creating model from config')
    model = CodeBERTBasedModel(config) if model_type == 'codebert' else CodeGenBasedModel(config)
    return model, tokenizer, config
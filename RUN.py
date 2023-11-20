import json
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import pickle
from tqdm import tqdm
import argparse
import random
import time
import numpy as np
from transformers import (OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
from transformers import (OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer, AutoTokenizer)
from transformers import AutoModel, BertModel, BertConfig


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import glorot, zeros
try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
# EJEMPLO DE UN STATEMENT CUALQUIERA, EN ESTE CASO DE TRAIN


{"id": "train-00000", 
    "question": 
        {"stem": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7\u00b0F (36.5\u00b0C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?", 
            "choices": [{"label": "A", "text": "Ampicillin"}, 
                          {"label": "B", "text": "Ceftriaxone"}, 
                          {"label": "C", "text": "Doxycycline"}, 
                          {"label": "D", "text": "Nitrofurantoin"}]}, 
        "answerKey": "D", 
        "statements": 
            [{"statement": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7\u00b0F (36.5\u00b0C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? Ampicillin"}, 
             {"statement": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7\u00b0F (36.5\u00b0C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? Ceftriaxone"}, 
             {"statement": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7\u00b0F (36.5\u00b0C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? Doxycycline"}, 
             {"statement": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7\u00b0F (36.5\u00b0C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient? Nitrofurantoin"}]}



""" 
En mi caso tanto el número de choices como el número de statements es variable, 
sin embargo el número de choices es igual al número de statements. 
Debería de cambiar el statement para hacer hincapié en la palabra que se va a desambiguar
"""


# LISTAS Y DICCIONARIOS DEFINIDOS

def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set):
            logits, _ = model(*input_data)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples

def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True

from transformers import AdamW
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


OPTIMIZER_CLASSES = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
    'radam': RAdam,
}

GPT_SPECIAL_TOKENS = ['_start_', '_delimiter_', '_classify_']

MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'lstm': ['lstm'],
}


MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}

EMB_PATHS = {
    'transe': 'data/transe/glove.transe.sgd.ent.npy',
    'lm': 'data/transe/glove.transe.sgd.ent.npy',
    'numberbatch': 'data/transe/concept.nb.npy',
    'tzw': 'data/cpnet/tzw.ent.npy',
    'ddb': 'data/ddb/ent_emb.npy',
}

DATASET_LIST = ['csqa', 'obqa', 'socialiqa', 'medqa_usmle']


DATASET_SETTING = {
    'csqa': 'inhouse',
    'obqa': 'official',
    'socialiqa': 'official',
    'medqa_usmle': 'official',
}

DATASET_NO_TEST = ['socialiqa']


ENCODER_DEFAULT_LR = {
    'default': 1e-3,
    'csqa': {
        'lstm': 3e-4,
        'openai-gpt': 1e-4,
        'bert-base-uncased': 3e-5,
        'bert-large-uncased': 2e-5,
        'roberta-large': 1e-5,
    },
    'obqa': {
        'lstm': 3e-4,
        'openai-gpt': 3e-5,
        'bert-base-cased': 1e-4,
        'bert-large-cased': 1e-4,
        'roberta-large': 1e-5,
    },
    'medqa_usmle': {
        'cambridgeltl/SapBERT-from-PubMedBERT-fulltext': 5e-5,
    },
}

DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'obqa': 3e-4,
    'medqa_usmle': 1e-3,
}


dataset = "medqa_usmle"
ent_emb = ['ddb']

# FUNCIONES

def get_gpt_token_num():
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    return len(tokenizer)

def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target

def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False

def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def load_input_tensors(input_jsonl_path, model_type, model_name, max_seq_length):
    if model_type in ('lstm',):
        raise NotImplementedError
    elif model_type in ('gpt',):
        raise NotImplementedError           #return load_gpt_input_tensors(input_jsonl_path, max_seq_length)
    elif model_type in ('bert', 'xlnet', 'roberta', 'albert'):
        return load_bert_xlnet_roberta_input_tensors(input_jsonl_path, model_type, model_name, max_seq_length)


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_data_arguments(parser):
    # arguments that all datasets share
    parser.add_argument('--ent_emb', default=ent_emb, nargs='+', help='sources for entity embeddings')
    # dataset specific
    parser.add_argument('-ds', '--dataset', default=dataset, choices=DATASET_LIST, help='dataset name')
    parser.add_argument('-ih', '--inhouse', type=bool_flag, nargs='?', const=True, help='run in-house setting')
    parser.add_argument('--inhouse_train_qids', default='data/{dataset}/inhouse_split_qids.txt', help='qids of the in-house training set')
    # statements
    parser.add_argument('--train_statements', default='data/{dataset}/statement/train.statement.jsonl')
    parser.add_argument('--dev_statements', default='data/{dataset}/statement/dev.statement.jsonl')
    parser.add_argument('--test_statements', default='data/{dataset}/statement/test.statement.jsonl')
    # preprocessing options
    parser.add_argument('-sl', '--max_seq_len', default=100, type=int)
    # set dataset defaults
    args, _ = parser.parse_known_args()
    parser.set_defaults(ent_emb_paths=[EMB_PATHS.get(s) for s in args.ent_emb],
                        inhouse=(DATASET_SETTING[args.dataset] == 'inhouse'),
                        inhouse_train_qids=args.inhouse_train_qids.format(dataset=args.dataset))
    data_splits = ('train', 'dev') if args.dataset in DATASET_NO_TEST else ('train', 'dev', 'test')
    for split in data_splits:
        for attribute in ('statements',):
            attr_name = f'{split}_{attribute}'
            parser.set_defaults(**{attr_name: getattr(args, attr_name).format(dataset=args.dataset)})
    if 'test' not in data_splits:
        parser.set_defaults(test_statements=None)

def add_encoder_arguments(parser):
    parser.add_argument('-enc', '--encoder', default='bert-large-uncased', help='encoder type')
    parser.add_argument('--encoder_layer', default=-1, type=int, help='encoder layer ID to use as features (used only by non-LSTM encoders)')
    parser.add_argument('-elr', '--encoder_lr', default=2e-5, type=float, help='learning rate')
    args, _ = parser.parse_known_args()
    parser.set_defaults(encoder_lr=ENCODER_DEFAULT_LR[args.dataset].get(args.encoder, ENCODER_DEFAULT_LR['default']))

def add_optimization_arguments(parser):
    parser.add_argument('--loss', default='cross_entropy', choices=['margin_rank', 'cross_entropy'], help='model type')
    parser.add_argument('--optim', default='radam', choices=['sgd', 'adam', 'adamw', 'radam'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule', default='fixed', choices=['fixed', 'warmup_linear', 'warmup_constant'], help='learning rate scheduler')
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('--warmup_steps', type=float, default=150)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='l2 weight decay strength')
    parser.add_argument('--n_epochs', default=100, type=int, help='total number of training epochs to perform.')
    parser.add_argument('-me', '--max_epochs_before_stop', default=10, type=int, help='stop training if dev does not increase for N epochs')

def add_additional_arguments(parser):
    parser.add_argument('--log_interval', default=10, type=int)
    #HEMOS MODIFICADO ESTA PARTE
    parser.add_argument('--cuda', default=False, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', default=False, type=bool_flag, nargs='?', const=True, help='run in debug mode')
    args, _ = parser.parse_known_args()
    if args.debug:
        parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)

def get_parser():
    """A helper function that handles the arguments that all models share"""
    parser = argparse.ArgumentParser(add_help=False)
    add_data_arguments(parser)
    add_encoder_arguments(parser)
    add_optimization_arguments(parser)
    add_additional_arguments(parser)
    return parser

def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)



# CÓDIGO PRINCIPAL

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)




def load_bert_xlnet_roberta_input_tensors(statement_jsonl_path, model_type, model_name, max_seq_length):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, label=None):
            self.example_id = example_id                    # 'train-00000'
            self.question = question                        # ''
            self.contexts = contexts                        # ['A 21-year-old sexual...was given?', 'A 21-year-old sexual...was given?', 'A 21-year-old sexual...was given?', 'A 21-year-old sexual...was given?']
            self.endings = endings                          # ['Gentamicin', 'Ciprofloxacin', 'Ceftriaxone', 'Trimethoprim']
            self.label = label                              # 2

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,             # lista de 100 identificadores según el id de las entidades
                    'input_mask': input_mask,           # mascara todo unos, tamaño 100
                    'segment_ids': segment_ids,         # lista de 100, todo 0 menos los 4 ultimos nodos
                    'output_mask': output_mask,         # sparse list, solo tiene tres unos
                }
                for _, input_ids, input_mask, segment_ids, output_mask in choices_features  # NOSOTROS ITERAREMOS UN NUMERO VARIADO DE VECES
            ]
            self.label = label              # El int de la respuesta correcta

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                contexts = json_dic["question"]["stem"]
                if "para" in json_dic:
                    contexts = json_dic["para"] + " " + contexts
                if "fact1" in json_dic:
                    contexts = json_dic["fact1"] + " " + contexts
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label
                    ))
        return examples
    
    """ examples es un archivo utils.data_utils.load_bert....tensors 
    contiene numerosas listas: contextos, endings, example_id, label y question

    contexts:
        0:
        'A 62-year-old woman presents for a regular check-up. She complains of lightheadedness and palpitations which occur episodically. Past medical history is significant for a myocardial infarction 6 months ago and NYHA class II chronic heart failure. She also was diagnosed with grade I arterial hypertension 4 years ago. Current medications are aspirin 81 mg, atorvastatin 10 mg, enalapril 10 mg, and metoprolol 200 mg daily. Her vital signs are a blood pressure of 135/90 mm Hg, a heart rate of 125/min, a respiratory rate of 14/min, and a temperature of 36.5°C (97.7°F). Cardiopulmonary examination is significant for irregular heart rhythm and decreased S1 intensity. ECG is obtained and is shown in the picture (see image). Echocardiography shows a left ventricular ejection fraction of 39%. Which of the following drugs is the best choice for rate control in this patient?'
        1:
        'A 62-year-old woman presents for a regular check-up. She complains of lightheadedness and palpitations which occur episodically. Past medical history is significant for a myocardial infarction 6 months ago and NYHA class II chronic heart failure. She also was diagnosed with grade I arterial hypertension 4 years ago. Current medications are aspirin 81 mg, atorvastatin 10 mg, enalapril 10 mg, and metoprolol 200 mg daily. Her vital signs are a blood pressure of 135/90 mm Hg, a heart rate of 125/min, a respiratory rate of 14/min, and a temperature of 36.5°C (97.7°F). Cardiopulmonary examination is significant for irregular heart rhythm and decreased S1 intensity. ECG is obtained and is shown in the picture (see image). Echocardiography shows a left ventricular ejection fraction of 39%. Which of the following drugs is the best choice for rate control in this patient?'
        2:
        'A 62-year-old woman presents for a regular check-up. She complains of lightheadedness and palpitations which occur episodically. Past medical history is significant for a myocardial infarction 6 months ago and NYHA class II chronic heart failure. She also was diagnosed with grade I arterial hypertension 4 years ago. Current medications are aspirin 81 mg, atorvastatin 10 mg, enalapril 10 mg, and metoprolol 200 mg daily. Her vital signs are a blood pressure of 135/90 mm Hg, a heart rate of 125/min, a respiratory rate of 14/min, and a temperature of 36.5°C (97.7°F). Cardiopulmonary examination is significant for irregular heart rhythm and decreased S1 intensity. ECG is obtained and is shown in the picture (see image). Echocardiography shows a left ventricular ejection fraction of 39%. Which of the following drugs is the best choice for rate control in this patient?'
        3:
        'A 62-year-old woman presents for a regular check-up. She complains of lightheadedness and palpitations which occur episodically. Past medical history is significant for a myocardial infarction 6 months ago and NYHA class II chronic heart failure. She also was diagnosed with grade I arterial hypertension 4 years ago. Current medications are aspirin 81 mg, atorvastatin 10 mg, enalapril 10 mg, and metoprolol 200 mg daily. Her vital signs are a blood pressure of 135/90 mm Hg, a heart rate of 125/min, a respiratory rate of 14/min, and a temperature of 36.5°C (97.7°F). Cardiopulmonary examination is significant for irregular heart rhythm and decreased S1 intensity. ECG is obtained and is shown in the picture (see image). Echocardiography shows a left ventricular ejection fraction of 39%. Which of the following drugs is the best choice for rate control in this patient?'

        len():
        4

        endings: ['Atenolol', 'Diltiazem', 'Propafenone', 'Digoxin']
        example_id:'train-00008'
        label:3
       """


    def convert_examples_to_features(examples, label_list, max_seq_length,
                                     tokenizer,
                                     cls_token_at_end=False,
                                     cls_token='[CLS]',
                                     cls_token_segment_id=1,
                                     sep_token='[SEP]',
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     sep_token_extra=False,
                                     pad_token_segment_id=0,
                                     pad_on_left=False,
                                     pad_token=0,
                                     mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}        # {0: 0, 1: 1, 2: 2, 3: 3}

        features = []
        for ex_index, example in enumerate(tqdm(examples)):
            choices_features = []
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                tokens_a = tokenizer.tokenize(context)                                  # lista de palabras
                tokens_b = tokenizer.tokenize(example.question + " " + ending)          # ['amp', '##ici', '##llin']

                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

                # The convention in BERT is:
                # (a) For sequence pairs:
                #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
                # (b) For single sequences:
                #  tokens:   [CLS] the dog is hairy . [SEP]
                #  type_ids:   0   0   0   0  0     0   0
                #
                # Where "type_ids" are used to indicate whether this is the first
                # sequence or the second sequence. The embedding vectors for `type=0` and
                # `type=1` were learned during pre-training and are added to the wordpiece
                # embedding vector (and position vector). This is not *strictly* necessary
                # since the [SEP] token unambiguously separates the sequences, but it makes
                # it easier for the model to learn the concept of sequences.
                #
                # For classification tasks, the first vector (corresponding to [CLS]) is
                # used as as the "sentence vector". Note that this only makes sense because
                # the entire model is fine-tuned.
                tokens = tokens_a + [sep_token]                 # a la lista le suma [SEP] al final
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]

                segment_ids = [sequence_a_segment_id] * len(tokens)

                if tokens_b:
                    tokens += tokens_b + [sep_token]            
                    """
                    suma a la lista cada una de las palabras de tokens_b seguida de [SEP] en este 
                    caso son los diferentes tokens en los que se divide una sola respuesta.

                    'Ampicillin' -> 'amp', '##ici', '##llin'
                    """
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)            # añade cuatro veces 1 al final de la lista

                if cls_token_at_end:
                    tokens = tokens + [cls_token]
                    segment_ids = segment_ids + [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids

                input_ids = tokenizer.convert_tokens_to_ids(tokens)     # convierte los tokens específicos a un número identificativo de esa palabra en bert

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.

                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([cls_token, sep_token])  # [101, 102]
                # Por cada uno de los tokens obtenidos, si el token corresponde a un token especial le pone un 1 sino 0
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                # ESTE CÓDIGO NO SE VA A EJECUTAR, SE HACE EL PADDING POR LA DERECHA
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    output_mask = ([1] * padding_length) + output_mask

                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                else:
                    # EL PAD TOKEN ES 0
                    input_ids = input_ids + ([pad_token] * padding_length)
                    # LA MÁSCARA DE ENTRADA LLEVA PADDING CON 0
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    output_mask = output_mask + ([1] * padding_length)
                    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask))
            label = label_map[example.label]
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.bool)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    # try:
    #     tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer}.get(model_type)
    # except:
    #     tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer}.get(model_type)
    tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name)
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features(examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer,
                                            cls_token_at_end=bool(model_type in ['xlnet']),  # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta', 'albert']),
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
                                            sequence_b_segment_id=0 if model_type in ['roberta', 'albert'] else 1)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = convert_features_to_tensors(features)
    return (example_ids, all_label, *data_tensors)


def load_sparse_adj_data_with_contextnode(adj_pk_path, max_node_num, num_choice, args):
    cache_path = adj_pk_path +'.loaded_cache'
    use_cache = True

    if use_cache and not os.path.exists(cache_path):
        use_cache = False

    if use_cache:
        with open(cache_path, 'rb') as f:
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel = pickle.load(f)
    # ESTE ELSE NO SE EJECUTA
    else:
        with open(adj_pk_path, 'rb') as fin:
            adj_concept_pairs = pickle.load(fin)

        n_samples = len(adj_concept_pairs) #this is actually n_questions x n_choices
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
        concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long) #default 2: "other node"
        node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)

        adj_lengths_ori = adj_lengths.clone()
        for idx, _data in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
            adj, concepts, qm, am, cid2score = _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], _data['cid2score']
            #adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
            #concepts: np.array(num_nodes, ), where entry is concept id
            #qm: np.array(num_nodes, ), where entry is True/False
            #am: np.array(num_nodes, ), where entry is True/False
            assert len(concepts) == len(set(concepts))
            qam = qm | am
            #sanity check: should be T,..,T,F,F,..F
            assert qam[0] == True
            F_start = False
            for TF in qam:
                if TF == False:
                    F_start = True
                else:
                    assert F_start == False
            num_concept = min(len(concepts), max_node_num-1) + 1 #this is the final number of nodes including contextnode but excluding PAD
            adj_lengths_ori[idx] = len(concepts)
            adj_lengths[idx] = num_concept

            #Prepare nodes
            concepts = concepts[:num_concept-1]
            concept_ids[idx, 1:num_concept] = torch.tensor(concepts +1)  #To accomodate contextnode, original concept_ids incremented by 1
            concept_ids[idx, 0] = 0 #this is the "concept_id" for contextnode

            #Prepare node scores
            if (cid2score is not None):
                for _j_ in range(num_concept):
                    _cid = int(concept_ids[idx, _j_]) - 1
                    assert _cid in cid2score
                    node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

            #Prepare node types
            node_type_ids[idx, 0] = 3 #contextnode
            node_type_ids[idx, 1:num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept-1]] = 0
            node_type_ids[idx, 1:num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept-1]] = 1

            #Load adj
            ij = torch.tensor(adj.row, dtype=torch.int64) #(num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(adj.col, dtype=torch.int64)  #(num_matrix_entries, ), where each entry is coordinate
            n_node = adj.shape[1]
            half_n_rel = adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node

            #Prepare edges
            i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
            extra_i, extra_j, extra_k = [], [], []
            for _coord, q_tf in enumerate(qm):
                _new_coord = _coord + 1
                if _new_coord > num_concept:
                    break
                if q_tf:
                    extra_i.append(0) #rel from contextnode to question concept
                    extra_j.append(0) #contextnode coordinate
                    extra_k.append(_new_coord) #question concept coordinate
            for _coord, a_tf in enumerate(am):
                _new_coord = _coord + 1
                if _new_coord > num_concept:
                    break
                if a_tf:
                    extra_i.append(1) #rel from contextnode to answer concept
                    extra_j.append(0) #contextnode coordinate
                    extra_k.append(_new_coord) #answer concept coordinate

            half_n_rel += 2 #should be 19 now
            if len(extra_i) > 0:
                i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                k = torch.cat([k, torch.tensor(extra_k)], dim=0)
            ########################

            mask = (j < max_node_num) & (k < max_node_num)
            i, j, k = i[mask], j[mask], k[mask]
            i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
            edge_index.append(torch.stack([j,k], dim=0)) #each entry is [2, E]
            edge_type.append(i) #each entry is [E, ]

        with open(cache_path, 'wb') as f:
            pickle.dump([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel], f)


    ori_adj_mean  = adj_lengths_ori.float().mean().item()
    ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
    print('| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |'.format(ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item()) +
          ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
          ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                      (node_type_ids == 1).float().sum(1).mean().item()))

    edge_index = list(map(list, zip(*(iter(edge_index),) * num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
    edge_type = list(map(list, zip(*(iter(edge_type),) * num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[E, ]

    concept_ids, node_type_ids, node_scores, adj_lengths = [x.view(-1, num_choice, *x.size()[1:]) for x in (concept_ids, node_type_ids, node_scores, adj_lengths)]
    #concept_ids: (n_questions, num_choice, max_node_num)
    #node_type_ids: (n_questions, num_choice, max_node_num)
    #node_scores: (n_questions, num_choice, max_node_num)
    #adj_lengths: (n_questions,　num_choice)
    return concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_type) #, half_n_rel * 2 + 1

# ESTE ESTÁ PENDIENTE DE REVISAR
class MultiGPUSparseAdjDataBatchGenerator(object):
    def __init__(self, args, mode, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], adj_data=None):
        self.args = args
        self.mode = mode
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        # self.adj_empty = adj_empty.to(self.device1)
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        if self.mode=='train' and self.args.drop_partial_batch:
            print ('dropping partial batch')
            n = (n//bs) *bs
        elif self.mode=='train' and self.args.fill_partial_batch:
            print ('filling partial batch')
            remain = n % bs
            if remain > 0:
                extra = np.random.choice(self.indexes[:-remain], size=(bs-remain), replace=False)
                self.indexes = torch.cat([self.indexes, torch.tensor(extra)])
                n = self.indexes.size(0)
                assert n % bs == 0

        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]


            edge_index_all, edge_type_all = self.adj_data
            #edge_index_all: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
            #edge_type_all:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
            edge_index = self._to_device([edge_index_all[i] for i in batch_indexes], self.device1)
            edge_type  = self._to_device([edge_type_all[i] for i in batch_indexes], self.device1)

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1, edge_index, edge_type])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)

class CustomizedEmbedding(nn.Module):
    def __init__(self, concept_num, concept_in_dim, concept_out_dim, use_contextualized=False,
                 pretrained_concept_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            self.emb = nn.Embedding(concept_num, concept_in_dim)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = GELU()

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, 'cpt_transform'):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, 'cpt_transform'):
                # index (4,199)
                # self.emb (9958, 768)
                # self.cpt_transform(768,100)
                return self.activation(self.cpt_transform(self.emb(index) * self.scale))
            else:
                return self.emb(index) * self.scale


class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): dimensionality of GNN hidden states
        n_ntype (int): number of node types (e.g. 4)
        n_etype (int): number of edge relation types (e.g. 38)
    """
    def __init__(self, args, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype; self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        #For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2*emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        #For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))


    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, return_attention_weights=False):
        # x: [N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39]
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim]

        #Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype +1) #[E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype +1).to(edge_vec.device)
        self_edge_vec[:,self.n_etype] = 1

        head_type = node_type[edge_index[0]] #[E,] #head=src
        tail_type = node_type[edge_index[1]] #[E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype) #[E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype) #[E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1) #[E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype) #[N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1) #[N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0) #[E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0) #[E+N, ?]
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1)) #[E+N, emb_dim]

        #Add self loops to edge_index
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  #[2, E+N]

        x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings) #[N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out


    def message(self, edge_index, x_i, x_j, edge_attr): #i: tgt, j:src
        # print ("edge_attr.size()", edge_attr.size()) #[E, emb_dim]
        # print ("x_j.size()", x_j.size()) #[E, emb_dim]
        # print ("x_i.size()", x_i.size()) #[E, emb_dim]
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2*self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key   = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]


        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2) #[E, heads]
        src_node_index = edge_index[0] #[E,]
        alpha = softmax(scores, src_node_index) #[E, heads] #group by src side node
        self._alpha = alpha

        #adjust by outgoing degree of src
        E = edge_index.size(1)            #n_edges
        N = int(src_node_index.max()) + 1 #n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index] #[E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1) #[E, heads]

        out = msg * alpha.view(-1, self.head_count, 1) #[E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  #[E, emb_dim]

class QAGNN_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size,
                    dropout=0.1):
        super().__init__()
        assert input_size == output_size
        self.args = args
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.emb_node_type = nn.Linear(self.n_ntype, hidden_size//2)

        self.basis_f = 'sin' #['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:                              
            self.emb_score = nn.Linear(1, hidden_size//2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, hidden_size//2)
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)

        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype +1 + n_ntype *2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))


        self.k = k
        self.gnn_layers = nn.ModuleList([GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])


        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout


    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra):
        for _ in range(self.k):
            _X = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training = self.training)
        return _X


    def forward(self, H, A, node_type, node_score, cache_output=False):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        _batch_size, _n_nodes = node_type.size()

        #Embed type
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        #Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) #[1,1,dim/2]
            js = torch.pow(1.1, js) #[1,1,dim/2]
            B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]


        X = H
        edge_index, edge_type = A #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim]

        _X = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra)

        X = _X.view(node_type.size(0), node_type.size(1), -1) #[batch_size, n_node, dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return output

class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn

class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)

class QAGNN(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, sent_dim,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02):
        super().__init__()
        self.init_range = init_range

        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                               use_contextualized=False, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
        self.svec2nvec = nn.Linear(sent_dim, concept_dim)       # de 1024 a 100

        self.concept_dim = concept_dim

        self.activation = GELU()

        self.gnn = QAGNN_Message_Passing(args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                                        input_size=concept_dim, hidden_size=concept_dim, output_size=concept_dim, dropout=p_gnn)

        self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)

        self.fc = MLP(concept_dim + sent_dim + concept_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, sent_vecs, concept_ids, node_type_ids, node_scores, adj_lengths, adj, emb_data=None, cache_output=False):
        """
        sent_vecs: (batch_size, dim_sent)
        concept_ids: (batch_size, n_node)
        adj: edge_index, edge_type
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns: (batch_size, 1)
        """
        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1) #(batch_size, 1, dim_node)
        gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, emb_data) #(batch_size, n_node-1, dim_node)
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1)) #(batch_size, n_node, dim_node)


        #Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2) #[batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
        node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]


        gnn_output = self.gnn(gnn_input, adj, node_type_ids, node_scores)

        Z_vecs = gnn_output[:,0]   #(batch_size, dim_node)

        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) #1 means masked out

        mask = mask | (node_type_ids == 3) #pool over all KG nodes
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)

        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs, Z_vecs), 1))
        # pasa por la MLP
        logits = self.fc(concat)
        return logits, pool_attn

class MeanPoolLayer(nn.Module):
    """
    A layer that performs mean pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
            lengths = mask_or_lengths.float()
        else:
            mask, lengths = mask_or_lengths, (1 - mask_or_lengths.float()).sum(1)
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), 0.0)
        mean_pooled = masked_inputs.sum(1) / lengths.unsqueeze(-1)
        return mean_pooled

class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
        else:
            mask = mask_or_lengths
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = masked_inputs.max(1)[0]
        return max_pooled

def dropout_mask(x, sz, p: float):
    """
    Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)

class EmbeddingDropout(nn.Module):
    """
    Apply dropout with probabily `embed_p` to an embedding layer `emb`.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m

# ESTE CÓDIGO NO SE UTILIZA
class LSTMTextEncoder(nn.Module):
    pool_layer_classes = {'mean': MeanPoolLayer, 'max': MaxPoolLayer}

    def __init__(self, vocab_size=1, emb_size=300, hidden_size=300, output_size=300, num_layers=2, bidirectional=True,
                 emb_p=0.0, input_p=0.0, hidden_p=0.0, pretrained_emb_or_path=None, freeze_emb=True,
                 pool_function='max', output_hidden_states=False):
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.output_hidden_states = output_hidden_states
        assert not bidirectional or hidden_size % 2 == 0

        if pretrained_emb_or_path is not None:
            if isinstance(pretrained_emb_or_path, str):  # load pretrained embedding from a .npy file
                pretrained_emb_or_path = torch.tensor(np.load(pretrained_emb_or_path), dtype=torch.float)
            emb = nn.Embedding.from_pretrained(pretrained_emb_or_path, freeze=freeze_emb)
            emb_size = emb.weight.size(1)
        else:
            emb = nn.Embedding(vocab_size, emb_size)
        self.emb = EmbeddingDropout(emb, emb_p)
        self.rnns = nn.ModuleList([nn.LSTM(emb_size if l == 0 else hidden_size,
                                           (hidden_size if l != num_layers else output_size) // (2 if bidirectional else 1),
                                           1, bidirectional=bidirectional, batch_first=True) for l in range(num_layers)])
        self.pooler = self.pool_layer_classes[pool_function]()

        self.input_dropout = nn.Dropout(input_p)
        self.hidden_dropout = nn.ModuleList([RNNDropout(hidden_p) for _ in range(num_layers)])

    def forward(self, inputs, lengths):
        """
        inputs: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)

        returns: tensor of shape (batch_size, hidden_size)
        """
        assert (lengths > 0).all()
        batch_size, seq_len = inputs.size()
        hidden_states = self.input_dropout(self.emb(inputs))
        all_hidden_states = [hidden_states]
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dropout)):
            hidden_states = pack_padded_sequence(hidden_states, lengths, batch_first=True, enforce_sorted=False)
            hidden_states, _ = rnn(hidden_states)
            hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True, total_length=seq_len)
            all_hidden_states.append(hidden_states)
            if l != self.num_layers - 1:
                hidden_states = hid_dp(hidden_states)
        pooled = self.pooler(all_hidden_states[-1], lengths)
        assert len(all_hidden_states) == self.num_layers + 1
        outputs = (all_hidden_states[-1], pooled)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        return outputs

class TextEncoder(nn.Module):
    valid_model_types = set(MODEL_CLASS_TO_NAME.keys())

    def __init__(self, model_name, output_token_states=False, from_checkpoint=None, **kwargs):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.output_token_states = output_token_states              # False
        assert not self.output_token_states or self.model_type in ('bert', 'roberta', 'albert')

        if self.model_type in ('lstm',):
            self.module = LSTMTextEncoder(**kwargs, output_hidden_states=True)
            self.sent_dim = self.module.output_size
        else:
            model_class = AutoModel
            self.module = model_class.from_pretrained(model_name, output_hidden_states=True)
            if from_checkpoint is not None:             # None
                self.module = self.module.from_pretrained(from_checkpoint, output_hidden_states=True)
            if self.model_type in ('gpt',):             # bert
                self.module.resize_token_embeddings(get_gpt_token_num())
            self.sent_dim = self.module.config.n_embd if self.model_type in ('gpt',) else self.module.config.hidden_size

    def forward(self, *inputs, layer_id=-1):
        '''
        layer_id: only works for non-LSTM encoders
        output_token_states: if True, return hidden states of specific layer and attention masks
        '''

        if self.model_type in ('lstm',):  # lstm
            input_ids, lengths = inputs
            outputs = self.module(input_ids, lengths)
        elif self.model_type in ('gpt',):  # gpt
            input_ids, cls_token_ids, lm_labels = inputs  # lm_labels is not used
            outputs = self.module(input_ids)
        else:  # bert / xlnet / roberta
            input_ids, attention_mask, token_type_ids, output_mask = inputs
            outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]

        if self.model_type in ('lstm',):
            sent_vecs = outputs[1]
        elif self.model_type in ('gpt',):
            cls_token_ids = cls_token_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
            sent_vecs = hidden_states.gather(1, cls_token_ids).squeeze(1)
        elif self.model_type in ('xlnet',):
            sent_vecs = hidden_states[:, -1]
        elif self.model_type in ('albert',):
            if self.output_token_states:
                return hidden_states, output_mask
            sent_vecs = hidden_states[:, 0]
        else:  # bert / roberta
            if self.output_token_states:
                return hidden_states, output_mask
            sent_vecs = self.module.pooler(hidden_states)
        return sent_vecs, all_hidden_states


class LM_QAGNN(nn.Module):
    def __init__(self, args, model_name, k, n_ntype, n_etype,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.0, encoder_config={}):
        super().__init__()
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = QAGNN(args, k, n_ntype, n_etype, self.encoder.sent_dim,
                                        n_concept, concept_dim, concept_in_dim, n_attention_head,
                                        fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                                        pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                        init_range=init_range)


    def forward(self, *inputs, layer_id=-1, cache_output=False, detail=False):
        """
        sent_vecs: (batch_size, num_choice, d_sent)    -> (batch_size * num_choice, d_sent)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        adj_lengths: (batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )
        returns: (batch_size, 1)
        """

        # batch_size, number_choices
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        #Here, merge the batch dimension and the num_choice dimension
        edge_index_orig, edge_type_orig = inputs[-2:]
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]] + [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]] + [sum(x,[]) for x in inputs[-2:]]

        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]

        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        logits, attn = self.decoder(sent_vecs.to(node_type_ids.device),
                                    concept_ids,
                                    node_type_ids, node_scores, adj_lengths, adj,
                                    emb_data=None, cache_output=cache_output)
        logits = logits.view(bs, nc)
        if not detail:
            return logits, attn
        else:
            return logits, attn, concept_ids.view(bs, nc, -1), node_type_ids.view(bs, nc, -1), edge_index_orig, edge_type_orig
            #edge_index_orig: list of (batch_size, num_choice). each entry is torch.tensor(2, E)
            #edge_type_orig: list of (batch_size, num_choice). each entry is torch.tensor(E, )


    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        #edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        #edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        return edge_index, edge_type

class LM_QAGNN_DataLoader(object):

    def __init__(self, args, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, use_cache=True):
        super().__init__()
        self.args = args                            # todos los argumentos
        self.batch_size = batch_size                # 32
        self.eval_batch_size = eval_batch_size      # 2
        self.device0, self.device1 = device         # 'cpu' 'cpu'
        self.is_inhouse = is_inhouse                # False

        model_type = MODEL_NAME_TO_CLASS[model_name]
        print ('train_statement_path', train_statement_path)

        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)

        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
        print ('num_choice', num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path, max_node_num, num_choice, args)
        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num, num_choice, args)
  
        assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length)
            *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path, max_node_num, num_choice, args)
            assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

        # esto no se va a ejecutar en nuestro caso
        """ if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])
        """
        assert 0. < subsample <= 1.

        # esto no se va a ejecutar en nuestro caso
         
        """ if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data) 

            assert self.train_size() == n_train """  
           
    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'train', self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def train_eval(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_data=self.test_adj_data)


# A PARTIR DE AQUI NO HACE FALTA REVISARLO
def train(args):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,dev_acc,test_acc\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    cp_emb = [np.load(path) for path in args.ent_emb_paths]                     # ['data/ddb/ent_emb.npy']
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    # cp_emb lleva el tensor H y T que son iguales
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)                   # concept_num = 9958    concept_dim = 768
    print('| num_concepts: {} |'.format(concept_num))

    # try:
    if True:
        if torch.cuda.device_count() >= 2 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
        elif torch.cuda.device_count() == 1 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")
        dataset = LM_QAGNN_DataLoader(args, args.train_statements, args.train_adj,
                                               args.dev_statements, args.dev_adj,
                                               args.test_statements, args.test_adj,
                                               batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                               device=(device0, device1),
                                               model_name=args.encoder,
                                               max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                               is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                               subsample=args.subsample, use_cache=args.use_cache)


        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################
        print ('args.num_relation', args.num_relation)
        model = LM_QAGNN(args, args.encoder, k=args.k, n_ntype=4, n_etype=args.num_relation, n_concept=concept_num,
                                   concept_dim=args.gnn_dim,
                                   concept_in_dim=concept_dim,
                                   n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
                                   p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                                   pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                                   init_range=args.init_range,
                                   encoder_config={})
        if args.load_model_path:
            print (f'loading and initializing model from {args.load_model_path}')
            model_state_dict, old_args = torch.load(args.load_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(model_state_dict)

        model.encoder.to(device0)
        model.decoder.to(device1)


    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

    print('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    def compute_loss(logits, labels):
        if args.loss == 'margin_rank':
            num_choice = logits.size(1)
            flat_logits = logits.view(-1)
            correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
            correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
            wrong_logits = flat_logits[correct_mask == 0]
            y = wrong_logits.new_ones((wrong_logits.size(0),))
            loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
        elif args.loss == 'cross_entropy':
            loss = loss_func(logits, labels)
        return loss

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print()
    print('-' * 71)
    if args.fp16:
        print ('Using fp16 training')
        scaler = torch.cuda.amp.GradScaler()

    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    freeze_net(model.encoder)
    if True:
    # try:
        for epoch_id in range(args.n_epochs):
            if epoch_id == args.unfreeze_epoch:
                unfreeze_net(model.encoder)
            if epoch_id == args.refreeze_epoch:
                freeze_net(model.encoder)
            model.train()
            for qids, labels, *input_data in dataset.train():
                optimizer.zero_grad()
                bs = labels.size(0)
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    if args.fp16:
                        with torch.cuda.amp.autocast():
                            logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                            loss = compute_loss(logits, labels[a:b])
                    else:
                        logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                        loss = compute_loss(logits, labels[a:b])
                    loss = loss * (b - a) / bs
                    if args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    total_loss += loss.item()
                if args.max_grad_norm > 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_lr()[0], total_loss, ms_per_batch))
                    total_loss = 0
                    start_time = time.time()
                global_step += 1

            model.eval()
            dev_acc = evaluate_accuracy(dataset.dev(), model)
            save_test_preds = args.save_model
            if not save_test_preds:
                test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
            else:
                eval_set = dataset.test()
                total_acc = []
                count = 0
                preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
                with open(preds_path, 'w') as f_preds:
                    with torch.no_grad():
                        for qids, labels, *input_data in tqdm(eval_set):
                            count += 1
                            logits, _, concept_ids, node_type_ids, edge_index, edge_type = model(*input_data, detail=True)
                            predictions = logits.argmax(1) #[bsize, ]
                            preds_ranked = (-logits).argsort(1) #[bsize, n_choices]
                            for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in enumerate(zip(qids, labels, predictions, preds_ranked, concept_ids, node_type_ids, edge_index, edge_type)):
                                acc = int(pred.item()==label.item())
                                print ('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                                f_preds.flush()
                                total_acc.append(acc)
                test_acc = float(sum(total_acc))/len(total_acc)

            print('-' * 71)
            print('| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, global_step, dev_acc, test_acc))
            print('-' * 71)
            with open(log_path, 'a') as fout:
                fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
                best_dev_epoch = epoch_id
                if args.save_model:
                    torch.save([model.state_dict(), args], f"{model_path}.{epoch_id}")
                    # with open(model_path +".{}.log.txt".format(epoch_id), 'w') as f:
                    #     for p in model.named_parameters():
                    #         print (p, file=f)
                    print(f'model saved to {model_path}.{epoch_id}')
            else:
                if args.save_model:
                    torch.save([model.state_dict(), args], f"{model_path}.{epoch_id}")
                    # with open(model_path +".{}.log.txt".format(epoch_id), 'w') as f:
                    #     for p in model.named_parameters():
                    #         print (p, file=f)
                    print(f'model saved to {model_path}.{epoch_id}')
            model.train()
            start_time = time.time()
            if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break
    # except (KeyboardInterrupt, RuntimeError) as e:
    #     print(e)



def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    dataset = "medqa_usmle"
    
    ent_emb = 'ddb'

    args.dataset = dataset
    args.train_adj = f'data/{dataset}/graph/dev.graph.adj.pk' 
    args.dev_adj = f'data/{dataset}/graph/dev.graph.adj.pk'
    args.test_adj = f'data/{dataset}/graph/test.graph.adj.pk'
    args.train_statements = f'data/{dataset}/statement/dev.statement.jsonl'
    args.dev_statements = f'data/{dataset}/statement/dev.statement.jsonl'
    args.test_statements = f'data/{dataset}/statement/test.statement.jsonl'
    args.encoder = 'bert'
    args.ent_emb = ent_emb
    args.ent_emb_paths = [f'data/{ent_emb}/ent_emb.py']
    #args.save_model
    args.save_dir = f'saved_models'
    args.mode = f'eval_detail'
    args.load_model_path = f'saved_models/medqa_usmle_model_hf3.4.0.pt'

    # vamos a añadir los argumentos con parser.add_argument en lugar de args. 
    # ya que parece ser que luego sobreescribe args...

    """ parser.add_argument('--dataset', default=dataset, choices=['csqa', 'obqa', 'medqa_usmle'], help='dataset name')
    parser.add_argument('--train_statements', default=f'data/{dataset}/statement/train.statement.jsonl')
    parser.add_argument('--dev_statements', default=f'data/{dataset}/statement/dev.statement.jsonl')
    parser.add_argument('--test_statements', default=f'data/{dataset}/statement/test.statement.jsonl')
    parser.add_argument('--ent_emb', default=ent_emb, choices=['ddb', 'umls'], help='entity embedding name')
    parser.add_argument('--ent_emb_paths', default=[f'data/{ent_emb}/ent_emb.py'])
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--save_dir', default=f'saved_models')
    parser.add_argument('--mode', default=f'eval_detail')
    parser.add_argument('--load_model_path', default=f'saved_models/medqa_usmle_model_hf3.4.0.pt')
     """

    parser.add_argument('--mode', default='train', choices=['train', 'eval_detail'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/qagnn/', help='model output directory')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--load_model_path', default=None)

    # data
    parser.add_argument('--num_relation', default=38, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/{args.dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True, help='use cached data to accelerate data loading')

    # model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')

    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')


    # regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--fp16', default=False, type=bool_flag, help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--drop_partial_batch', default=False, type=bool_flag, help='')
    parser.add_argument('--fill_partial_batch', default=False, type=bool_flag, help='')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(k=1)
    args = parser.parse_args()
    args.fp16 = args.fp16 and (torch.__version__ >= '1.6.0')

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval_detail':
        # raise NotImplementedError
        pass
    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    main()
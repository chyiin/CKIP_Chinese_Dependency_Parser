import os
import json
import numpy as np
import argparse
import rpyc
import torch
from transformers import BertTokenizer
from BERT_encoder import BERT_Encoder1
from BERT_semantic import BERT_Semantic
from opencc import OpenCC
from mst import mst_parse

s2t = OpenCC('s2t') # 簡體 --> 繁體
t2s = OpenCC('t2s') # 繁體 --> 簡體

def tokenize_and_preserve_labels(tokenizer, subsent):
    
    tokenized_sentence, labels_idx, seq = [], [], [0]
    n, idx = 0, 0
    
    for word in subsent:

        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        n = n + n_subwords
        tokenized_sentence.extend(tokenized_word)
        labels_idx.extend([idx] * n_subwords)
        seq.append(n)
        idx = idx + 1
        
    return tokenized_sentence, torch.tensor(labels_idx), seq[:-1]

# CKIP Chinese Dependency Parser
class chinese_parser():

    def __init__(self, port):
        
        # load ckip tagger
        self.conn = rpyc.classic.connect('localhost', port=port)
        self.conn.execute('from ckiptagger import data_utils, construct_dictionary, WS, POS, NER')
        self.conn.execute('ws = WS("./data")')
        self.conn.execute('pos = POS("./data")')

        # load model
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.model = BERT_Encoder1(hidden_size=1024, pretrained="hfl/chinese-roberta-wwm-ext-large").cuda()
        self.model.load_state_dict(torch.load(f'demo_model/bert_encoder.pt')) 

        # semantic relation model
        self.ids_to_labels = {0: 'root', 1: 'nn', 2: 'conj', 3: 'cc', 4: 'nsubj', 5: 'dep', 6: 'punct', 7: 'lobj', 8: 'loc', 9: 'comod', 10: 'asp', 11: 'rcmod', 12: 'etc', 13: 'dobj', 14: 'cpm', 15: 'nummod', 16: 'clf', 17: 'assmod', 18: 'assm', 19: 'amod', 20: 'top', 21: 'attr', 22: 'advmod', 23: 'tmod', 24: 'neg', 25: 'prep', 26: 'pobj', 27: 'cop', 28: 'dvpmod', 29: 'dvpm', 30: 'lccomp', 31: 'plmod', 32: 'det', 33: 'pass', 34: 'ordmod', 35: 'pccomp', 36: 'range', 37: 'ccomp', 38: 'xsubj', 39: 'mmod', 40: 'prnmod', 41: 'rcomp', 42: 'vmod', 43: 'prtmod', 44: 'ba', 45: 'nsubjpass'}
        self.semantic_model = BERT_Semantic(loaded_model='demo_model/bert_encoder.pt', hidden_size=1024, num_labels=len(self.ids_to_labels)).cuda()
        self.semantic_model.load_state_dict(torch.load(f'demo_model/bert_semantic_encoder.pt')) 

    def parse(self, sent):

        ws_sent = self.conn.eval(f'ws(["{sent}"])')[0]
        t2s_sent = ['root'] + [t2s.convert(word) for word in ws_sent]
        pos_sent = ['root'] + list(self.conn.eval(f'pos([{ws_sent}])')[0])
        parse_input = ['root'] + list(ws_sent)

        input_token, input_idx, input_seqs = tokenize_and_preserve_labels(self.tokenizer, t2s_sent)
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(txt) for txt in input_token])
        attention_masks = torch.tensor([float(i != 0.0) for i in input_ids])
        
        self.model.eval()
        self.semantic_model.eval()

        with torch.no_grad():
            
            output = self.model(input_ids=input_ids.unsqueeze(0).cuda(), token_type_ids=None, attention_mask=attention_masks.unsqueeze(0).cuda())
            semantic_output = self.semantic_model(input_ids=input_ids.unsqueeze(0).cuda(), token_type_ids=None, attention_mask=attention_masks.unsqueeze(0).cuda())

            semantic_label_indices = np.argmax(semantic_output[1].to('cpu').numpy(), axis=2)
            seq_output = torch.index_select(output[1][0], 0, torch.tensor(input_seqs).cuda())
            seq_output = torch.index_select(seq_output, 1, torch.tensor(input_seqs).cuda())
            label_indices = mst_parse(torch.transpose(seq_output, 0, 1).fill_diagonal_(-100000).to('cpu').numpy(), one_root=True)

            final_predict = [int(input_idx[input_seqs[label_idx]].cpu()) for label_idx in label_indices]
            final_predict[0] = -1

            semantic_labels = [self.ids_to_labels[sem_idx] for sem_idx in semantic_label_indices[0]]
            semantic_predict = [semantic_labels[idx] for idx in input_seqs]

        parse = [(f'{final_predict[1:][i]} - {parse_input[final_predict[1:][i]]} {pos_sent[final_predict[1:][i]]}', f'{i+1} - {parse_input[i+1]} {pos_sent[i+1]}', semantic_predict[1:][i]) for i in range(len(ws_sent))]

        return parse

if __name__ == '__main__':
    
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    def parse_args():

        parser = argparse.ArgumentParser()
        parser.add_argument('--parse_sentence', type=str, default='今天天氣真好。')
        parser.add_argument('--port', type=int, default=3333)
        parser.add_argument('--gpu', type=int, default=0)
        args = parser.parse_args()

        return args

    args = parse_args()
    args = vars(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(args['gpu'])

    chinese_parser = chinese_parser(port=args['port'])    
    dependency_tree = chinese_parser.parse(args['parse_sentence'])
    saved_tree = '\n'.join(list(map(str, dependency_tree)))
    print()
    print('----------------------------------------------------------------------------------')
    print('------------------------------------- OUTPUT -------------------------------------')
    print('----------------------------------------------------------------------------------')
    print()
    print(saved_tree)
    print()
    print('----------------------------------------------------------------------------------')


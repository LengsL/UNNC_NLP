"""Offline synthetic training compatibility check, NOT research evaluation."""
import os
from pathlib import Path
import json
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data/.cache/matplotlib'))

import torch
from transformers import BertConfig, BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer, models
from sklearn.linear_model import LogisticRegression
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments


def main():
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(dir=ROOT / 'data') as tmp:
        tmp = Path(tmp)
        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'hello', 'hi',
                 'goodbye', 'bye', 'friend', 'there', 'now']
        (tmp / 'vocab.txt').write_text('\n'.join(vocab), encoding='utf-8')
        BertTokenizer(vocab_file=str(tmp / 'vocab.txt')).save_pretrained(tmp)
        BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
                             num_attention_heads=2, intermediate_size=64)).save_pretrained(tmp)
        encoder = models.Transformer(str(tmp), max_seq_length=16)
        body = SentenceTransformer(modules=[encoder, models.Pooling(32)], device='cuda')
        model = SetFitModel(model_body=body, model_head=LogisticRegression())
        ds = Dataset.from_dict({'text': ['hello friend', 'hi there', 'goodbye now', 'bye friend'],
                                'label': [0, 0, 1, 1]})
        args = TrainingArguments(output_dir=str(tmp / 'training'), batch_size=2, max_steps=1,
                                 num_epochs=1, num_iterations=1, report_to='none',
                                 save_strategy='no', show_progress_bar=False, max_length=16)
        trainer = Trainer(model=model, args=args, train_dataset=ds)
        trainer.train()
        pred = model.predict(['hello friend', 'bye friend'])
        assert len(pred) == 2
        torch.cuda.synchronize()
        report = {'status': 'PASS', 'device': str(model.model_body.device),
                  'training_steps': 1, 'prediction_count': len(pred),
                  'note': 'Synthetic random tiny BERT; no pretrained model download or research performance validation.'}
        (ROOT / 'results/setfit_smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report))


if __name__ == '__main__':
    main()

# EE 641 Homework 4: Attention and Transformers

Author: Kuntian Tang
Student ID: 4855814938

Three problems on attention: a hand-built encoder-decoder transformer for
three-digit addition (q1), position encodings and length extrapolation
(q2), and a decoder-only SMILES language model with a norm-placement /
learning-rate / warmup stability sweep (q3). The report is
`report/report.pdf`.

## Contents

```
generate_datasets.py          dataset generation (seed 641)
q1/
├── dataset.py                AdditionDataset
├── attention.py              scaled dot-product attention, causal mask, MultiHeadAttention
├── model.py                  post-norm EncoderLayer / DecoderLayer
├── train.py                  training (6 runs: {forward,reversed}_n{1000,2000,4000})
├── evaluate.py               test metrics, head ablation, greedy pruning, attention maps
├── test_interfaces.py
├── provided/
└── results/
    ├── training_log_*.json   six training logs
    ├── metrics.json
    └── visualizations/       val_exact_match, test_accuracy_vs_size,
                              cross_attention_*_n4000, encoder_self_attention_*_n4000,
                              pruning_curves
q2/
├── dataset.py                OffsetDataset
├── positional.py             sinusoidal table, shift rotation, rotary, shift-invariance check
├── train.py                  training (none, learned, sinusoidal, rotary)
├── evaluate.py               in-range / beyond / bucket accuracy, encoding checks, maps
├── test_interfaces.py
├── provided/
└── results/
    ├── training_log_*.json   four training logs
    ├── metrics.json
    └── visualizations/       bucket_accuracy, attention_maps
q3/
├── dataset.py                build_vocab, SmilesDataset
├── model.py                  causal mask, pre/post-norm Block, DecoderLM
├── train.py                  training with warmup, gradient-norm logging, checkpoints
├── sample.py                 batched temperature sampling
├── evaluate.py               sweep summary, well-formedness, ring span, sample quality
├── test_interfaces.py
├── requirements.txt          rdkit, pytest
├── provided/
└── results/
    ├── training_log_*.json   12 sweep runs + pre_lr0.001_w500_s4000
    ├── samples_pre_lr0.001_w500_s4000_T1.txt
    ├── metrics.json
    └── visualizations/       sweep_grid, validity_curve, failure_by_length,
                              failure_by_span, property_distributions, sample_grid
report/
├── report.tex
└── report.pdf
```

## Running

Standard commands from the assignment; nothing differs. q3 needs `rdkit`
(`pip install -r q3/requirements.txt`).

## Implementation notes

- q1 `train.py` logs the mean training loss over each 50-step window.
  `evaluate.py` also stores the mean attention maps under
  `metrics.json['attention']`.
- q2 `metrics.json` additionally stores `all_buckets` (the 8 buckets over
  0-127) used for the bucket plot. The first bucket contains positions
  7-15 only, because positions 0-6 have no target.
- q3 logs window means over 50 steps for loss and both gradient norms, plus
  the window maxima (`head_grad_norm_max`, `total_grad_norm_max`). The
  gradient norms are taken after backward and before clipping. Checkpoints are
  saved every 1000 steps. The final model is used for step 4000.
  `metrics.json['main']['examples']` holds example samples of each outcome.
- Training was run on an RTX 4060 Ti (q2, q3) and CPU (q1).

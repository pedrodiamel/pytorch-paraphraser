# Pytorch Paraphrase Generation

This project providers users the ability to do paraphrase generation for sentences through a clean and simple API.

## TODO

- [X] Training framework (train|visualization|test)
- [X] Evaluate methods
- [X] Module for test/api (UI Flask module)
- [ ] Manifold implementation
- [ ] Pip installable package
- [ ] Docker configurate packaging
- [ ] Readme and doc

## Datasets

- [para-nmt-5m](http://www.cs.cmu.edu/~jwieting/)
- [Quora]( https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)
- [snli](https://nlp.stanford.edu/projects/snli/)
- [PPDB](http://paraphrase.org/#/download)
- Semeval

## Models

- Attention NMT
- Embedded Representation
- Manifold

## Results 

| Server   | Name     | Dataset    | Bleu      | Description                          |
|---------:|---------:|:----------:|:---------:|:-------------------------------------|
| @Moto    | AttNMT   | CMDS       | 0.7163    | Train: DB(para-nmt-5m-short+cmd)     |
| @Moto    | AttNMT   | CMDS+G     | 0.1351    | Train: DB(para-nmt-5m-shoet+cmd)     |
| @Learn   | TRP+GRU  | CMDS+G     | 0.0047    | Train: DB(para-nmt-5m-shoet+cmd)     |




## Reference

- https://arxiv.org/pdf/1706.01847.pdf
- http://www.cs.cmu.edu/~jwieting/wieting2017Millions.pdf
- https://arxiv.org/pdf/1711.05732.pdf
- https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16353/16062
- https://arxiv.org/pdf/1808.04364.pdf
- https://arxiv.org/pdf/1803.01465.pdf
- http://alborz-geramifard.com/workshops/nips18-Conversational-AI/Papers/18convai-Neural%20Machine%20Translation.pdf
- https://arxiv.org/pdf/1901.02998.pdf
- https://arxiv.org/pdf/1804.06059.pdf

## Repos

- https://github.com/kefirski?tab=repositories
- https://github.com/jhclark/multeval
- https://github.com/vsuthichai/paraphraser
- https://github.com/jwieting/para-nmt-50m

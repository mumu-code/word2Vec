Using pytorch to implement word2vec algorithm Skip-gram Negative Sampling (SGNS), and refer paper [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546v1).
dataset http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
## Dependency
- python 3.6
- pytorch 0.4+

## Usage
Run `main.py`.

Initialize the dataset and model.



## Evaluate
Refer repository [eval-word-vectors](https://github.com/mfaruqui/eval-word-vectors).
Like this:
```
eval/wordsim.py vector.txt eval/data/EN-MTurk-287.txt
```
```
eval/wordsim.py vector.txt eval/data/EN-MC-30.txt
```







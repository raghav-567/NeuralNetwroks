# CharTransformer

A lightweight, from-scratch character-level Transformer model implemented in PyTorch, inspired by Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT). This project aims to build a deeper understanding of the Transformer architecture by re-implementing its components manually.

---

## 🔧 Features

- Character-level tokenizer
- Configurable Transformer with:s
  - Multi-head self-attention
  - Position-wise feedforward layers
  - Dropout regularization
- Simple training loop with evaluation and model checkpointing
- PyTorch-based implementation, GPU compatible
- Trains on custom `.txt` datasets


## 🚀 Model Variants

| Model        | Parameters | Description                         | Status      |
|--------------|------------|-------------------------------------|-------------|
| small_model  | ~1M        | Lightweight, fast, runs on Colab    | ✅ Completed |
| large_model  | ~10M       | Deeper, more accurate, GPU required | ⚠️ Experimental |

We provide both configurations for educational and scalability purposes.
Use `models/small_config.py` to train and test easily. The large model is currently not trained due to compute limits, but is fully defined in code.



## HyperParameters 

### for state level output(~ 9 million params)
- max_iterations = 5000
- batch_size = 64
- learning_rate = 3e-4
- block_size = 256
- n_embd = 384
- eval_iters = 200
- eval_interval = 500
- dropout = 0.2
- no_of_head = 6

### for running(~1 million params)
- max_iterations = 5000
- batch_size = 64
- learning_rate = 3e-4
- block_size = 64
- n_embd = 126
- eval_iters = 200
- eval_interval = 500
- dropout = 0.2
- no_of_head = 6



## Sample output 

- step 0: train loss 4.3771, val loss 4.3727
- step 500: train loss 2.3067, val loss 2.3222
- step 1000: train loss 2.0418, val loss 2.0874
- step 1500: train loss 1.8874, val loss 1.9742
- step 2000: train loss 1.7825, val loss 1.9081
- step 2500: train loss 1.7037, val loss 1.8452
- step 3000: train loss 1.6513, val loss 1.8091
- step 3500: train loss 1.6120, val loss 1.7799
- step 4000: train loss 1.5827, val loss 1.7455
- step 4500: train loss 1.5513, val loss 1.7210
- step 4999: train loss 1.5302, val loss 1.6983

## Sample text generated 

ROSTREY:
So falsones clows of royise alact,
That iman clamistion told cuerforting,
On ther. Go you know me hath sondss malost
The but wars, as in asul know it.

Shipper: the own my body unclared, cravaliffate
Shalt plail dive, thou tistroace
Good oe.

DUKE VORCOMN:
On I so aboin, why and to Bolackion,
Make the Laliriou thine-frisate, which wome
Shall experche the gourniur: in master it.

Nurse:
I bear my more fule.

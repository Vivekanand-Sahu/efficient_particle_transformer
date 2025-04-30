

### Particle Transformer (ParT)

The **Particle Transformer (ParT)** architecture is described in "[Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772)", which can serve as a general-purpose backbone for jet tagging and similar tasks in particle physics. It is a Transformer-based architecture, enhanced with pairwise particle interaction features that are incorporated in the multi-head attention as a bias before softmax. The ParT architecture outperforms the previous state-of-the-art, ParticleNet, by a large margin on various jet tagging benchmarks.

![arch](figures/arch.png)

## Attention Matrices: 
The attention matrix shows how much each token (or input unit) attends to others. In transformers, attention is often spread out, but in well-trained models, you often see sparse or focused attention patterns. This sparsity can reflect learned dependencies, such as how words relate in a sentence or how objects interact in an image. Visualizing attention helps interpret model behavior and optimize computational efficiency by focusing only on significant interactions.

![image](https://github.com/user-attachments/assets/17ec9a88-755f-44ad-92d6-50cf5c5e44e2)

## Linformer ParT (LinParT):
Standard attention has quadratic complexity with respect to input length, which becomes costly for long sequences. Linformer tackles this by using low-rank projections, reducing attention computation to linear complexity. This allows Transformers to scale to large input sizes, such as long documents or detailed sequences, while consuming less memory and compute. It's a key innovation for making Transformers practical in real-time or resource-constrained scenarios.

![image](https://github.com/user-attachments/assets/eaa4d26c-3a36-4973-9e37-77245f70da4c)

## Int-8 Quantization:
Int-8 quantization reduces model precision from 32-bit to 8-bit integers, enabling faster inference and smaller model size. This is especially helpful for deploying Transformers on mobile devices or edge hardware. By using quantization-aware training or calibration techniques, models retain most of their original accuracy while gaining significant performance benefits. It's a go-to technique for optimizing models without architectural changes.

![image](https://github.com/user-attachments/assets/3b156033-42f6-4a4d-a7e9-c08f6028419d)

## Flash Attention: 
FlashAttention is a faster implementation of scaled dot-product attention, reducing memory usage by minimizing redundant data movement between GPU and memory. It fuses operations like softmax and matrix multiplication, allowing much larger batch sizes and sequence lengths. This innovation is especially helpful for training large Transformers efficiently on GPUs, enabling higher throughput without hitting memory bottlenecks.

<img width="621" alt="image" src="https://github.com/user-attachments/assets/348801e0-f489-4246-b4d4-01faaec04596" />

### Model Profiling and Results:
Profiling a Transformer involves examining each layer to identify performance bottlenecks—whether it’s attention layers, feedforward networks, or embedding operations. Profiling reveals if the model is memory-bound (slowed by data access) or compute-bound (limited by arithmetic operations). This guides optimization strategies like kernel fusion, quantization, or memory layout changes, ensuring smooth deployment across various hardware platforms.


Base:

<img width="800" alt="image" src="https://github.com/user-attachments/assets/f14e7d8c-6b8a-476f-bad7-f0d5772e5b99" />



Quantized:

<img width="800" alt="image" src="https://github.com/user-attachments/assets/db28fe5a-6a95-4f87-b99d-a1778df99754" />


Base:

<img src="https://github.com/user-attachments/assets/972bae55-d1fb-485a-afe5-1b8fa5c43451" width="800"/>


Flash Attention:

<img src="https://github.com/user-attachments/assets/883bde3a-49e7-4f6f-91f8-88a7af383fe2" width="800"/>


Base:

<img src="https://github.com/user-attachments/assets/0e86e45e-d6e0-41ff-9317-2ac03bdbf1a0" width="800"/>


Optimized:

<img src="https://github.com/user-attachments/assets/eb0c83b4-97ec-425a-92c8-3d92a21db2a7" width="800"/>


Base:

<img src="https://github.com/user-attachments/assets/8c4b3c90-8c6e-442c-b76e-7c498dc72fa9" width="800"/>


Optimized:

<img src="https://github.com/user-attachments/assets/6119b1eb-7669-47c7-9269-5ac1cf0c123d" width="800"/>


Base:

<img src="https://github.com/user-attachments/assets/18c84a3d-8347-449c-b824-dc89b57a4115" width="800"/>


Optimized:

<img src="https://github.com/user-attachments/assets/707e7c7c-e0b2-45c5-9781-5bff2fc80d3c" width="800"/>

## Getting started

### Download the datasets

To download the JetClass/QuarkGluon/TopLandscape datasets:

```
./get_datasets.py [JetClass|QuarkGluon|TopLandscape] [-d DATA_DIR]
```

After download, the dataset paths will be updated in the `env.sh` file.

### Training

The ParT models are implemented in PyTorch and the training is based on the [weaver](https://github.com/hqucms/weaver-core) framework for dataset loading and transformation. To install `weaver`, run:

```python
pip install 'weaver-core>=0.4'
```

**To run the training on the JetClass dataset:**

```
./train_JetClass.sh [ParT|PN|PFN|PCNN] [kin|kinpid|full] ...
```

where the first argument is the model:

- ParT: [Particle Transformer](https://arxiv.org/abs/2202.03772)
- PN: [ParticleNet](https://arxiv.org/abs/1902.08570)
- PFN: [Particle Flow Network](https://arxiv.org/abs/1810.05165)
- PCNN: [P-CNN](https://arxiv.org/abs/1902.09914)

and the second argument is the input feature sets:

- [kin](data/JetClass/JetClass_kin.yaml): only kinematic inputs
- [kinpid](data/JetClass/JetClass_kinpid.yaml): kinematic inputs + particle identification
- [full](data/JetClass/JetClass_full.yaml) (_default_): kinematic inputs + particle identification + trajectory displacement

Additional arguments will be passed directly to the `weaver` command, such as `--batch-size`, `--start-lr`, `--gpus`, etc., and will override existing arguments in `train_JetClass.sh`.

**Multi-gpu support:**

- using PyTorch's DataParallel multi-gpu training:

```
./train_JetClass.sh ParT full --gpus 0,1,2,3 --batch-size [total_batch_size] ...
```

- using PyTorch's DistributedDataParallel:

```
DDP_NGPUS=4 ./train_JetClass.sh ParT full --batch-size [batch_size_per_gpu] ...
```

**To run the training on the QuarkGluon dataset:**

```
./train_QuarkGluon.sh [ParT|ParT-FineTune|PN|PN-FineTune|PFN|PCNN] [kin|kinpid|kinpidplus] ...
```

**To run the training on the TopLandscape dataset:**

```
./train_TopLandscape.sh [ParT|ParT-FineTune|PN|PN-FineTune|PFN|PCNN] [kin] ...
```

The argument `ParT-FineTune` or `PN-FineTune` will run the fine-tuning using [models pre-trained on the JetClass dataset](models/).



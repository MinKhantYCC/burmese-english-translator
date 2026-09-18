# Burmese--English Machine Translation with NLLB-200

This project explores **Burmese-to-English machine translation**, with
the primary focus on fine-tuning a quantized **NLLB-200 1.3B**
sequence-to-sequence model. The project uses Parameter-Efficient
Fine-Tuning (PEFT) with **LoRA** to adapt the model to a
Burmese--English parallel corpus while keeping GPU memory requirements
manageable.

A separate notebook, `mig_burmese_llm.ipynb`, contains an experimental
translation test using **MIG Burmese LLM**, a causal language model. It
is included for comparison and exploration, but **NLLB-200 is the main
model used in this project**.

## Project Structure

``` text
.
├── docs                              # Documents
├── nllb_200_1_3B_8bit.ipynb          # Main NLLB fine-tuning and evaluation notebook
├── mig_burmese_llm.ipynb             # Experimental causal-LM translation notebook
├── mig-model-fine-tuning.ipynb       # Experimental causal-LM translation notebook
├── Pipfile                           # Project dependencies
├── Pipfile.lock                      # Project dependencies
├── pyproject.toml
└── README.md
```

## Main Model: NLLB-200

The main experiment uses:

-   **Base checkpoint:** `Emilio407/nllb-200-1.3B-8bit`
-   **Architecture:** Sequence-to-sequence Transformer
-   **Task:** Burmese (`my_MM`) → English (`en_Latn`)
-   **Fine-tuning method:** PEFT with LoRA
-   **Quantization:** k-bit training with BitsAndBytes
-   **Evaluation metric:** SacreBLEU
-   **Training framework:** Hugging Face Transformers
-   **Training environment:** GPU/Google Colab

The tokenizer processes Burmese source sentences and English target
sentences with a maximum sequence length of 128 tokens.

## Dataset

The project uses the Hugging Face dataset:

[`kalixlouiis/Myanmar-English-general-text-translation`](https://huggingface.co/datasets/kalixlouiis/Myanmar-English-general-text-translation)

The complete dataset contains:

  Split          Samples
  ------------ ---------
  Training         9,616
  Validation       1,202
  Test             1,203

The notebook may use smaller subsets during development and debugging to
reduce execution time and GPU usage.

## LoRA Configuration

The quantized NLLB model is prepared for k-bit training before LoRA
adapters are attached.

``` text
Rank (r):          8
LoRA alpha:        16
LoRA dropout:      0.05
Bias:              none
Task type:         SEQ_2_SEQ_LM
Target modules:    k_proj, q_proj, v_proj, out_proj, fc1, fc2
```

This approach keeps the quantized base weights frozen while training a
relatively small set of adapter parameters.

## Training

The NLLB notebook uses `Seq2SeqTrainer` with generation enabled so that
translation quality can be evaluated during training.

Key settings in the current experiment include:

``` text
Learning rate:          2e-5
Train batch size:       8
Evaluation batch size:  8
Epochs:                 2
Weight decay:           0.01
Precision:              FP16
Evaluation strategy:    Per epoch
```

`DataCollatorForSeq2Seq` provides dynamic padding at batch time instead
of padding every sequence to the global maximum length.

## Evaluation

Translation quality is measured using **SacreBLEU**, a standardized
implementation of BLEU. Generated English translations are decoded and
compared against their reference translations using n-gram overlap.

The evaluation workflow compares:

1.  the original quantized NLLB-200 model as the baseline; and
2.  the LoRA fine-tuned NLLB-200 model.

Both models are evaluated on the same held-out test samples using the
same generation and metric pipeline.

A development run recorded the following results:

  Model                   SacreBLEU   Average Generated Length
  --------------------- ----------- --------------------------
  Base NLLB-200              6.4056                       19.3
  Fine-tuned NLLB-200        6.4576                       19.4

> **Note:** These values were produced using the small test subset
> configured in the notebook (10 samples). They should be treated as
> development results rather than a definitive assessment of model
> quality. Final experiments should evaluate both models on the full
> held-out test set or a sufficiently large fixed subset.

## Inference

The fine-tuned NLLB model can generate an English translation by setting
Burmese as the source language and forcing the English NLLB language
token during generation.

The notebook currently loads the fine-tuned checkpoint from:

`minkhantycc/nlbb`

Example input:

``` text
အိမ်မှာ ဝိုင်ဖိုင်ပျက်သွားလို့ လာကြည့်ပေးပါ
```

See `nllb_200_1_3B_8bit.ipynb` for the complete tokenization and
generation workflow.

## MIG Burmese LLM Experiment

`mig_burmese_llm.ipynb` explores `Ko-Yin-Maung/mig-burmese-llm` for the
same general Burmese-to-English translation task.

Unlike NLLB, MIG Burmese LLM is loaded with `AutoModelForCausalLM`.
Translation is therefore formulated as prompted causal text generation,
for example:

``` text
Translate to English: <Burmese sentence>
```

The notebook fine-tunes the model with the Hugging Face `Trainer` and
tests translation through autoregressive generation. This experiment is
secondary to the NLLB work and is included to explore the difference
between using a general causal language model and a sequence-to-sequence
model designed for translation-oriented tasks.

## Requirements

The notebooks use the following main libraries:

``` text
torch
transformers
datasets
evaluate
sacrebleu
accelerate
sentencepiece
tiktoken
peft
bitsandbytes
safetensors
trl
huggingface_hub
```

A CUDA-capable GPU is recommended. The NLLB experiment was designed to
operate within constrained GPU memory by combining quantization and
PEFT.

## Running the Project

1.  Create or activate the Python environment defined by the `Pipfile`.
2.  Install the required dependencies.
3.  Open `nllb_200_1_3B_8bit.ipynb`.
4.  Authenticate with Hugging Face when required.
5.  Load and preprocess the Burmese--English dataset.
6.  Configure the quantized NLLB model and LoRA adapters.
7.  Train the model.
8.  Evaluate the base and fine-tuned models using SacreBLEU.
9.  Run inference with the resulting fine-tuned checkpoint.

For the causal-model experiment, run `mig_burmese_llm.ipynb` separately.

## Future Work

Future experiments can use the complete training and test splits,
increase training duration, and tune LoRA rank, learning rate, batch
size, and generation parameters. Evaluation can also be expanded beyond
SacreBLEU with additional translation metrics and human evaluation,
particularly for semantic adequacy and fluency in Burmese--English
translation.

## Purpose

This repository is an experimental machine translation project intended
to investigate efficient adaptation of multilingual pretrained models
for Burmese--English translation under limited computational resources.

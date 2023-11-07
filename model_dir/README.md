---
pipeline_tag: text-generation
license: other
---

# 🦙 LLaMA-13B

LLaMA-13B is a base model for text generation with 13B parameters and a 1T token training corpus. It was built and released by the FAIR team at Meta AI alongside the paper "[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)". 

This model repo was converted to work with the transformers package. It is under a bespoke **non-commercial** license, please see the [LICENSE](https://huggingface.co/dfurman/llama-13b/blob/main/LICENSE) file for more details.

## Model Summary

- **Model Type:** Causal decoder-only.
- **Dataset:** The model was trained on 1T tokens using the following data sources: CCNet [67%], C4 [15%], GitHub [4.5%], Wikipedia [4.5%], Books [4.5%], ArXiv [2.5%], Stack Exchange[2%]. 
- **Language(s):** The Wikipedia and Books domains include data in the following languages: bg, ca, cs, da, de, en, es, fr, hr, hu, it, nl, pl, pt, ro, ru, sl, sr, sv, uk. 
- **License:** Bespoke non-commercial license, see [LICENSE](https://huggingface.co/dfurman/llama-13b/blob/main/LICENSE) file.
- **Model date:** LLaMA was trained between Dec 2022 and Feb 2023.

**Where to send inquiries about the model:**
Questions and comments about LLaMA can be sent via the [GitHub repository](https://github.com/facebookresearch/llama) of the project, by opening an issue.

## Intended use
**Primary intended uses:**
The primary use of LLaMA is research on large language models, including: exploring potential applications such as question answering, natural language understanding or reading comprehension, understanding capabilities and limitations of current language models, and developing techniques to improve those, evaluating and mitigating biases, risks, toxic and harmful content generations, and hallucinations.

**Primary intended users:**
The primary intended users of the model are researchers in natural language processing, machine learning and artificial intelligence.

**Out-of-scope use cases:**
LLaMA is a base model, also known as a foundation model. As such, it should not be used on downstream applications without further risk evaluation, mitigation, and additional fine-tuning. In particular, the model has not been trained with human feedback, and can thus generate toxic or offensive content, incorrect information or generally unhelpful answers.

## Factors
**Relevant factors:**
One of the most relevant factors for which model performance may vary is which language is used. Although 20 languages were included in the training data, most of the LLaMA dataset is made of English text, and the model is thus expected to perform better for English than other languages. Relatedly, it has been shown in previous studies that performance might vary for different dialects, which is likely also the case for LLaMA.

**Evaluation factors:**
As LLaMA is trained on data from the Web, it is expected that the model reflects biases from this source. The RAI datasets are thus used to measure biases exhibited by the model for gender, religion, race, sexual orientation, age, nationality, disability, physical appearance and socio-economic status. The toxicity of model generations is also measured, depending on the toxicity of the context used to prompt the model.

## Ethical considerations
**Data:**
The data used to train the model is collected from various sources, mostly from the Web. As such, it contains offensive, harmful and biased content. LLaMA is thus expected to exhibit such biases from the training data.

**Human life:**
The model is not intended to inform decisions about matters central to human life, and should not be used in such a way.

**Mitigations:**
The data was filtered from the Web based on its proximity to Wikipedia text and references. For this, the Kneser-Ney language model is used with a fastText linear classifier.

**Risks and harms:**
Risks and harms of large language models include the generation of harmful, offensive or biased content. These models are often prone to generating incorrect information, sometimes referred to as hallucinations. LLaMA is not expected to be an exception in this regard.

**Use cases:**
LLaMA is a foundational model, and as such, it should not be used for downstream applications without further investigation and mitigations of risks. These risks and potential fraught use cases include, but are not limited to: generation of misinformation and generation of harmful, biased or offensive content.

## How to Get Started with the Model

### Setup
```python
!pip install -q -U transformers accelerate torch
```
### GPU Inference in fp16

This requires a GPU with at least 26GB of VRAM.

### First, Load the Model

```python
import transformers
import torch

model_name = "dfurman/llama-13b"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
streamer = transformers.TextStreamer(tokenizer)

model = transformers.LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### Next, Run the Model

```python
prompt = "An increasing sequence: one,"

inputs = tokenizer(
    prompt,
    padding=True,
    truncation=True,
    return_tensors='pt',
    return_token_type_ids=False,
).to("cuda")

_ = model.generate(
    **inputs, 
    max_new_tokens=20,
    streamer=streamer,
)
```

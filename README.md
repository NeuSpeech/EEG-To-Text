# Correction on [(AAAI 2022) Open Vocabulary EEG-To-Text Decoding and Zero-shot sentiment classification](https://arxiv.org/abs/2112.02690)

**First of all, we are not pointing at others, we do this correction due to no offense, but a kind reminder of being careful of the string generation process. 
We repsect Mr. Wang ver much, and appreciate his great contribution in this area.**

After scrutilizing [the original code shared by Wang Zhenhailong](https://github.com/MikeWangWZHL/EEG-To-Text), we discovered that the eval method have an unintentional but very serious mistake in generating predicted strings, which is using teacher forcing implicitly. 

The code which reaches my concern is:


```python
seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)
logits = seq2seqLMoutput.logits # bs*seq_len*voc_sz
probs = logits[0].softmax(dim = 1)
values, predictions = probs.topk(1)
predictions = torch.squeeze(predictions)
predicted_string = tokenizer.decode(predictions) 
```

Therefore resulting in [predictions like below](https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/results/task1_task2_taskNRv2-BrainTranslator_skipstep1-all_generation_results-7_22.txt#L61):

![](https://img-blog.csdnimg.cn/39c3cad1650f41a3ba01948ac60700a4.png)


In addition, we noticed that some people are using it as code base which generates concerning results. We are not condemning these researchers, we just want to notice them and maybe we can do something together to resolve this problem. 

[BELT Bootstrapping Electroencephalography-to-Language Decoding and Zero-Shot SenTiment Classification by Natural Language Supervision](https://arxiv.org/pdf/2309.12056)
[Aligning Semantic in Brain and Language: A Curriculum Contrastive Method for Electroencephalography-to-Text Generation](https://ieeexplore.ieee.org/iel7/7333/4359219/10248031.pdf)
[UniCoRN: Unified Cognitive Signal ReconstructioN bridging cognitive signals and human language](https://arxiv.org/pdf/2307.05355)
[Semantic-aware Contrastive Learning for Electroencephalography-to-Text Generation with Curriculum Learning](https://arxiv.org/pdf/2301.09237)
[DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation](https://arxiv.org/pdf/2309.14030)

We have written a corrected version to use model.generate to evaluate the model, the result is not so good. 
Basicly, we changed the model_decoding.py and eval_decoding.py to add model.generate for its originally nn.Module class model, and used model.generate to predict strings.

**We are open to everyone to scrutinize on this corrected code and run the code. Then, we will show the final performance of this model in this repo and formalize a technical paper.**
# We really appreciate the great contribution made by Mr. Wang, however, we should prevent others from continuing this misunderstanding. 



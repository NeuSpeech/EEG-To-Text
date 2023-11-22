# Correction on [(AAAI 2022) Open Vocabulary EEG-To-Text Decoding and Zero-shot sentiment classification](https://arxiv.org/abs/2112.02690)

After scrutilizing the original code shared by Wang Zhenhailong, we discovered that the eval method have an unintentional but very serious mistake in generating predicted strings, which is using teacher forcing implicitly. 
The code which reaches my concern is:


```python
seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)
logits = seq2seqLMoutput.logits # bs*seq_len*voc_sz
probs = logits[0].softmax(dim = 1)
values, predictions = probs.topk(1)
predictions = torch.squeeze(predictions)
predicted_string = tokenizer.decode(predictions) 
```

Therefore resulting in predictions like below:

![在这里插入图片描述](https://img-blog.csdnimg.cn/39c3cad1650f41a3ba01948ac60700a4.png)


In addition, we noticed that some people are using it as code base which generates concerning results. We are not condemning these researchers, we just want to notice them and maybe we can do something together to resolve this problem. 

BELT Bootstrapping Electroencephalography-to-Language Decoding and Zero-Shot SenTiment Classification by Natural Language Supervision
Aligning Semantic in Brain and Language: A Curriculum Contrastive Method for Electroencephalography-to-Text Generation
UniCoRN: Unified Cognitive Signal ReconstructioN bridging cognitive signals and human language
Semantic-aware Contrastive Learning for Electroencephalography-to-Text Generation with Curriculum Learning
DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation

We have written a corrected version to use model.generate to evaluate the model, the result is not so good. 
Basicly, we changed the model_decoding.py and eval_decoding.py to add model.generate for its originally nn.Module class model, and used model.generate to predict strings.

**We are open to everyone to scrutinize on this corrected code and run the code. Then, we will show the final performance of this model in this repo and formalize a technical paper.**
# We really appreciate the great contribution made by Mr. Wang, however, we should prevent others from continuing this misunderstanding. 



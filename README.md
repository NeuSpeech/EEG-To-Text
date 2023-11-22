# correction on [(AAAI 2022) Open Vocabulary EEG-To-Text Decoding and Zero-shot sentiment classification](https://arxiv.org/abs/2112.02690)

After scrutilizing the original code shared by Wang Zhenhailong, we discovered that the eval method have an unintentional but very serious mistake in generating predicted strings, which is using teacher forcing implicitly. 
The code which reaches my concern is:

'''
seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)
logits = seq2seqLMoutput.logits # bs*seq_len*voc_sz
probs = logits[0].softmax(dim = 1)
values, predictions = probs.topk(1)
predictions = torch.squeeze(predictions)
predicted_string = tokenizer.decode(predictions) 
'''
Therefore resulting in predictions like below:

EEG-To-Text/results/task1_task2_taskNRv2-BrainTranslator_skipstep1-all_generation_results-7_22.txt at main · MikeWangWZHL/EEG-To-Text (github.com)
![image](https://github.com/NeuSpeech/EEG-To-Text/assets/151606332/7bfbd600-6591-4812-9e22-2b43a0855942)


In addition, we noticed that some people are using it as code base which generates concerning results. We are not condemning these researchers, we just want to notice them and maybe we can do something together to resolve this problem.

BELT Bootstrapping Electroencephalography-to-Language Decoding and Zero-Shot SenTiment Classification by Natural Language Supervision
Aligning Semantic in Brain and Language: A Curriculum Contrastive Method for Electroencephalography-to-Text Generation
UniCoRN: Unified Cognitive Signal ReconstructioN bridging cognitive signals and human language
Semantic-aware Contrastive Learning for Electroencephalography-to-Text Generation with Curriculum Learning
DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation

# We really appreciate the great contribution made by Mr. Wang, however, we should prevent others from continuing this misunderstanding. 




When I pass in pure noise with same input shape, the model will get the same high performance score.

I have written code to use model.generate to eval the model, the result is not so good. **We are open to everyone to scrutinize on this corrected code and run the code. Then, we will show the final performance of this model in this repo and formalize a technical paper.**


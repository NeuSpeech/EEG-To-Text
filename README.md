The **main branch** contains the final code for our "Are EEG-to-Text Models Working?" paper. 

Accepted by [IJCAI workshop 2024](https://github.com/user-attachments/files/16624318/IJCAI_hyejeongjo_poster_Final.pdf)

If you have any questions, you can write them in the Issues section or email Hyejeong Jo at girlsending0@khu.ac.kr.

check our new paper with full detailed comparison of different models on this task at [https://arxiv.org/abs/2405.06459](https://arxiv.org/abs/2405.06459)

overview
![image](https://github.com/NeuSpeech/EEG-To-Text/assets/151606332/57212488-b75f-44c7-a265-e2a51483e9f5)

performance
![image](https://github.com/NeuSpeech/EEG-To-Text/assets/151606332/df58870c-5277-4935-8c66-15efd58e9283)



# Correction on [(AAAI 2022) Open Vocabulary EEG-To-Text Decoding and Zero-shot sentiment classification](https://arxiv.org/abs/2112.02690)
# results and code is updated on **master** branch
# results and code is updated on **master** branch
# results and code is updated on **master** branch
**First of all, we are not pointing at others, we do this correction due to no offense, but a kind reminder of being careful of the string generation process. 
We repsect Mr. Wang very much, and appreciate his great contribution in this area.**

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

```
target string: It isn't that Stealing Harvard is a horrible movie -- if only it were that grand a failure!
predicted string:  was't a the. is was a bad place, it it it were a.. movie.
################################################


target string: It just doesn't have much else... especially in a moral sense.
predicted string:  was so't work the to to and not the country sense.
################################################


target string: Those unfamiliar with Mormon traditions may find The Singles Ward occasionally bewildering.
predicted string:  who with the history may be themselves Mormoning''s amusingering.
################################################


target string: Viewed as a comedy, a romance, a fairy tale, or a drama, there's nothing remotely triumphant about this motion picture.
predicted string:  the from a whole, it film, and comedy tale, and a tragic, it is nothing quite romantic about it. picture.
################################################


target string: But the talented cast alone will keep you watching, as will the fight scenes.
predicted string:  the most and of cannot not the entertained. and they the music against.
################################################


target string: It's solid and affecting and exactly as thought-provoking as it should be.
predicted string:  was a, it, it what it.provoking as it is be.
################################################


target string: Thanks largely to Williams, all the interesting developments are processed in 60 minutes -- the rest is just an overexposed waste of film.
predicted string:  to to the, the of films and in in in a minutes. and longest is a a afteragerposure, of time time
################################################


target string: Cantet perfectly captures the hotel lobbies, two-lane highways, and roadside cafes that permeate Vincent's days
predicted string: urtor was describes the spirit'sies and the ofstory streets, and the parking of areate the's life.</s>'sgggggggg,,,,,,,,,,,,,,</s>,,,,,
################################################


target string: An important movie, a reminder of the power of film to move us and to make us examine our values.
predicted string: nie part in " classic of the importance of the, shape people, our make us think our lives,
################################################


target string: Too much of this well-acted but dangerously slow thriller feels like a preamble to a bigger, more complicated story, one that never materializes.
predicted string:  bad of a is-known film not over- is like a film-ble to a much, more dramatic story. which that is endsizes.
```

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


This work was supported by the Culture, Sports and Tourism R&D Program through the Korea Creative Content Agency grant funded by the Ministry of Culture, Sports and Tourism (RS-2023-00226263), the Institute for Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2024-00509257, Global AI Frontier Lab), the Information Technology Research Center (ITRC) support program (IITP-2024-RS-2024-00438239) supervised by the IITP, and the IITP grant funded by the Korea government (MSIT) (No. RS-2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development, Kyung Hee University).

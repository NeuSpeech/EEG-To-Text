CUDA_VISIBLE_DEVICES=3 python3 eval_decoding.py \
    --checkpoint_path checkpoints/decoding/best/task1_task2_taskNRv2_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_noise.pt \
    --config_path config/decoding/task1_task2_taskNRv2_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_noise.json \
    --test_input EEG \
    --train_input noise \
    -cuda cuda:0

CUDA_VISIBLE_DEVICES=3 python3 eval_decoding.py \
    --checkpoint_path checkpoints/decoding/best/task1_task2_taskNRv2_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_noise.pt \
    --config_path config/decoding/task1_task2_taskNRv2_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_noise.json \
    --test_input noise \
    --train_input noise \
    -cuda cuda:0


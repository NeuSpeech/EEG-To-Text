CUDA_VISIBLE_DEVICES=0 python3 eval_decoding.py \
    --checkpoint_path checkpoints/decoding/best/task1_task2_task3_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.pt \
    --config_path config/decoding/task1_task2_task3_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.json \
    --test_input EEG \
    --train_input EEG \
    -cuda cuda:0

CUDA_VISIBLE_DEVICES=0 python3 eval_decoding.py \
    --checkpoint_path checkpoints/decoding/best/task1_task2_task3_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.pt \
    --config_path config/decoding/task1_task2_task3_finetune_T5Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.json \
    --test_input noise \
    --train_input EEG \
    -cuda cuda:0


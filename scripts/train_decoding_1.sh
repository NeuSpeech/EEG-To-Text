CUDA_VISIBLE_DEVICES=2,3 python3 train_decoding.py --model_name T5Translator \
    --task_name task1_task2_taskNRv2 \
    --one_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 20 \
    --num_epoch_step2 30 \
    --train_input EEG \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b 32 \
    -s ./checkpoints/decoding \

CUDA_VISIBLE_DEVICES=2,3 python3 train_decoding.py --model_name T5Translator \
    --task_name task1_task2_task3 \
    --one_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 20 \
    --num_epoch_step2 30 \
    --train_input EEG \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b 32 \
    -s ./checkpoints/decoding \

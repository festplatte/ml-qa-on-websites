# Add parent directory to python path to access lightning_base.py
# export PYTHONPATH="../":"${PYTHONPATH}"

python finetune.py \
--overwrite_output_dir \
--model_name_or_path $MODEL_PATH \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--gpus $N_GPUS \
--learning_rate=3e-5 \
--train_batch_size=$BATCH_SIZE \
--eval_batch_size=$BATCH_SIZE \
--max_source_length=2048 \
--max_target_length=56 \
--val_check_interval=0.1 --n_val=200 \
--do_train --do_predict \
 "$@"


# BATCH_SIZE=1 ./finetune_t5.sh --model_name_or_path ../../../mg-masterarbeit/data/T5-2048 --data_dir ../../../mg-masterarbeit/data/ms-marco/s2s_wellformed --output_dir ../../../mg-masterarbeit/data/T5-ms-marco --gpus 0 --overwrite_output_dir
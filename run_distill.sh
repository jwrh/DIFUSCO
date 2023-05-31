export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1,2

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u difusco/distill_train.py \
  --task "tsp_distill" \
  --wandb_logger_name "tsp50_first" \
  --diffusion_type "gaussian" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --storage_path "" \
  --lr_scheduler "cosine-decay" \
  --storage_path "/usr0/home/junweih/DIFUSCO" \
  --training_split "data/data/tsp/tsp50_train_concorde.txt" \
  --validation_split "data/data/tsp/tsp50_valid_concorde.txt" \
  --test_split "data/data/tsp/tsp50_test_concorde.txt" \
  --batch_size 50 \
  --num_epochs 50 \
  --diffusion_steps 1024 \
  --validation_examples 8 \
  --inference_schedule "linear" \
  --inference_diffusion_steps 2 \
  --skip 2 \
  --fp16 \
  --who_eval "teacher" \
  --ckpt_path "/usr0/home/junweih/DIFUSCO/models/tsp50_first/bl5y2e00/checkpoints/last.ckpt"
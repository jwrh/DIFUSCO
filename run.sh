export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_gaussian_tsp50" \
  --diffusion_type "gaussian" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --storage_path "" \
  --lr_scheduler "cosine-decay" \
  --storage_path "/usr0/home/junweih/DIFUSCO" \
  --training_split "data/data/tsp/tsp50_train_concorde.txt" \
  --validation_split "data/data/tsp/tsp50_valid_concorde.txt" \
  --test_split "data/data/tsp/tsp50_test_concorde.txt" \
  --batch_size 32 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --fp16
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0q

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u difusco/distill_train.py \
  --task "tsp_distill" \
  --wandb_logger_name "tsp_diffusion_graph_gaussian_tsp50_distill" \
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
  --batch_size 20 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --inference_trick "ddim" \
  --ckpt_path "models/tsp_diffusion_graph_gaussian_tsp50/u463siy5/checkpoints/last.ckpt" \
  --fp16
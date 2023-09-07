export CUDA_VISIBLE_DEVICES=1

cd ..

python cont_assoc/evaluate_4dpanoptic.py --ckpt_ps ../ckpt/panoptic_pq_564.pth --ckpt_ag ../ckpt/aggregation_aq_724.ckpt

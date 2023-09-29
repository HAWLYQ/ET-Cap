# train captioner

# trajectory-aware captioner: 
#    ../configs/single_captioner/captioner_timeglobal_meanview_detrinit.py
# trajectory-aware captioner w/o detr init: 
#   ../configs/single_captioner/captioner_timeglobal_meanview.py
# trajectory-aware captioner w/o detr init, w/o region-level cross-att: 
#   ../configs/single_captioner/captioner_timeglobal.py
# trajectory-aware captioner w/o detr init, w/o trajectory-level cross-att: 
#   ../configs/single_captioner/captioner_meanview.py
# captioner w/ only end view: 
#   ../configs/single_captioner/captioner_endview.py

#e.g. train trajectory-aware captioner
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port 1400 embodied_trajcap_train.py \
../configs/single_captioner/captioner_timeglobal_meanview_detrinit.py \
--work-dir ./work_dirs/captioner_detrinit-tune-backbone_timeglobal_meanview_de2layer_lr3e5-liner-anneal_epoch20 \
--launcher pytorch
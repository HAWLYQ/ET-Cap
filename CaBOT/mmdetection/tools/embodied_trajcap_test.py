# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.append('../')
# print(sys.path)
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test, embodied_multi_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
# from mmdet.models import build_detector
from mmdet.models import build_navigator
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    # anwen hu 2022/9/16
    parser.add_argument(
        '--eval_set',
        type=str,
        help='evaluate which set')
    parser.add_argument(
        '--calculate_metrics_with_saved_result',
        type=str,
        default='False',
        choices=['True', 'False'],
        help='evaluate which saved result')
    parser.add_argument(
        '--path_len_weight_eval',
        type=str,
        default='False',
        choices=['True','False'],
        help='evaluate which path len as weight')
    parser.add_argument(
        '--use_val_best_checkpoint',
        type=str,
        default='False',
        choices=['True', 'False'],
        help='whether to load the checkpoint performing best in val set')
        
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    """assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')"""

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)


    test_dataloader_default_args = dict(
        samples_per_gpu=8, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    cfg.data.test.ann_file = args.eval_set
    print('test on ', cfg.data.test.ann_file)
    cfg.data.test.pred_result_save_dir = '/'.join(args.checkpoint.split('/')[:-1])

    if args.path_len_weight_eval=='True':
        cfg.data.test.path_len_weight_eval=True

    dataset = build_dataset(cfg.data.test)
    print('calculate_metrics_with_saved_result:', args.calculate_metrics_with_saved_result)
    if args.calculate_metrics_with_saved_result == 'True':
        if rank == 0:
            pred_result_save_name = dataset.ann_file.split('/')[-1].replace('.json', '_pred.json')
            pred_result_save_path = dataset.pred_result_save_dir + '/'+ pred_result_save_name
            if not os.path.exists(pred_result_save_path):
                if 'navicaption_v1_test.json' in args.eval_set:
                    sub_pred_save_names = [
                    'navicaption_v1_test_common_pred.json',
                    'navicaption_v1_test_novel_instance_pred.json',
                    'navicaption_v1_test_novel_category_pred.json'
                    ]
                elif 'navigation_v1_trajcapinfer_test.json' in args.eval_set:
                    sub_pred_save_names = [
                    'navigation_v1_trajcapinfer_test_common_pred.json',
                    'navigation_v1_trajcapinfer_test_novel_instance_pred.json',
                    'navigation_v1_trajcapinfer_test_novel_category_pred.json'
                    ]
                elif 'navigation_v1_trajcapinfer_earlystop_test.json' in args.eval_set:
                    sub_pred_save_names = [
                    'navigation_v1_trajcapinfer_earlystop_test_common_pred.json',
                    'navigation_v1_trajcapinfer_earlystop_test_novel_instance_pred.json',
                    'navigation_v1_trajcapinfer_earlystop_test_novel_category_pred.json'
                    ]
                elif 'navicaption_v1_val.json' in args.eval_set:
                    sub_pred_save_names = [
                    'navicaption_v1_val_common_pred.json',
                    'navicaption_v1_val_novel_instance_pred.json',
                    'navicaption_v1_val_novel_category_pred.json'
                    ]
                elif 'navigation_v1_trajcapinfer_val.json' in args.eval_set:
                    sub_pred_save_names = [
                    'navigation_v1_trajcapinfer_val_common_pred.json',
                    'navigation_v1_trajcapinfer_val_novel_instance_pred.json',
                    'navigation_v1_trajcapinfer_val_novel_category_pred.json'
                    ]
                elif 'navigation_v1_trajcapinfer_earlystop_val.json' in args.eval_set:
                    sub_pred_save_names = [
                    'navigation_v1_trajcapinfer_earlystop_val_common_pred.json',
                    'navigation_v1_trajcapinfer_earlystop_val_novel_instance_pred.json',
                    'navigation_v1_trajcapinfer_earlystop_val_novel_category_pred.json'
                    ]
                test_pred_data = []
                for sub_pred_save_name in sub_pred_save_names:
                    sub_pred_save_path = dataset.pred_result_save_dir + '/'+ sub_pred_save_name
                    if not os.path.exists(sub_pred_save_path):
                        print(sub_pred_save_path, " doesn't exists, please inference first")
                        exit(0)
                    test_pred_data += json.load(open(sub_pred_save_path, 'r'))
                json.dump(test_pred_data, open(pred_result_save_path, 'w', encoding='utf-8'))
                print('save %d pred captions to %s' % (len(test_pred_data), pred_result_save_path))
                
            result = json.load(open(pred_result_save_path, 'r'))
            if 'earlystop' not in args.eval_set:
                dataset.pred_result_save_dir = None
            metric = dataset.evaluate(result)
            print('calculating metrics with result saved on ', pred_result_save_path)
            print(metric)
            
        exit(0)
    # print(len(dataset))
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    cfg.model.test_cfg = None
    model = build_navigator(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    if args.use_val_best_checkpoint:
        work_dir = '/'.join(args.checkpoint.split('/')[:-1])
        train_logs = [filename for filename in os.listdir(work_dir) if '.log.json' in filename]
        train_log = sorted(train_logs)[-1]
        log_file = open(work_dir+'/'+train_log, 'r')
        best_epoch = 0
        metric = 'CIDEr'
        best_result = 0
        for line in log_file.readlines():
            line = json.loads(line)
            if 'mode' not in line:
                continue
            if line['mode']=='val' and line[metric]>best_result:
                best_epoch = line['epoch']
                best_result = line[metric]
        args.checkpoint = work_dir+'/epoch_'+str(best_epoch)+'.pth'
        print('load checkpoint by val performance:', args.checkpoint)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        """outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False))"""
        outputs = embodied_multi_gpu_test(
            model, data_loader, args.tmpdir, False)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(args.checkpoint)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()

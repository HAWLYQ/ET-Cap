import sys
sys.path.append('../')
import math
import json
import os
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from tqdm import tqdm

DATASET_DIR='/data5/haw/ETCAP/'

MAX_NAVI_LEN=13


def transfer_navi_result_for_trajcap_dataset(split, subset, navi_model_dir):
    if subset == 'all':
        subset_filenames = ['navigation_v1_trajcapinfer_'+split+'_common.json',
                            'navigation_v1_trajcapinfer_'+split+'_novel_instance.json',
                            'navigation_v1_trajcapinfer_'+split+'_novel_category.json']
        all_data = []
        for subset_filename in subset_filenames:
            subset_data = json.load(open(navi_model_dir+subset_filename, 'r', encoding='utf-8'))
            all_data += subset_data
        save_path = navi_model_dir + 'navigation_v1_trajcapinfer_'+split+'.json'
        json.dump(all_data, open(save_path, 'w', encoding='utf-8'))
        print('save %d scene data to %s' % (len(all_data), save_path))
        return

    navicap_path = DATASET_DIR + 'anno/navicaption_v1_'+split+'_'+subset+'.json'
    navicap_data = json.load(open(navicap_path, 'r', encoding='utf-8'))
    ## start imgs are not stored in pred files, so fetch start imgs first
    scene_start2start_img = {}
    for scene in navicap_data:
        scene_id = scene['scene_id']
        for navi in scene['navigation_data']:
            start_pos_id = navi['pathid'].split('_')[1]
            pathid = scene_id + '_' +str(start_pos_id)
            start_img = DATASET_DIR + 'scenes/'+scene_id+navi['render_dir']+navi['shorter_images'][0]
            assert os.path.exists(start_img)
            scene_start2start_img[pathid] = start_img

    navi_result_path = navi_model_dir+'navigation_v1_'+split+'_'+subset+'_online-pred.json'
    navi_results = json.load(open(navi_result_path, 'r', encoding='utf-8'))
    scene_start2pred_imgs = {}
    for navi_result in navi_results:
        scene_startpos = navi_result['scene_id'] + '_'+ str(navi_result['start_pos_id'])
        if scene_startpos not in scene_start2start_img:
            print('scene_startpos:', scene_startpos, ' in navigation data but bot in navicaption data')
            continue
        pred_imgs = []
        # render_dir = '/'.join(navi_result['preds'][1]['img_path'].split('/')[:-1])+'/'
        for i, step_pred in enumerate(navi_result['preds']):
            assert i == step_pred['step']
            if i == 0:
                img = scene_start2start_img[scene_startpos]
            else:
                img = step_pred['img_path']
            assert os.path.exists(img)
            pred_imgs.append(img)

        scene_start2pred_imgs[scene_startpos] = {'images':pred_imgs, 'render_dir':''}
    
    
    new_data = []
    pathids = []
    for scene in navicap_data:
        scene_id = scene['scene_id']
        scene_captions = scene['scene_captions']
        assert len(scene_captions)==3
        navigation_data = []
        for navi in scene['navigation_data']:
            start_pos_id = navi['pathid'].split('_')[1]
            pathid = scene_id + '_' +str(start_pos_id)
            gt_path_len = len(navi['shorter_positions'])
            if pathid in pathids:
                print('repeat pathid:', pathid)
            pathids.append(pathid)
            images = scene_start2pred_imgs[pathid]['images']
            render_dir = scene_start2pred_imgs[pathid]['render_dir']
            captions = navi['final_view_captions']
            navigation_data.append({
                                'pathid':pathid,
                                'images':images,
                                'render_dir':render_dir,
                                'gt_path_len':gt_path_len})
            # scene_captions = scene_captions.union(set(captions))
        new_data.append({'scene_id':scene_id,
                        'navigation_data':navigation_data,
                        'gt_captions':scene_captions})
    save_path = navi_model_dir + 'navigation_v1_trajcapinfer_'+split+'_'+subset+'.json'
    json.dump(new_data, open(save_path, 'w', encoding='utf-8'))
    print('save %d inference data to %s' % (len(pathids), save_path))


def transfer_rule_navi_result_for_trajcap_dataset(split, subset, navi_model_dir):
    if subset == 'all':
        subset_filenames = ['navigation_v1_trajcapinfer_rule_'+split+'_common.json',
                            'navigation_v1_trajcapinfer_rule_'+split+'_novel_instance.json',
                            'navigation_v1_trajcapinfer_rule_'+split+'_novel_category.json']
        all_data = []
        for subset_filename in subset_filenames:
            subset_data = json.load(open(navi_model_dir+subset_filename, 'r', encoding='utf-8'))
            all_data += subset_data
        save_path = navi_model_dir + 'navigation_v1_trajcapinfer_rule_'+split+'.json'
        json.dump(all_data, open(save_path, 'w', encoding='utf-8'))
        print('save %d scene data to %s' % (len(all_data), save_path))
        return

    navicap_path = DATASET_DIR + 'navicaption_v1_'+split+'_'+subset+'.json'
    navicap_data = json.load(open(navicap_path, 'r', encoding='utf-8'))
    ## start imgs are stored in rule-based nav pred files
    scene_start2start_img = {}
    for scene in navicap_data:
        scene_id = scene['scene_id']
        for navi in scene['navigation_data']:
            start_pos_id = navi['pathid'].split('_')[1]
            pathid = scene_id + '_' +str(start_pos_id)
            start_img = DATASET_DIR + 'scenes/'+scene_id+navi['render_dir']+navi['shorter_images'][0]
            assert os.path.exists(start_img)
            scene_start2start_img[pathid] = start_img

    navi_result_path = navi_model_dir+'navigation_v1_'+split+'_'+subset+'_online-rule-pred.json'
    navi_results = json.load(open(navi_result_path, 'r', encoding='utf-8'))
    scene_start2pred_imgs = {}
    for navi_result in navi_results:
        scene_startpos = navi_result['scene_id'] + '_'+ str(navi_result['start_pos_id'])
        if scene_startpos not in scene_start2start_img:
            print('scene_startpos:', scene_startpos, ' in navigation data but bot in navicaption data')
            continue
        pred_imgs = []
        # render_dir = '/'.join(navi_result['preds'][1]['img_path'].split('/')[:-1])+'/'
        for i, step_pred in enumerate(navi_result['preds'][:MAX_NAVI_LEN]):
            assert i == step_pred['step']
            img = step_pred['img_path']
            assert os.path.exists(img)
            pred_imgs.append(img)

        scene_start2pred_imgs[scene_startpos] = {'images':pred_imgs, 'render_dir':''}
    
    
    new_data = []
    pathids = []
    for scene in navicap_data:
        scene_id = scene['scene_id']
        scene_captions = scene['scene_captions']
        assert len(scene_captions)==3
        navigation_data = []
        for navi in scene['navigation_data']:
            start_pos_id = navi['pathid'].split('_')[1]
            pathid = scene_id + '_' +str(start_pos_id)
           
            if pathid in pathids:
                print('repeat pathid:', pathid)
            pathids.append(pathid)
            images = scene_start2pred_imgs[pathid]['images']
            render_dir = scene_start2pred_imgs[pathid]['render_dir']
            captions = navi['final_view_captions']
            navigation_data.append({
                                'pathid':pathid,
                                'images':images,
                                'render_dir':render_dir})
            # scene_captions = scene_captions.union(set(captions))
        new_data.append({'scene_id':scene_id,
                        'navigation_data':navigation_data,
                        'gt_captions':scene_captions})
    save_path = navi_model_dir + 'navigation_v1_trajcapinfer_rule_'+split+'_'+subset+'.json'
    json.dump(new_data, open(save_path, 'w', encoding='utf-8'))
    print('save %d inference data to %s' % (len(pathids), save_path))


if __name__ == '__main__':
    navi_model_dir='./work_dirs/single_navigator_waction_region2layer_time1layer_de2layer_lr1e4_epoch10/'
    for subset in ['common', 'novel_instance', 'novel_category', 'all']:
        transfer_navi_result_for_trajcap_dataset(split='val', # val/test
                                                subset=subset, 
                                                navi_model_dir=navi_model_dir)

    
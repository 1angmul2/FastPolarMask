import pycocotools.coco as cocoapi
import sys
import cv2
import numpy as np
import pickle
import json
from tqdm import tqdm
import random
import os
from collections import defaultdict, OrderedDict

# cls_dict = OrderedDict()
CLASSES = ('__bg', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def sample_(range_, num):
    pos_idxs = np.array(random.sample(list(range(range_)), k=int(num)), dtype=np.int64)
    mask = np.ones(range_, dtype=np.int64)
    mask[pos_idxs] = 0

    neg_idxs = np.nonzero(mask)[0]

    return np.sort(pos_idxs), np.sort(neg_idxs)


def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.int32)
    return bbox


def start_transform(SPLITS, mode):
    for split in SPLITS:
        data_infos = [dict(label_idx=i, name=cls, count=0, img_ids=[]) for i, cls in enumerate(CLASSES)]

        coco = cocoapi.COCO(ANN_PATH.format(split))
        cat_ids = coco.getCatIds()
        cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(cat_ids)
        }
        img_ids = coco.getImgIds()
        num_images = len(img_ids)

        print('num_images', num_images)
        img_without_anns = defaultdict(int)
        def get_cls_single(coco, img_id, img_without_anns):
            ann_ids = coco.getAnnIds(img_id)
            anns = coco.loadAnns(ann_ids)
            if len(anns) == 0:
                img_without_anns[img_id] = 0
            ann_cat = [ann['category_id'] for ann in anns]
            for cat in ann_cat:
                data_infos[cat2label[cat]]['count'] += 1
                data_infos[cat2label[cat]]['img_ids'].append(img_id)
            return list(map(lambda x: cat2label[x], list(set(ann_cat))))

        for i, img_id in enumerate(tqdm(img_ids)):
            cls_ids_set = get_cls_single(coco, img_id, img_without_anns)

        for data_info in data_infos:
            data_info['img_ids'] = list(set(data_info['img_ids']))

        data_info_count_sort = sorted(data_infos, key=lambda x: x['count'])
        unique_ids = defaultdict(int)
        print("过滤重复")
        for data_info in tqdm(data_info_count_sort):
            temp = [ids for ids in data_info['img_ids'] if ids not in unique_ids]
            for ids in temp:
                unique_ids[ids] = 0
            data_info['img_ids'] = temp
            data_info['filter_count'] = len(temp)

        select_img_ids = defaultdict(int)
        if 'fraction' in mode:
            select_num = int(num_images * float(mode[8:]))
        else:  # number
            select_num = int(float(mode[6:]))
        select_num_per_cls = int(select_num / (len(CLASSES) - 1))

        for data in data_info_count_sort:
            if data['name'] == '__bg':
                continue
            if data['filter_count'] <= select_num_per_cls:
                temp_ids = data['img_ids']
            else:
                temp_ids = random.sample(data['img_ids'], select_num_per_cls)
            for i in temp_ids:
                select_img_ids[i] = 0

        if len(select_img_ids) < select_num:
            add_select = [ids for ids in img_ids
                          if ids not in select_img_ids and ids not in img_without_anns]
            random_add_select = random.sample(add_select, select_num-len(select_img_ids))
            for i in random_add_select:
                select_img_ids[i] = 0

        select_ids = list(select_img_ids.keys())
        select_img_infos = coco.loadImgs(select_ids)

        write_json = {'images': [],
                      'categories': [],
                      'annotations': []}
        imgs = []
        annotations = []

        print("开始生成新的标注文件")
        ann_idx = 1
        for true_img_idx, info in enumerate(tqdm(select_img_infos)):
            img = info.copy()
            img['id'] = true_img_idx + 1
            imgs.append(img)
            ann_ids = coco.getAnnIds(imgIds=info['id'])
            anns = coco.loadAnns(ann_ids)  # list
            anns_ = []
            for ann in anns:
                ann_ = ann.copy()
                ann_['image_id'] = img['id']
                ann_['id'] = ann_idx
                ann_idx += 1
                anns_.append(ann_)
            annotations += anns_
        print('write {} to:'.format(mode))
        print(OUT_PATH.format(split, mode))
        write_json['annotations'] = annotations
        write_json['images'] = imgs
        write_json['categories'] = coco.dataset['categories']
        json.dump(write_json, open(OUT_PATH.format(split, mode), 'w'))

        _ = 1
    print("Done")


if __name__ == '__main__':
    mode = 'fraction0.30'  # number1000, fraction0.2
    SPLITS = ['train']#['train', 'val']
    data_root = 'data/coco'
    ANN_PATH = os.path.join(data_root, 'annotations/instances_{}2017.json')
    OUT_PATH = 'data/instances_{}2017_{}.json'
    start_transform(SPLITS, mode)


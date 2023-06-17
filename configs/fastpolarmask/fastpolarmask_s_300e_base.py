_base_ = ['../_base_/default_runtime.py']

img_scale = (640, 640)

width_mult=0.50
depth_mult=0.33
neck_head_channels = [768, 384, 192]
point_num = 36

# model settings
model = dict(
    type='PolarMask',
    sybn=True,
    input_size=img_scale,
    size_multiplier=32,
    random_size_range=(15, 25),
    random_size_interval=1,
    backbone=dict(
        type='CSPResNet',
        layers=[3, 6, 6, 3],
        channels=[64, 128, 256, 512, 1024],
        return_idx=[1, 2, 3],
        depth_wise=False,
        use_large_stem=True,
        width_mult=width_mult,
        depth_mult=depth_mult,
        act='swish',
        init_cfg=dict(type='Pretrained',
                      checkpoint='pretrain/_CSPResNetb_s_pretrained.pth')),
    neck=dict(
        type='CustomCSPPAN',
        in_channels=[256, 512, 1024],
        out_channels=neck_head_channels,
        stage_num=1,
        block_num=3,
        width_mult=width_mult,
        depth_mult=depth_mult,
        # drop_block=False,
        # block_size=3,
        # keep_prob=0.9,
        act='swish',
        spp=True),
    mask_head=dict(
        type="PolarMaskFastHead",
        num_classes=80,
        point_num=point_num,
        in_channels=neck_head_channels,
        width_mult=width_mult,
        depth_mult=depth_mult,
        strides=[32, 16, 8],
        grid_cell_scale=5.0,
        grid_cell_offset=0.5,
        mask_ct_mode='ori',
        norm_on_bbox=False,
        use_varifocal_loss=True,
        eval_input_size=[640, 640],
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='Swish'),

        bbox_coder=dict(
            type='DistancePointMaskCoder',
            point_num=point_num,
            dist_k=3,
            mem_limit=2.),
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            weight_grad=True,
            alpha=0.75,
            gamma=2.0,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='GIoULoss',
            loss_weight=2.0),
        loss_mask=dict(
            type='MaskIOULoss',
            loss_weight=2.),
        ),
    train_cfg=dict(
        initial_epoch=100,
        static_assigner=dict(
            type="PolarMaskATSSAssigner",
            topk=9),
        assigner=dict(
            type="PolarMaskTaskAlignedAssigner",
            topk=13,
            alpha=1.0,
            beta=6.0
            ),
        # assigner=dict(
        #     type='SimOTAAssigner',
        #     center_radius=2.5,
        #     iou_weight=6.,
        #     use_vfl=True
        #     )
        ),
    test_cfg=dict(
        score_thr=0.05,
        nms=dict(type='nms',
                 iou_threshold=0.6),
        max_per_img=100,
        nms_pre=1000)
    )

# dataset settings
data_root = 'data/coco/'
dataset_type = 'CocoDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         with_bbox=True, with_mask=True),
    
    # data aug
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=18,  # (0.5, 1.5)
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
        ),
    dict(type='Expand',
         mean=(123.675, 116.28, 103.53),
         to_rgb=True),
    dict(type="MinIoURandomCrop"),
    dict(type='Resize',
        img_scale=[img_scale],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type='FilterAnnotations',
         min_gt_bbox_wh=(2, 2),
         min_gt_mask_area=5,
         by_box=True,
         by_mask=True,
         keep_empty=True),
    dict(type='PolarMaskTarget', point_num=point_num,
         oversampling_rate=int(32/point_num)),
    
    # dict(type='PolarMaskCoderDebugShow', point_num=point_num),  # debug
    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PolarMaskFormatBundle'),
    # dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',
                               'gt_masks', 'mask_centers', 'mask_contours'])
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize',
                img_scale=[(640, 640)],
                multiscale_mode='value',
                keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]


data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    persistent_workers=True,
    pin_memory=True,
    
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01 / 16 * 32 * 8,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(
        norm_decay_mult=0.,
        bias_decay_mult=0.,
        )
    )
optimizer_config = dict(grad_clip=None)


max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 10

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(type='NumClassCheckHook'),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0002,
        priority=49),
    
    # dict(type='MemoryProfilerHook', interval=10)
]
checkpoint_config = dict(interval=30)
evaluation = dict(
    save_best='auto',
    interval=20,
    dynamic_intervals=[(max_epochs - 5, 1)],
    metric='segm')

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
find_unused_parameters = True

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
# mp_start_method = 'spawn'
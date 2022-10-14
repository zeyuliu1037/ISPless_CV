_base_ = [
    '../../_base_/datasets/fine_tune_based/base_coco.py',
    '../../_base_/schedules/schedule.py',
    '../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../_base_/default_runtime.py'
]
lr_config = dict(warmup_iters=1000, step=[85000, 100000])
runner = dict(max_iters=110000)
# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(
        depth=101,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False),
    roi_head=dict(bbox_head=dict(num_classes=80)))
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromNpy'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromNpy'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# 'ALL_CLASSES', 'BASE_CLASSES'
data_root='/home/ubuntu/Invertible-ISP/COCO_RAW_WB/results_latest'
data = dict(
    train=dict(
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/ann/instances_maxitrain.json',
                img_prefix='/home/ubuntu/Invertible-ISP/COCO_RAW_WB/results_latest')
        ],
        pipeline=train_pipeline,
        classes='ALL_CLASSES'
    ),
    val=dict(
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/ann/instances_minival.json',
                img_prefix='/home/ubuntu/Invertible-ISP/COCO_RAW_WB/results_latest')
        ],
        classes='ALL_CLASSES',
        pipeline=test_pipeline
    ),
    test=dict(
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/ann/instances_minival.json',
                img_prefix='/home/ubuntu/Invertible-ISP/COCO_RAW_WB/results_latest')
        ],
        classes='ALL_CLASSES',
        pipeline=test_pipeline
    )
)
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
work_dir = 'work_dirs/'
load_from = 'ckpt/faster_rcnn_r101_caffe_fpn_mstrain_3x_coco_20210526_095742-a7ae426d.pth'


'''
python -m tools.detection.misc.initialize_bbox_head \
    --src1 work_dirs/your_model.pth \
    --method randinit \
    --save-dir work_dirs/ --coco
'''

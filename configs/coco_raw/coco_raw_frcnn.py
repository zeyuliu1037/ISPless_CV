# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        frozen_stages=-1,
        norm_cfg=dict(requires_grad=True),
        norm_eval=False,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromNpy'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        # img_scale=[(1333, 640), (1333, 800)],
        # multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    train=dict(
        pipeline=train_pipeline,
        img_prefix='/home/ubuntu/Invertible-ISP/COCO_RAW_WB/results_latest',
        ann_file='data/ann/instances_maxitrain.json'),
    val=dict(
        pipeline=test_pipeline,
        img_prefix='/home/ubuntu/Invertible-ISP/COCO_RAW_WB/results_latest',
        ann_file='data/ann/instances_minival.json'),
    test=dict(
        pipeline=test_pipeline,
        img_prefix='/home/ubuntu/Invertible-ISP/COCO_RAW_WB/results_latest',
        ann_file='data/ann/instances_minival.json'))

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)





# We can use the pre-trained Mask RCNN model to obtain higher performance
work_dir = 'work_dirs/coco_raw/frcnn_lr_0.002_new_image_norm'
load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth' # Ori mAP: 39.9

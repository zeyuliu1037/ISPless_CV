_base_ = [
    '../../_base_/datasets/fine_tune_based/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../tfa_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
evaluation = dict(interval=6000)
checkpoint_config = dict(interval=6000)
optimizer = dict(lr=0.0001)
lr_config = dict(
    warmup_iters=10, step=[
        24000,
    ])
runner = dict(max_iters=30000)
model = dict(roi_head=dict(bbox_head=dict(num_classes=80)))
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFileDemosaic'),
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
    dict(type='LoadImageFromFileDemosaic'),
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
data = dict(
    train=dict(
        type='FewShotCocoDataset',
        ann_cfg=[dict(type='ann_file', method='TFA', setting='30SHOT', 
                ann_file='data/VOCRAW/output_trainX2.json',
                img_prefix='data/VOCRAW/JPEGImages')],
        classes='ALL_CLASSES',
        pipeline=train_pipeline,
        num_novel_shots=30,
        num_base_shots=30
    ),
    val=dict(
        ann_cfg=[dict(type='ann_file',
                ann_file='data/VOCRAW/output_testX2.json',
                img_prefix='data/VOCRAW/JPEGImages')],
        classes='ALL_CLASSES',
        pipeline=test_pipeline
    ),
    test=dict(
        ann_cfg=[dict(type='ann_file',
                ann_file='data/VOCRAW/output_testX2.json',
                img_prefix='data/VOCRAW/JPEGImages')],
        classes='ALL_CLASSES',
        pipeline=test_pipeline
    )
)
load_from = ('work_dirs/base_model_random_init_bbox_head.pth')


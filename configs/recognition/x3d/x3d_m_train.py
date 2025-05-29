_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='X3D',
        gamma_w=1,
        gamma_b=2.25,
        gamma_d=2.2,
        freeze=True,
        pretrained='./pretrained_weights/x3d_m.pth'
        ),
    cls_head=dict(
        type='X3DHead',
        in_channels=432,
        num_classes=3,
        spatial_type='avg',
        dropout_ratio=0.0,
        fc1_bias=False,
        average_clips='prob',
        topk=(1, 2),
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.38, 57.38, 57.38],
        format_shape='NCTHW'),
        )


dataset_type = 'VideoDataset'
data_root_train = '../../datasets/data/train/'
data_root_val = '../../datasets/data/val/'
ann_file_train = './tools/data/sthv2/subsets/office_data/train.txt'
ann_file_val = './tools/data/sthv2/subsets/office_data/val.txt'

num_epochs = 10

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit',**file_client_args),
    dict(
        type='UniformSampleFrames',
        clip_len=16,
        num_clips=1,
    ),
    dict(type='DecordDecode'),
    # dict(type='PytorchVideoWrapper', op='RandAugment', magnitude=2, num_layers=4),
    dict(type='TorchVisionWrapper', op='RandomVerticalFlip',p=0.3),
    dict(type='TorchVisionWrapper', op='RandomHorizontalFlip',p=0.3),
    dict(type='TorchVisionWrapper', op='GaussianBlur',kernel_size=3),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='UniformSampleFrames',
        clip_len=16,
        num_clips=9,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataset_cfg =dict(
    type=dataset_type,
    ann_file=ann_file_train,
    data_prefix=dict(video=data_root_train),
    pipeline=train_pipeline)


val_dataset_cfg =dict(
    type=dataset_type,
    ann_file=ann_file_val,
    data_prefix=dict(video=data_root_val),
    pipeline=val_pipeline,
    test_mode=True)

train_dataloader = dict(
batch_size=16,
num_workers=8,
persistent_workers=True,
sampler=dict(type='weighted_sampler',num_samples=400),
batch_sampler=dict(type='batch_sampler',drop_last=True),
dataset=train_dataset_cfg,
)

val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset_cfg
)

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=num_epochs, 
    val_interval=1,
    )

val_cfg = dict(type='ValLoop')

val_evaluator = [
    dict(type='AccMetric',metric_options=dict(top_k_accuracy=dict(topk=(1, 2)))),
]

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=num_epochs)

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001),
)


default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',save_best='auto')
)

custom_hooks = [
    dict(type ='ConfusionMatrixHook',
         class_map="./tools/data/sthv2/subsets/office_data/label_mapping.json"),
    dict(
        type="MLflowHook"
        ),
    ]

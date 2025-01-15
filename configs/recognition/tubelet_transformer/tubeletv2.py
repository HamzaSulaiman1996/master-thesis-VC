_base_ = ['../../_base_/default_runtime.py']

img_size = 224

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViTubelet',
        pretrained='./pretrained_weights/vit_unpooled.pth',
        freeze=True, ## freeze all stages
    ),
    neck=dict(
        type='TransformerTubelet',
        feature_inchannels=768,
        tubelet_feature_size=128,
        transformer_heads = 2,
        transformer_layers = 1,
        transformer_ffn = 256,
        tubelet_size = 4,
        add_cls_token = True,
    ),
    cls_head=dict(
        type='MLP',
        in_channels= 128,
        label_smooth_eps = 0.01,
        dropout_rate= 0.1,
        num_classes=3,
        average_clips='prob',
        topk=(1, 2),
        use_cls_token=True,
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW')
)

dataset_type = 'VideoDataset'
data_root_train = '../../datasets/data/train/'
data_root_val = '../../datasets/data/val/'
ann_file_train = './tools/data/sthv2/subsets/office_data/train.txt'
ann_file_val = './tools/data/sthv2/subsets/office_data/val.txt'

num_epochs = 80
tags = {
    "Issue_id":"AFC-4014",
    }


file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit',**file_client_args),
    dict(
        type='UniformSampleFrames',
        clip_len=16,
        num_clips=1,
    ),
    dict(type='DecordDecode'),
    # dict(type='CustomPytorchVideoWrapper',magnitude=0, num_layers=4, prob=0.3),
    dict(type='TorchVisionWrapper', op='RandomVerticalFlip',p=0.3),
    dict(type='TorchVisionWrapper', op='RandomHorizontalFlip',p=0.3),
    dict(type='TorchVisionWrapper', op='GaussianBlur',kernel_size=3),
    dict(type='Resize', scale=(img_size, img_size), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit',**file_client_args),
    dict(
        type='UniformSampleFrames',
        clip_len=16,
        num_clips=9,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(img_size, img_size), keep_ratio=False),
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
sampler=dict(type='weighted_sampler'),
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
    val_interval=5,
    )

val_cfg = dict(type='ValLoop')

val_evaluator = [
    dict(type='AccMetric',metric_options=dict(top_k_accuracy=dict(topk=(1, 2)))),
]


param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=num_epochs)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001
        ),
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',save_best='auto')
)



custom_hooks = [
    dict(type ='ConfusionMatrixHook',
         class_map="./tools/data/sthv2/subsets/office_data/label_mapping.json"),
    dict(
        type="MLflowHook",
        tags=tags,
        ),
    ]
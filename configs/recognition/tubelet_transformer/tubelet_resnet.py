_base_ = ['../../_base_/default_runtime.py']

img_size = 224
batch_size = 32
sampled_frames = 16

base_lr = 0.0005
scaled_lr = base_lr * (batch_size/256)

num_epochs = 20

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='Resnet50',
        pretrained=True,
        freeze=True, ## freeze all stages
    ),
    neck=dict(
        type='TransformerTubelet',
        feature_inchannels=2048,
        tubelet_feature_size=128,
        sample_frames = sampled_frames,
        patch_size = (7,7),
        image_size = (7,7),
        tubelet_size = 2,
        transformer_heads = 2,
        transformer_layers = 1,
        transformer_ffn = 256,
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

# data_root_train = '../../datasets/ssv2_dataset/20bn-something-something-v2/'
# data_root_val = '../../datasets/ssv2_dataset/20bn-something-something-v2/'
# ann_file_train = './tools/data/sthv2/subsets/single_class_1/1000_samples/train.txt'
# ann_file_val = './tools/data/sthv2/subsets/single_class_1/1000_samples/val.txt'



file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit',**file_client_args),
    dict(
        type='UniformSampleFrames',
        clip_len=sampled_frames,
        num_clips=1,
    ),
    dict(type='DecordDecode'),
    dict(type='CustomPytorchVideoWrapper',magnitude=9, num_layers=2, prob=0.5),
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
        clip_len=sampled_frames,
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
batch_size=batch_size,
num_workers=8,
persistent_workers=True,
sampler=dict(type='weighted_sampler', num_samples=500),
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
    val_interval=2,
    )

val_cfg = dict(type='ValLoop')

val_evaluator = [
    dict(type='AccMetric',metric_options=dict(top_k_accuracy=dict(topk=(1, 2)))),
]


# param_scheduler = [
#     dict(T_max=num_epochs, by_epoch=True, type='CosineAnnealingLR')
# ]

param_scheduler = [
    dict(by_epoch=True, end=5, start_factor=0.7, type='LinearLR'),
    dict(begin=5, by_epoch=True, gamma=0.5, step_size=2, type='StepLR'),
]



# optimizer
optim_wrapper = dict(
    
    optimizer=dict(
        betas=(
            0.9,
            0.999,
            ),
        lr=scaled_lr, type='AdamW', weight_decay=0.005),
        clip_grad=dict(max_norm=40, norm_type=2),
        )

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',save_best='auto')
)


custom_hooks = [
    dict(type ='ConfusionMatrixHook',
         class_map='./tools/data/sthv2/subsets/office_data/label_mapping.json'),
    dict(
        type="MLflowHook",
        log_interval=1,
        ),
    ]

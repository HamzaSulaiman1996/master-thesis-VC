_base_ = ['../../_base_/default_runtime.py']

img_size = 224
batch_size = 32
sampled_frames = 16

base_lr = 0.0005
scaled_lr = base_lr * (batch_size/256)
num_epochs = 80

# model settings
model = dict(
    backbone=dict(
        freeze=True,
        pretrained = True,
        type='ViTubelet'),
    cls_head=dict(
        average_clips='prob',
        dropout_rate=0.1,
        in_channels=128,
        label_smooth_eps=0.01,
        num_classes=3,
        topk=(
            1,
            2,
        ),
        type='MLP',
        use_cls_token=True),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    neck=dict(
        add_cls_token=True,
        feature_inchannels=768,
        transformer_ffn=256,
        transformer_heads=2,
        transformer_layers=1,
        tubelet_feature_size=128,
        sample_frames = sampled_frames,
        tubelet_size = 16,
        image_size = (14,14),
        patch_size = (2,2),
        type='TransformerTubelet'),
    type='Recognizer3D')

dataset_type = 'VideoDataset'
data_root_train = '../../datasets/data/train/'
data_root_val = '../../datasets/data/val/'
ann_file_train = './tools/data/sthv2/subsets/office_data/train.txt'
ann_file_val = './tools/data/sthv2/subsets/office_data/val.txt'


file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=sampled_frames, num_clips=1, type='UniformSampleFrames'),
    dict(type='DecordDecode'),
    dict(
        magnitude=9, num_layers=2, prob=0.5, type='CustomPytorchVideoWrapper'),
    dict(op='RandomVerticalFlip', p=0.3, type='TorchVisionWrapper'),
    dict(op='RandomHorizontalFlip', p=0.3, type='TorchVisionWrapper'),
    dict(kernel_size=3, op='GaussianBlur', type='TorchVisionWrapper'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]

val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=sampled_frames, num_clips=9, test_mode=True, type='UniformSampleFrames'),
    dict(type='DecordDecode'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
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
sampler=dict(type='weighted_sampler',num_samples=500),
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



param_scheduler = dict(T_max=num_epochs, by_epoch=True, type='CosineAnnealingLR')


# optimizer
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=scaled_lr, type='AdamW', weight_decay=0.005))

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
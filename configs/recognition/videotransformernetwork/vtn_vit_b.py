_base_ = [
    '../../_base_/models/vtn_vit_b.py', '../../_base_/default_runtime.py'
]


dataset_type = 'VideoDataset'
data_root = '../../datasets/ssv2_dataset/20bn-something-something-v2/'
data_root_val = '../../datasets/ssv2_dataset/20bn-something-something-v2/'
ann_file_train = './tools/data/sthv2/subsets/single_class_1/100_samples/train.txt'
ann_file_val = './tools/data/sthv2/subsets/single_class_1/100_samples/val.txt'


num_epochs = 80


file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSampleFrames',clip_len=32,num_clips=1),
    # dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='PytorchVideoWrapper', op='RandAugment', magnitude=0, num_layers=4),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1,test_mode=True),
    dict(type='UniformSampleFrames',clip_len=32,num_clips=1,test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataset_cfg =dict(
    type=dataset_type,
    ann_file=ann_file_train,
    data_prefix=dict(video=data_root),
    pipeline=train_pipeline)

val_dataset_cfg =dict(
    type=dataset_type,
    ann_file=ann_file_val,
    data_prefix=dict(video=data_root_val),
    pipeline=val_pipeline)



train_dataloader = dict(
batch_size=16,
num_workers=8,
persistent_workers=True,
sampler=dict(type='DefaultSampler', shuffle=True),
dataset=train_dataset_cfg,
)

val_dataloader = dict(
batch_size=2,
num_workers=4,
persistent_workers=True,
sampler=dict(type='DefaultSampler', shuffle=False),
dataset=val_dataset_cfg,
)

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=num_epochs, 
    val_interval=5
    )
val_cfg = dict(type='ValLoop')

val_evaluator = [
    dict(type='AccMetric',metric_options=dict(top_k_accuracy=dict(topk=(1, 2)))),
    dict(type='ConfusionMatrix')
]

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=num_epochs)

# param_scheduler = dict(
#     type='StepLR',
#     by_epoch = True,
#     step_size = 6,
#     gamma = 0.9
#     )
    

# param_scheduler = [
#     dict(type='LinearLR',
#          by_epoch = True,
#          end=13,
#          end_factor=1
#          ),
#     dict(type='LinearLR',
#          by_epoch=True,
#          end=23,
#          end_factor=0.1,
#          ),
#     dict(type='LinearLR',
#          by_epoch=True,
#          end=num_epochs,
#          end_factor=0.01,
#          ),
# ]

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001
        ),
    clip_grad=dict(max_norm=40, norm_type=2))



# base_lr = 3e-4
# optim_wrapper = dict(
#     optimizer=dict(
#         type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',save_best='auto'),
)


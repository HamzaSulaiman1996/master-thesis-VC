_base_ = ['../../_base_/default_runtime.py']

img_size = 224

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViTubelet',
        freeze=True, ## freeze all stages
    ),
    neck=dict(
        type='TransformerTubelet',
        feature_inchannels=768,
        tubelet_feature_size=128,
        transformer_heads = 2,
        transformer_layers = 1,
        transformer_ffn = 256,
        tubelet_size = 2,
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
data_root_val = '../../datasets/data/val/'
ann_file_val = './tools/data/sthv2/subsets/office_data/val.txt'

tags = {
    "Issue_id":"AFC-4014",
    "train_experiment_id":"tubeletv2_20250114_071802",
    }


file_client_args = dict(io_backend='disk')


test_pipeline = [
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



test_dataset_cfg =dict(
    type=dataset_type,
    ann_file=ann_file_val,
    data_prefix=dict(video=data_root_val),
    pipeline=test_pipeline,
    test_mode=True)


test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset_cfg
)


test_cfg = dict(type='TestLoop')

test_evaluator = [
    dict(type='AccMetric',metric_options=dict(top_k_accuracy=dict(topk=(1, 2)))),
]

custom_hooks = [
    dict(type ='ConfusionMatrixHook',
         class_map="./tools/data/sthv2/subsets/office_data/label_mapping.json"),
    dict(
        type="MLflowHook",
        tags=tags,
        ),
    dict(type ='MisclassificationHook',
    interval=1,
    class_map="./tools/data/sthv2/subsets/office_data/label_mapping.json"
    ),
    ]
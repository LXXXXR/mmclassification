_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

data = dict(samples_per_gpu=64)

# optimizer
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(_delete_=True, policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=120)

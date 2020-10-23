# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MyOwnNet'),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=16*8*8,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    )
)
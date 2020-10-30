# optimizer
#optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='SGD', lr=1e-3, momentum=0)
#optimizer = dict(type='RMSprop', lr=1e-3, alpha=0.99)
optimizer = dict(type='Adam',lr=5e-4, betas=(0.9,0.99))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=10)

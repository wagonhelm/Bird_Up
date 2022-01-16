TRAINING_PARAMS = \
{
    "backbone": "nasnetamobile",
    "backbonePretrained": "backboneweights/nasnetamobile.pth", #  None to disable

    "algorithm": "yolo",

    "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
    "labels": ["raccoon"],
    "detectionThreshold": 0.60,
    "nmsTheshold": 0.45,

    "lr": {
        "backboneLR": 0.001,
        "otherLR": 0.001,
        "freezeBackbone": True,   #  freeze backbone weights
        "thawBackboneEpoch": 100,
    },

    "batchSize": 16,
    "numDataloaderThread":8,
    "epochs": 600,
    "trainImageSize":(416,416), # (h,w)

    "trainListFile": ["datasets/train_raccoon.txt"],
    "valListFile": ["datasets/test_raccoon.txt"],
    "imageDirectory": "datasets/raccoon/",


    "workingDirectory": "trainingLogs/raccoonDataset",
}
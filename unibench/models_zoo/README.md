# List of Implemented VLMs

|                                 |   Dataset Size (Million) |   Number of Parameters (Million) | Learning Objective          | Architecture   | Model Name               |
|:--------------------------------|---------------:|-------------:|:----------------------------|:---------------|:-------------------|
| blip_vitB16_14m                 |             14 |           86 | BLIP                        | vit            | BLIP ViT B 16      |
| blip_vitL16_129m                |            129 |          307 | BLIP                        | vit            | BLIP ViT L 16      |
| blip_vitB16_129m                |            129 |           86 | BLIP                        | vit            | BLIP ViT B 16      |
| blip_vitB16_coco                |            129 |           86 | BLIP                        | vit            | BLIP ViT B 16      |
| blip_vitB16_flickr              |            129 |           86 | BLIP                        | vit            | BLIP ViT B 16      |
| blip_vitL16_coco                |            129 |          307 | BLIP                        | vit            | BLIP ViT L 16      |
| blip_vitL16_flickr              |            129 |          307 | BLIP                        | vit            | BLIP ViT L 16      |
| eva02_vitE14_plus_2b            |           2000 |         4350 | EVA02                       | vit            | EVA02 ViT E 14     |
| eva02_vitE14_2b                 |           2000 |         4350 | EVA02                       | vit            | EVA02 ViT E 14     |
| eva02_vitL14_2b                 |           2000 |          307 | EVA02                       | vit            | EVA02 ViT L 14     |
| eva02_vitB16_2b                 |           2000 |           86 | EVA02                       | vit            | EVA02 ViT B 16     |
| eva01_vitG14_plus_2b            |           2000 |         1011 | EVA01                       | vit            | EVA01 ViT g 14     |
| eva01_vitG14_400m               |            400 |         1011 | EVA01                       | vit            | EVA01 ViT g 14     |
| clipa_vitbigG14                 |           1280 |         1843 | CLIPA                       | vit            | CLIPA ViT G 14     |
| clipa_vitH14                    |           1280 |          633 | CLIPA                       | vit            | CLIPA ViT H 14     |
| clipa_vitL14                    |           1280 |          307 | CLIPA                       | vit            | CLIPA ViT L 14     |
| siglip_vitL16                   |          10000 |          307 | Contrastive (sigmoid-based) | vit            | SigLIP ViT L 16    |
| siglip_vitB16                   |          10000 |           86 | Contrastive (sigmoid-based) | vit            | SigLIP ViT B 16    |
| openclip_vitB32_metaclip_fullcc |           2500 |           86 | Contrastive                 | vit            | MetaCLIP ViT B 32  |
| openclip_vitB16_metaclip_400m   |            400 |           86 | Contrastive                 | vit            | MetaCLIP ViT B 16  |
| openclip_vitB32_metaclip_400m   |            400 |           86 | Contrastive                 | vit            | MetaCLIP ViT B 32  |
| openclip_vitB16_metaclip_fullcc |           2500 |           86 | Contrastive                 | vit            | MetaCLIP ViT B 16  |
| openclip_vitL14_dfn2b           |           2000 |          307 | Contrastive                 | vit            | OpenCLIP ViT L 14  |
| openclip_vitL14_metaclip_400    |            400 |          307 | Contrastive                 | vit            | MetaCLIP ViT L 14  |
| openclip_vitL14_metaclip_fullcc |           2500 |          307 | Contrastive                 | vit            | MetaCLIP ViT L 14  |
| openclip_vitH14_metaclip_fullcc |           2500 |          633 | Contrastive                 | vit            | MetaCLIP ViT H 14  |
| openclip_vitH14_dfn5b           |           5000 |          633 | Contrastive                 | vit            | OpenCLIP ViT H 14  |
| openclip_convnext_base          |            400 |           88 | Contrastive                 | conv           | OpenCLIP ConvNext  |
| openclip_vitB32_datacomp_s      |             13 |           86 | Contrastive                 | vit            | DataComp ViT B 32  |
| openclip_vitB32_datacomp_m      |            128 |           86 | Contrastive                 | vit            | DataComp ViT B 32  |
| openclip_vitB32_datacomp_xl     |          12800 |           86 | Contrastive                 | vit            | DataComp ViT B 32  |
| openclip_vitB16_datacomp_xl     |          12800 |           86 | Contrastive                 | vit            | DataComp ViT B 16  |
| openclip_vitB16_datacomp_l      |           1280 |           86 | Contrastive                 | vit            | DataComp ViT B 16  |
| openclip_vitH14                 |           2000 |          633 | Contrastive                 | vit            | OpenCLIP ViT H 14  |
| xvlm_flickr                     |             16 |           86 | XVLM                        | Swin           | XVLM Swin B        |
| flava_full                      |             70 |           86 | Other                       | vit            | FLAVA ViT B 32     |
| openclip_vitL14_400m            |            400 |          307 | Contrastive                 | vit            | OpenCLIP ViT L 14  |
| openclip_vitL14_datacomp_xl     |          12800 |          307 | Contrastive                 | vit            | DataComp ViT L 14  |
| openclip_vitL14_2b              |           2000 |          307 | Contrastive                 | vit            | OpenCLIP ViT L 14  |
| clip_vitL14                     |            400 |          307 | Contrastive                 | vit            | CLIP ViT L 14      |
| xvlm_coco                       |             16 |           86 | XVLM                        | Swin           | XVLM Swin B        |
| openclip_vitB32_400m            |            400 |           86 | Contrastive                 | vit            | OpenCLIP ViT B 32  |
| openclip_vitB32_2b              |           2000 |           86 | Contrastive                 | vit            | OpenCLIP ViT B 32  |
| openclip_vitG14_2b              |           2000 |         1011 | Contrastive                 | vit            | OpenCLIP ViT g 14  |
| openclip_vitbigG14_2b           |           2000 |         1843 | Contrastive                 | vit            | OpenCLIP ViT G 14  |
| openclip_vitB16_2b              |           2000 |           86 | Contrastive                 | vit            | OpenCLIP ViT B 16  |
| openclip_vitB16_400m            |            400 |           86 | Contrastive                 | vit            | OpenCLIP ViT B 16  |
| opencoca_vitL14_2b              |           2000 |          307 | Other                       | vit            | OpenCOCA ViT L 14  |
| opencoca_vitB32_2b              |           2000 |           86 | Other                       | vit            | OpenCOCA ViT B 32  |
| negclip_vitB32                  |            400 |           86 | Negative CLIP               | vit            | NegCLIP ViT B 32   |
| clip_vitB16                     |            400 |           86 | Contrastive                 | vit            | CLIP ViT B 16      |
| clip_resnet50                   |            400 |           38 | Contrastive                 | conv           | CLIP ResNet50      |
| openclip_resnet101_yfcc         |             15 |           56 | Contrastive                 | conv           | OpenCLIP ResNet101 |
| openclip_resnet50_yfcc          |             15 |           38 | Contrastive                 | conv           | OpenCLIP ResNet50  |
| openclip_resnet50_cc            |             12 |           38 | Contrastive                 | conv           | OpenCLIP ResNet50  |
| clip_resnet101                  |            400 |           56 | Contrastive                 | conv           | CLIP ResNet101     |
| clip_resnet50x4                 |            400 |           87 | Contrastive                 | conv           | CLIP ResNet50x4    |
| clip_resnet50x16                |            400 |          167 | Contrastive                 | conv           | CLIP ResNet50x16   |
| clip_resnet50x64                |            400 |          420 | Contrastive                 | conv           | CLIP ResNet50x64   |
| clip_vitB32                     |            400 |           86 | Contrastive                 | vit            | CLIP ViT B 32      |
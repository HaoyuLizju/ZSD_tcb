# TCB

Tcb is an open source implementation on our proposed method--''Zero-shot Object Detection Based on Dynamic Semantic Vectors''. Our work is based on [ZSI](https://github.com/zhengye1995/Zero-shot-Instance-Segmentation) and mmdetection 1.0rc0.

## Requirements

​	python3.7.12

​	pytorch1.1.0

​	GCC = 7.3.0

​	G++ = 7.3.0  (for CentOS) 

​	NCCL 2

Use the following command to install the environment:

```shell
conda env create -f tcb_zsd.yaml
```

## Dataset Prepare

Download  [annotations files](https://drive.google.com/drive/folders/1-7YCeapy6aOBszYFTBmIKSsPfyvZPKGl?usp=sharing) to

```
data/coco/annotations/
```

Put MSCOCO-2014 dataset to path:

```
data/coco/train2014/
data/coco/val2014/
```

## Model Train ＆ Test

### Train

```
./tools/dist_train.sh + #path_to_config + #num_OF_GPU + #port_number
```

##### COCO 65_15 split:

```
./tools/dist_train.sh configs/zsd/65_15/train/zsd_TCB_train.py 4 1234
```

##### COCO 48_17 split:

```
./tools/dist_train.sh configs/zsd/48_17/train/zsd_TCB_train.py 4 1234
```

### Test

​	-inference

```
./tools/dist_test.sh  #path_to_config #path_to_checkpoint #num_OF_GPU --json_out #path_to_resultdirectory
```

​	-evaluation

```
python tools/zsi_coco_eval.py #path_to_resultdirectory --ann #path_to_annotationfile
```

##### COCO 65_15 split:

Download [tcb_zsd](https://drive.google.com/drive/folders/1uFb6YfC99js_N4LfRcGCqRTuGQ6pSgYq?usp=sharing) on COCO65_15, and put it into work_dirs/zsd/65_15/latest.pth

###### ZSD

​	-inference

```
./tools/dist_test.sh configs/zsd/65_15/test/zsd/zsd_TCB_test.py work_dirs/zsd/65_15/latest.pth 4 --json_out results/zsd_65_15.json
```

​	-evaluation

```
python tools/zsi_coco_eval.py results/zsd_65_15.bbox.json --ann data/coco/annotations/instances_val2014_unseen_65_15.json
```

###### GZSD

​	-inference

```
./tools/dist_test.sh configs/zsd/65_15/test/gzsd/gzsd_TCB_test.py work_dirs/zsd/65_15/latest.pth 4 --json_out results/gzsd_65_15.json
```

​	-evaluation

```
python tools/gzsi_coco_eval.py results/gzsd_65_15.bbox.json --ann data/coco/annotations/instances_val2014_gzsi_65_15.json --gzsi --num-seen-classes 65
```

##### COCO 48_17 split:

Download [tcb_zsd](https://drive.google.com/drive/folders/1uFb6YfC99js_N4LfRcGCqRTuGQ6pSgYq?usp=sharing) on COCO48_17, and put it into work_dirs/zsd/48_17/latest.pth

###### ZSD

​	-inference

```
./tools/dist_test.sh configs/zsd/48_17/test/zsd/zsd_TCB_test.py work_dirs/zsd/48_17/latest.pth 4 --json_out results/zsd_48_17.json
```

​	-evaluation

```
python tools/zsi_coco_eval.py results/zsd_48_17.bbox.json --ann data/coco/annotations/instances_val2014_unseen_48_17.json 
```

###### GZSD

​	-inference

```
./tools/dist_test.sh configs/zsd/48_17/test/gzsd/gzsd_TCB_test.py work_dirs/zsd/48_17/latest.pth 4 --json_out results/gzsd_48_17.json
```

​	-evaluation

```
python tools/gzsi_coco_eval.py results/gzsd_48_17.bbox.json --ann data/coco/annotations/instances_val2014_gzsi_48_17.json --gzsi --num-seen-classes 48
```


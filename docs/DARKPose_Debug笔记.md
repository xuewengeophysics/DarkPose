# HRNet_Debug笔记

1、TEST.SHIFT_HEATMAP

```
Traceback (most recent call last):
  File "tools/train.py", line 223, in <module>
    main()
  File "tools/train.py", line 78, in main
    update_config(cfg, args)
  File "/data/wenxue/DarkPose-master-20200912/tools/../lib/config/default.py", line 129, in update_config
    cfg.merge_from_file(args.cfg)
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/yacs/config.py", line 213, in merge_from_file
    self.merge_from_other_cfg(cfg)
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/yacs/config.py", line 217, in merge_from_other_cfg
    _merge_a_into_b(cfg_other, self, self, [])
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/yacs/config.py", line 478, in _merge_a_into_b
    _merge_a_into_b(v, b[k], root, key_list + [k])
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/yacs/config.py", line 491, in _merge_a_into_b
    raise KeyError("Non-existent config key: {}".format(full_key))
KeyError: 'Non-existent config key: TEST.SHIFT_HEATMAP'
```

 



```
Traceback (most recent call last):
  File "tools/train.py", line 232, in <module>
    main()
  File "tools/train.py", line 80, in main
    update_config(cfg, args)
  File "/data/wenxue/hrnet/tools/../lib/config/default.py", line 128, in update_config
    cfg.merge_from_file(args.cfg)
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/yacs/config.py", line 213, in merge_from_file
    self.merge_from_other_cfg(cfg)
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/yacs/config.py", line 217, in merge_from_other_cfg
    _merge_a_into_b(cfg_other, self, self, [])
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/yacs/config.py", line 478, in _merge_a_into_b
    _merge_a_into_b(v, b[k], root, key_list + [k])
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/yacs/config.py", line 491, in _merge_a_into_b
    raise KeyError("Non-existent config key: {}".format(full_key))
KeyError: 'Non-existent config key: TEST.BLUR_KERNEL'
```





解决方法：

```
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
```

+ https://github.com/pytorch/pytorch/issues/30459

```
class NoOpModule(nn.Module):
    """
    https://github.com/pytorch/pytorch/issues/30459#issuecomment-597679482
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return args
```

Then in `make_transition_layer`: `transition_layers.append(NoOpModule())`;
And in `_make_fuse_layers`: `fuse_layer.append(NoOpModule())`

And then in forward, for each respective stage (e.g. stage 3 here):

```
            if not isinstance(self.transition3[i], NoOpModule):
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
```





 File "/data/wenxue/projects_work/ir_pig_keypoints/hrnet/tools/../lib/core/function.py", line 200, in validate
    filenames, imgnums
  File "/data/wenxue/projects_work/ir_pig_keypoints/hrnet/tools/../lib/dataset/infrared.py", line 433, in evaluate
    res_file, res_folder)
  File "/data/wenxue/projects_work/ir_pig_keypoints/hrnet/tools/../lib/dataset/infrared.py", line 503, in _do_python_keypoint_eval
    coco_dt = self.coco.loadRes(res_file)
  File "/home/wenxue/.local/lib/python3.6/site-packages/pycocotools-2.0-py3.6-linux-x86_64.egg/pycocotools/coco.py", line 325, in loadRes
    'Results do not correspond to current coco set'
AssertionError: Results do not correspond to current coco set



```
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# 根据annotion文件,加载数据集信息,该处只加载了person关键点的数据
self.coco = COCO(self._get_ann_file_keypoint()
        # 获得数据集中标注的类别，该处只有person一个类
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
                
                
                    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids
        
        
            def _load_coco_keypoint_annotation_kernal(self, index):
        """
        根据index，加载单个person关键点数据信息
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        # 获得包含person图片信息
        im_ann = self.coco.loadImgs(index)[0]
        # 获得图片的大小
        width = im_ann['width']
        height = im_ann['height']

        # 获得包含person图片的注释id
        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        # 根据注释id,获得对应的注释信息
        objs = self.coco.loadAnns(annIds)
        

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str
```





```
  File "/data/wenxue/projects_work/ir_pig_keypoints/hrnet/tools/../lib/core/function.py", line 200, in validate
    filenames, imgnums
  File "/data/wenxue/projects_work/ir_pig_keypoints/hrnet/tools/../lib/dataset/infrared.py", line 443, in evaluate
    res_file, res_folder)
  File "/data/wenxue/projects_work/ir_pig_keypoints/hrnet/tools/../lib/dataset/infrared.py", line 523, in _do_python_keypoint_eval
    coco_eval.evaluate()
  File "/home/wenxue/.local/lib/python3.6/site-packages/pycocotools-2.0-py3.6-linux-x86_64.egg/pycocotools/cocoeval.py", line 149, in evaluate
    for imgId in p.imgIds
  File "/home/wenxue/.local/lib/python3.6/site-packages/pycocotools-2.0-py3.6-linux-x86_64.egg/pycocotools/cocoeval.py", line 150, in <dictcomp>
    for catId in catIds}
  File "/home/wenxue/.local/lib/python3.6/site-packages/pycocotools-2.0-py3.6-linux-x86_64.egg/pycocotools/cocoeval.py", line 229, in computeOks
    e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
ValueError: operands could not be broadcast together with shapes (4,) (17,) 



```







**loss这么低，怎么回事？**

```
Epoch: [0][0/113]	Time 2.354s (2.354s)	Speed 6.8 samples/s	Data 0.893s (0.893s)	Loss 0.00119 (0.00119)	Accuracy 0.000 (0.000)
```





```
  File "tools/train.py", line 232, in <module>
    main()
  File "tools/train.py", line 202, in main
    final_output_dir, tb_log_dir, writer_dict
  File "/data/wenxue/hrnet/tools/../lib/core/function.py", line 200, in validate
    filenames, imgnums
  File "/data/wenxue/hrnet/tools/../lib/dataset/infrared.py", line 442, in evaluate
    res_file, res_folder)
  File "/data/wenxue/hrnet/tools/../lib/dataset/infrared.py", line 521, in _do_python_keypoint_eval
    pig_eval = COCOeval(self.pig, pig_dt, 'keypoints')
  File "/data/wenxue/hrnet/tools/../lib/dataset/pigeval.py", line 76, in __init__
    self.params = Params(iouType=iouType) # parameters
  File "/data/wenxue/hrnet/tools/../lib/dataset/pigeval.py", line 529, in __init__
    self.setKpParams()
  File "/data/wenxue/hrnet/tools/../lib/dataset/pigeval.py", line 518, in setKpParams
    self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
  File "<__array_function__ internals>", line 6, in linspace
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/numpy/core/function_base.py", line 113, in linspace
    num = operator.index(num)
TypeError: 'numpy.float64' object cannot be interpreted as an integer
```

解决办法：

```
self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05) + 1), endpoint=True)
self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01) + 1), endpoint=True)
```



```
  File "/data/wenxue/projects_work/ir_pig_keypoints/hrnet/tools/../lib/core/function.py", line 200, in validate
    filenames, imgnums
  File "/data/wenxue/projects_work/ir_pig_keypoints/hrnet/tools/../lib/dataset/infrared.py", line 437, in evaluate
    res_file, res_folder)
  File "/data/wenxue/projects_work/ir_pig_keypoints/hrnet/tools/../lib/dataset/infrared.py", line 517, in _do_python_keypoint_eval
    pig_eval.evaluate()
  File "/data/wenxue/projects_work/ir_pig_keypoints/hrnet/tools/../lib/dataset/pigeval.py", line 137, in evaluate
    self._prepare()
  File "/data/wenxue/projects_work/ir_pig_keypoints/hrnet/tools/../lib/dataset/pigeval.py", line 100, in _prepare
    for gt in gts:
TypeError: 'PIG' object is not iterable
```

```
https://blog.csdn.net/lanyang123456/article/details/72812070
class Animal(object):
        def __init__(self, name):
                self.name = name
                self.age = 12
                self._i = 0

        def __iter__(self):
                return self

        def next(self):
                if self._i == 0:
                        self._i += 1
                        return self.name
                elif self._i == 1:
                        self._i += 1
                        return self.age
                else:
                        raise StopIteration()

a1 = Animal("panda")


for one in a1:
        print one
```





```
Epoch: [0][0/113]	Time 0.833s (0.833s)	Speed 19.2 samples/s	Data 0.226s (0.226s)	Loss 0.00136 (0.00136)	Accuracy 0.000 (0.000)
Epoch: [0][100/113]	Time 0.401s (0.430s)	Speed 39.9 samples/s	Data 0.193s (0.202s)	Loss 0.00065 (0.00142)	Accuracy 0.025 (0.043)
Test: [0/12]	Time 0.296 (0.296)	Loss 0.0008 (0.0008)	Accuracy 0.021 (0.021)
Traceback (most recent call last):
  File "tools/train.py", line 232, in <module>
    main()
  File "tools/train.py", line 202, in main
    final_output_dir, tb_log_dir, writer_dict
  File "/data/wenxue/hrnet/tools/../lib/core/function.py", line 200, in validate
    filenames, imgnums
  File "/data/wenxue/hrnet/tools/../lib/dataset/infrared.py", line 429, in evaluate
    oks_thre
  File "/data/wenxue/hrnet/tools/../lib/nms/nms.py", line 119, in oks_nms
    oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)
  File "/data/wenxue/hrnet/tools/../lib/nms/nms.py", line 89, in oks_iou
    e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
ValueError: operands could not be broadcast together with shapes (4,) (17,) 

```







```
shape output =  torch.Size([16, 4, 64, 48])
shape target =  torch.Size([16, 4, 64, 64])
Traceback (most recent call last):
  File "tools/train.py", line 232, in <module>
    main()
  File "tools/train.py", line 196, in main
    final_output_dir, tb_log_dir, writer_dict)
  File "/data/wenxue/hrnet/tools/../lib/core/function.py", line 57, in train
    loss = criterion(output, target, target_weight)
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/torch/nn/modules/module.py", line 550, in __call__
    result = self.forward(*input, **kwargs)
  File "/data/wenxue/hrnet/tools/../lib/core/loss.py", line 34, in forward
    heatmap_gt.mul(target_weight[:, idx])
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/torch/nn/modules/module.py", line 550, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 432, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/torch/nn/functional.py", line 2542, in mse_loss
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
  File "/home/wenxue/.conda/envs/hrnet/lib/python3.6/site-packages/torch/functional.py", line 62, in broadcast_tensors
    return _VF.broadcast_tensors(tensors)
RuntimeError: The size of tensor a (3072) must match the size of tensor b (4096) at non-singleton dimension 1
```









### 参考资料

[1] Overview of Human Pose Estimation Neural Networks — HRNet + HigherHRNet, Architectures and FAQ — 2d3d.ai : https://towardsdatascience.com/overview-of-human-pose-estimation-neural-networks-hrnet-higherhrnet-architectures-and-faq-1954b2f8b249

[2] Distribution-Aware Coordinate Representation for Human Pose Estimation 姿态估计 CVPR2019: https://blog.csdn.net/u012925946/article/details/103868530

[3] [小结]Distribution-Aware Coordinate Representation for Human Pose Estimation: http://www.mclover.cn/blog/index.php/archives/588.html

[4] 寻找通用表征：CVPR 2020上重要的三种解决方案: https://www.jiqizhixin.com/articles/2020-04-25-2

[5] 【论文阅读笔记】HRNet--从代码来看论文: https://blog.csdn.net/weixin_38715903/article/details/101629781

[6] 姿态估计1-00：HR-Net(人体姿态估算)-目录-史上最新无死角讲解: https://blog.csdn.net/weixin_43013761/article/details/106621525
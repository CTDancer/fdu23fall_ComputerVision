# Assignment 2 - Object Detection

## Part 1 - Faster RCNN

### 1. ProposalTargetCreator

1.  **类定义**：
   
   ```python
   class ProposalTargetCreator(object):
   ```
   定义了一个名为 `ProposalTargetCreator` 的类。它在 Faster R-CNN 中用于将真实边界框分配给 proposed RoIs。
   
2. **构造函数 `__init__`**:
   
   ```python
   def __init__(self, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
   ```
   构造函数使用默认参数初始化对象。这些参数控制采样数量（`n_sample`）、正样本比例（`pos_ratio`）以及基于与真实边界框的交并比（IoU）得分确定一个 RoI 被视为正例或负例的阈值（`pos_iou_thresh`、`neg_iou_thresh_hi` 和 `neg_iou_thresh_lo`）。
   
3. **`__call__` 方法**:
   
   ```python
   def __call__(self, roi, bbox, label, loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
   ```
   它接受 RoIs、真实边界框（`bbox`）和标签作为输入，以及边界框坐标的归一化参数。
   
4. **ROI 和 Bounding Box 准备**:
   
   ```python
   n_bbox, _ = bbox.shape
   roi = np.concatenate((roi, bbox), axis=0)
   ```
   该方法首先将输入的 RoIs 与真实bounding box连接起来。这样做是为了在采样过程中将真实框视为候选 RoIs
   
5. **采样前景 RoIs**:
   ```python
   pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
   iou = bbox_iou(roi, bbox)
   gt_assignment = iou.argmax(axis=1)
   max_iou = iou.max(axis=1)
   gt_roi_label = label[gt_assignment] + 1
   ```
   这些代码负责选择一部分 RoIs 作为前景示例。它们计算每个 RoI 与每个真实框的 IoU，确定每个 RoI 的最大 IoU，并根据具有最高 IoU 的真实框的标签为每个 RoI 分配标签。

6. **分类前景和背景**:
   
   ```python
   pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
   pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
   neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
   ```
   算法随后将 RoIs 分类为前景或背景。前景 RoIs 的 IoU 高于 `pos_iou_thresh`，背景 RoIs 的 IoU 低于 `neg_iou_thresh_hi` 但高于 `neg_iou_thresh_lo`。
   
7. **采样**:
   
   ```python
   if pos_index.size > 0:
       pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)
   ```
   随机采样一定数量的前景和背景 RoIs。对于 `neg_index` 也是类似的操作。
   
8. **为背景 RoIs 分配标签**:
   
   ```python
   keep_index = np.append(pos_index, neg_index)
   gt_roi_label = gt_roi_label[keep_index]
   gt_roi_label[pos_roi_per_this_image:] = 0
   sample_roi = roi[keep_index]
   ```
   采样后，将背景 RoIs 的标签设置为 0。该方法跟踪选择了哪些 RoIs。
   
9. **Bounding Box Offsets Calculation**:
   
   ```python
   gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
   gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))
   ```
   最后一步计算每个采样 RoI 的偏移量，以匹配它们与相应的真实框。这是通过使用函数 bbox2loc 完成的，然后对这些偏移量进行归一化。
   
10. **返回**:
   ```python
   return sample_roi, gt_roi_loc, gt_roi_label
   ```
   最后返回采样的 RoIs、计算的偏移量和比例以匹配真实边界框，以及每个采样 RoI 的标签。



总结来说，Faster R-CNN 中的 `ProposalTargetCreator` 负责选择一组提议的 RoIs，根据它们与真实框的 IoU 将它们分类为前景或背景，并为训练网络准备必要的目标数据（边界框偏移量和标签）。



### 2. AnchorTargetCreator

Faster R-CNN 中的 `AnchorTargetCreator` 类将真实边界框分配给锚点以训练RPN

#### a.  `__init__`
```python
def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
```
构造函数使用默认参数初始化类：
- `n_sample`：要采样的锚点数量。
- `pos_iou_thresh`：锚点被视为正样本的 IoU 阈值。
- `neg_iou_thresh`：锚点被视为负样本的 IoU 阈值。
- `pos_ratio`：采样锚点中正样本的比例。



#### b. `__call__` 
```python
def __call__(self, bbox, anchor, img_size):
```
它接受边界框（`bbox`）、锚点框和图像尺寸（`img_size`）作为输入。

1. **图像尺寸和锚点准备**：
   
   ```python
   img_H, img_W = img_size
   n_anchor = len(anchor)
   inside_index = _get_inside_index(anchor, img_H, img_W)
   anchor = anchor[inside_index]
   ```
   提取图像的高度和宽度，并识别图像边界内的锚点。`inside_index` 是图像边界内锚点的索引列表。
   
2. **创建标签和位置目标**：
   
   ```python
   argmax_ious, label = self._create_label(inside_index, anchor, bbox)
   ```
   调用 `_create_label` 方法为每个锚点创建标签（正样本、负样本或忽略），基于它们与真实边界框的 IoU。
   
3. **计算边界框回归目标**：
   ```python
   loc = bbox2loc(anchor, bbox[argmax_ious])
   ```
   使用 `bbox2loc` 函数计算偏移量和比例，以匹配锚点到真实边界框。

4. **映射回原始锚点集**：
   
   ```python
   label = _unmap(label, n_anchor, inside_index, fill=-1)
   loc = _unmap(loc, n_anchor, inside_index, fill=0)
   ```
   `label` 和 `loc` 数组被映射回完整的锚点集。图像边界外的锚点被分配一个标签 -1（忽略）和位置 0。
   
5. **返回**：
   
   ```python
   return loc, label
   ```
   返回每个锚点的位置目标（`loc`）和标签。
   
   

#### c. `_create_label`
这个方法为每个锚点创建标签。

1. **初始化标签数组**：
   
   ```python
   label = np.empty((len(inside_index),), dtype=np.int32)
   label.fill(-1)
   ```
   用 -1 初始化标签数组，表示所有锚点最初都被标记为“忽略”。
   
2. **计算 IoUs**：
   
   ```python
   argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)
   ```
   计算每个锚点与真实边界框之间的 IoU。
   
3. **分配负标签**：
   ```python
   label[max_ious < self.neg_iou_thresh] = 0
   ```
   最大 IoU 小于负 IoU 阈值的锚点被标记为负（0）。

4. **为最高 IoU 的锚点分配正标签**：
   ```python
   label[gt_argmax_ious] = 1
   label[max_ious >= self.pos_iou_thresh] = 1
   ```
   如果锚点对于任何真实框具有最高 IoU 或其 IoU 大于正 IoU 阈值，则将其标记为正（1）。

5. **正样本标签的子采样**：
   
   ```python
   n_pos = int(self.pos_ratio * self.n_sample)
   pos_index = np.where(label == 1)[0]
   ```
   ```python
   if len(pos_index) > n_pos:
       disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
       label[disable_index] = -1
   ```
   如果正样本过多，子采样以满足期望的 `pos_ratio`。
   
6. **负样本标签的子采样**：
   
   ```python
   n_neg = self.n_sample - np.sum(label == 1)
   neg_index = np.where(label == 0)[0]
   ```
   ```python
   if len(neg_index) > n_neg:
       disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
       label[disable_index] = -1
   ```
   类似地，如果负样本过多，子采样以保持总样本数为 `n_sample`。
   
   

#### d.`_calc_ious`
这个方法计算每个锚点与真实边界框之间的 IoUs。

1. **计算 IoUs**：

   ```python
   ious = bbox_iou(anchor, bbox)
   ```
   计算每个锚点与每个真实框之间的 IoU。`ious` 是一个二维数组，其中每个元素 [i, j] 代表第 i 个锚点和第 j 个真实框之间的 IoU。

2. **确定最大 IoUs**：

   ```
   argmax_ious = ious.argmax(axis=1)
   ```

   `argmax(axis=1)` 用于找到每个锚点的最大 IoU 的索引。这意味着，对于每个锚点，它识别出与哪个真实框重叠最多。 因此，`argmax_ious` 是一个数组，其中每个元素对应一个锚点，并包含它与 IoU 最高的真实框的索引。

   

   ```python
   max_ious = ious[np.arange(len(inside_index)), argmax_ious]
   ```

   然后创建了一个数组，包含每个锚点的最大 IoU 值。 `np.arange(len(inside_index))` 生成每个锚点的索引序列。 `ious[np.arange(len(inside_index)), argmax_ious]` 使用这些索引来选择每个锚点与其对应的最佳匹配真实框（由 `argmax_ious` 确定）的 IoU。

   

   ```
   gt_argmax_ious = ious.argmax(axis=0)
   ```

   找到每个真实框的最大 IoU 的索引，即对于每个真实框，识别出与哪个锚点重叠最多。`gt_argmax_ious` 是一个数组，其中每个元素对应一个真实框，并包含与之 IoU 最高的锚点的索引。

   

   ```
   gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
   ```

   这行代码创建了一个数组，包含每个真实框的最大 IoU 值。 `np.arange(ious.shape[1])` 生成每个真实框的索引序列。 `ious[gt_argmax_ious, np.arange(ious.shape[1])]` 使用这些索引来选择每个真实框与其对应的最佳匹配锚点（由 `gt_argmax_ious` 确定）的 IoU。

   

   ```
   gt_argmax_ious = np.where(ious == gt_max_ious)[0]
   ```

   这行代码找到所有与任何真实框具有最高 IoU 的锚点的索引。 `np.where(ious == gt_max_ious)` 查找 `ious` 中 IoU 等于任何真实框的最高 IoU 的位置。这可能包括单个真实框的多个锚点（在平局的情况下）。 最后的 `[0]` 用于从结果中提取锚点索引。

   

   所以这些代码用来确定每个真实框的最高 IoU 的锚点，反之亦然。这用于给锚点分配标签。

   


总结来说，`AnchorTargetCreator` 负责为图像中的锚点分配真实标签和回归目标。它在训练 Faster R-CNN 的 RPN 中发挥着至关重要的作用，确保网络学习提出与图像中实际对象准确重叠的区域。
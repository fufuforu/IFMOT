import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from typing import List, Optional, Union
from PIL import Image
import copy
class ImageVisualizer:
    def __init__(self, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225], denormalized=False):
        """
        mean, std: 用于反归一化，如果图像没归一化过可以不填
        """
        self.mean = mean
        self.std = std
        self.id_color_map = {}  # 记录每个id对应的颜色
        self.denormalized = denormalized
        # self.fig = None
        # self.ax = None

    def _denormalize(self, img):
        """反归一化图像"""
        #import pdb;pdb.set_trace()
        if self.mean is not None and self.std is not None:
            # 确保mean/std是numpy数组
            mean = np.array(self.mean).reshape(1, 1, -1)
            std = np.array(self.std).reshape(1, 1, -1)
            img = img * std + mean
            img = np.clip(img, 0, 1)
        return img

    def _prepare_img(self, img):
        """把 Tensor / PIL Image 转成可以 imshow 的格式"""
        if isinstance(img, torch.Tensor):
            # 处理批次维度
            if img.dim() == 4:
                img = img[0]
            img = img.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
            if self.denormalized:
                img = self._denormalize(img)
        elif isinstance(img, Image.Image):  # 处理 PIL Image
            img = np.array(img)  # 转成 np.array 格式
        else:
            raise TypeError(f"不支持的图像类型: {type(img)}")
        return img

    def _get_color_for_id(self, obj_id):
        """为每个id分配固定的颜色"""
        #import pdb;pdb.set_trace()
        if obj_id not in self.id_color_map:
            random.seed(int(obj_id))  # 让同一个id每次颜色一样
            color = (random.random(), random.random(), random.random())  # RGB三通道随机
            self.id_color_map[obj_id] = color
        return self.id_color_map[obj_id]

    def _draw_instances(self, ax, instances, box_format=None, img_shape=None, affset_x=None,affset_y=None):
        """
        在matplotlib轴上画方框和ID
        instances: 包含 boxes, obj_ids
        box_format: 指定边界框格式 'cxcywh'或'xyxy'
        """
        # import pdb;pdb.set_trace()
        if isinstance(instances, dict):
            if 'boxes' in instances:
                boxes = instances['boxes'].detach().cpu()
            elif 'pred_boxes' in instances:
                boxes = instances['pred_boxes'].detach().cpu()
            else:
                raise KeyError("instances 中未找到 'boxes' 或 'pred_boxes' 键")
            obj_ids = None
            for k in ('obj_ids', 'ids','obj_idxes'):
                if k in instances:
                    obj_ids = instances[k].detach().cpu()
        elif type(instances) == torch.Tensor: #proposals
            boxes = instances[:,0:4].detach().cpu()
            obj_ids = torch.full((boxes.shape[0],), -1)
        else:
            if hasattr(instances, 'boxes'):
                boxes = instances.boxes.detach().cpu()
            elif hasattr(instances, 'pred_boxes'):
                boxes = instances.pred_boxes.detach().cpu()
            else:
                raise AttributeError("instances 中未找到 'boxes' 或 'pred_boxes' 属性")
            
            if hasattr(instances, 'obj_ids'):
                obj_ids = instances.obj_ids.detach().cpu()
            elif hasattr(instances, 'obj_idxes'):
                obj_ids = instances.obj_idxes.detach().cpu()
            else:
                raise AttributeError("instances 中未找到 'obj_ids' 或 'obj_idxes' 属性")
        
        # 获取图像尺寸
        if hasattr(instances, '_image_size'):
            height, width = instances._image_size
        elif isinstance(instances, dict) and 'size' in instances:
            height, width = instances['size']
        else:
            # 如果没有_image_size属性，尝试从ax获取
            height, width = ax.get_images()[0].get_size()[1], ax.get_images()[0].get_size()[0]
        if img_shape is not None:
            height,width = img_shape
        # import pdb;pdb.set_trace()
        # boxes = boxes.detach().cpu()
        # obj_ids = obj_ids.detach().cpu()
        def is_normalized(boxes):
                # 统计超过1的元素占比，如果大部分都大于1，说明不是归一化
                boxes = np.array(boxes)
                return np.all(boxes >= 0) and np.all(boxes <= 1)
        flag_normalized =  self.denormalized
        # import pdb;pdb.set_trace()
        for box, obj_id in zip(boxes, obj_ids):
            box_i = copy.deepcopy(box)
            if obj_id == -1:
                continue
            if box_format == 'cxcywh':
                cx, cy, w, h = box_i  # cx, cy, w, h 格式
                if flag_normalized:
                    cx *= width
                    cy *= height
                    w *= width
                    h *= height
                x1 = cx - w / 2
                y1 = cy - h / 2
            elif box_format == 'xyxy':
                x1, y1, x2, y2 = box_i
                if flag_normalized:
                    x1 *= width
                    y1 *= height
                    x2 *= width
                    y2 *= height
                w = x2 - x1
                h = y2 - y1
            else:
                raise ValueError(f"不支持的box_format: {box_format}")

            color = self._get_color_for_id(obj_id.item())  # 根据id拿颜色
            # import pdb;pdb.set_trace()
            # if x1 < 0 or y1 < 0 or x1 + w > width or y1+h > height:
            #     print("矩形越界了")
            #     import pdb;pdb.set_trace()
            # 画矩形
            if affset_x is not None:
                x1 += affset_x
                y1 += affset_y
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                     edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # 在方框上写ID
            ax.text(x1, y1 - 5, str(int(obj_id.item())),
                    fontsize=8, color='white',
                    bbox=dict(facecolor=color, alpha=0.6, pad=1))

    def _draw_bboxes_with_ids_scores(self, ax, bboxes: List[List[float]], ids: Optional[List[int]] = None,scores: Optional[List[int]] = None, color='red'):
        """
        绘制XYXY格式的边界框和ID
        bboxes: [[x1, y1, x2, y2], ...] 绝对坐标
        ids: 可选的ID列表
        """
        for i, box in enumerate(bboxes):
            #import pdb;pdb.set_trace()
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # 获取颜色：如果有ID则使用ID对应颜色，否则用红色
            
            if ids is not None and i < len(ids) and color is None:
                c = self._get_color_for_id(ids[i])
            else:
                c = color
            # import pdb;pdb.set_trace()
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=c,
                facecolor='none'
            )
            #import pdb;pdb.set_trace()
            ax.add_patch(rect)

            # 如果有ID，则在左上角显示
            txt = ''
            if ids is not None and i < len(ids):
                txt += f"ID:{ids[i]}"
            if scores is not None and i < len(scores):
                if torch.is_tensor(scores[i]):
                    txt += f"  S:{round(scores[i].item(),2)}"
                else:
                    txt += f"  S:{round(scores[i],2)}"
            
            if txt:
                ax.text(x1, y1 - 5, txt,
                        fontsize=8, color='white',
                        bbox=dict(facecolor=c, alpha=0.6, pad=1))
            

    def show(self, imgs, instances=None, figsize=(10, 19), title=None, box_format=None, img_shape = None):
        """
        显示单张或多张图像，并可选画框。
        
        imgs: 单张图片，或图片列表（支持 torch.Tensor 或 PIL.Image）
        instances: 对应的Instance对象或列表（可选）
        box_format: 'cxcywh' 或 'xyxy'
        """
        
        # 转为列表统一处理
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        if instances is None:
            instances = [None] * len(imgs)
        elif not isinstance(instances, (list, tuple)):
            instances = [instances]

        pad = 300
        
        # 计算子图排布
        n = len(imgs)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        # import pdb;pdb.set_trace()
        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0]*cols, figsize[1]*rows), facecolor='black')
        axes = np.array(axes).reshape(-1)  # 保证是1维数组，方便处理

        for idx, (img, inst) in enumerate(zip(imgs, instances)):
            if img_shape == None:
                image_size =  img.shape[1:3]
            else:
                image_size = img_shape
            canvas_size = (image_size[0]+pad,image_size[1]+pad)
            canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255 
            # instance = copy.deepcopy(inst)
            img = self._prepare_img(img)
               # 计算居中位置
            start_y = (canvas_size[0] - image_size[0]) // 2
            start_x = (canvas_size[1] - image_size[1]) // 2
            # import pdb;pdb.set_trace()
            # 将图像粘贴到画布中间
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            canvas[start_y:start_y+image_size[0], start_x:start_x+image_size[1]] = img
            # import pdb;pdb.set_trace()
            ax = axes[idx]
            # ax.set_facecolor('black')
            ax.imshow(canvas, aspect='auto')
            # import pdb;pdb.set_trace()
            if inst is not None:
                self._draw_instances(ax, inst, box_format=box_format, img_shape=image_size,affset_x=pad/2,affset_y=pad/2)
                
            if title:
                ax.set_title(f"{title} #{idx}")
            ax.axis('off')
            # 添加边框
            h, w = img.shape[:2]
            border = patches.Rectangle(
                (pad/2, pad/2), w, h,
                linewidth=2, edgecolor='black', facecolor='none'
            )
            ax.add_patch(border)

            ax.axis('off')
        # 清除多余子图
        for ax in axes[n:]:
            ax.axis('off')

        # plt.tight_layout()
        plt.show()
        
    def show_batch(self, imgs, instances_list=None, ncols=3, figsize=(15, 10), titles=None, box_format='cxcywh'):
        """
        一次性画一批图，按网格排好，并在图上标索引号
        imgs: list of images
        instances_list: list of Instances，或 None
        ncols: 每行最多显示多少列
        figsize: 整体figure大小
        titles: list of title字符串，可选
        box_format: 指定边界框格式 'cxcywh'或'xyxy'
        """
        n_images = len(imgs)
        nrows = (n_images + ncols - 1) // ncols  # 计算需要几行
        
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        axs = np.array(axs).reshape(-1)  # 平铺成一维，方便访问

        for idx, img in enumerate(imgs):
            ax = axs[idx]
            img_prepared = self._prepare_img(img)
            ax.imshow(img_prepared)

            # 如果有实例，就画上方框
            if instances_list is not None and idx < len(instances_list) and instances_list[idx] is not None:
                self._draw_instances(ax, instances_list[idx], box_format=box_format)

            # 设置标题：可以用传入的，也可以默认用index
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=10)
            else:
                ax.set_title(f"index={idx}", fontsize=10)

            ax.axis('off')

        # 把多余的子图关掉
        for idx in range(n_images, len(axs)):
            axs[idx].axis('off')

        plt.tight_layout()
        plt.show()

    def proposals_to_boxes(self, proposals, img_size=(1080, 1920)):
        """
        将 [cx, cy, w, h, score] 格式的 proposals 转换为 [x1, y1, x2, y2] 形式，
        并根据 img_size 反归一化到像素坐标。
        
        参数:
            proposals (Tensor): shape [N, 5]，数据为 [cx, cy, w, h, score]
            img_size (tuple): 原始图像尺寸 (height, width)

        返回:
            boxes (Tensor): shape [N, 4]，数据为 [x1, y1, x2, y2]
        """
        height, width = img_size
        device = proposals.device

        # 提取 cx, cy, w, h
        cx = proposals[:, 0]
        cy = proposals[:, 1]
        w = proposals[:, 2]
        h = proposals[:, 3]

        # 计算 x1, y1, x2, y2
        x1 = (cx - w / 2) * width
        y1 = (cy - h / 2) * height
        x2 = (cx + w / 2) * width
        y2 = (cy + h / 2) * height

        # 拼接成 boxes
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        return boxes.to(device)

    def show_bbox(self, img, bbox=None,ori_img_size=None,
                proposals=None, ids=None, scores=None,title=None, duration=0.001,color=None):
        """
        在同一个窗口中连续绘制图像和边界框，用于动态显示
        
        参数:
            img (np.ndarray or torch.Tensor): 输入图像 (H, W, C)
            bbox (list of list): 边界框列表 [[x1, y1, x2, y2], ...]
            ids (list): 每个框对应的 ID 或标签
            title (str): 图像标题
            duration (float): 每帧停留时间（秒）
        """
        if proposals is not None:
            bbox = self.proposals_to_boxes(proposals=proposals,img_size=ori_img_size)
        # 第一次调用时初始化绘图窗口
        if not hasattr(self, 'fig') or not hasattr(self, 'ax'):
            self.fig, self.ax = plt.subplots(1, figsize=(8, 6))
            plt.ion()  # 开启交互模式
            plt.show()

        # 转换图像格式（如需要）
        img = self._prepare_img(img)

        # 清除上一帧内容
      
        self.ax.clear()
        self.ax.imshow(img)

        # 绘制边界框
        if bbox is not None:
            self._draw_bboxes_with_ids_scores(self.ax, bbox, ids,scores, color)

        # 设置标题
        if title:
            self.ax.set_title(title)

        self.ax.axis('off')

        # 刷新图像
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # 控制刷新间隔
        plt.pause(duration)

    def save(self, img, save_path, instances=None, figsize=(8, 10), title=None, box_format='cxcywh'):
        """
        保存图像并可选画框
        """
        img = self._prepare_img(img)

        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(img)

        if instances is not None:
            self._draw_instances(ax, instances, box_format=box_format)

        if title:
            plt.title(title)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def save_bbox(self, img, save_path, bbox: List[List[float]], 
                 ids: Optional[List[int]] = None, figsize=(8, 10), title=None):
        """
        保存带有XYXY边界框和ID的图像
        """
        img = self._prepare_img(img)

        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(img)
        
        if bbox is not None:
            self._draw_bboxes_with_ids(ax, bbox, ids)

        if title:
            plt.title(title)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

# update 2025 07 19
import warnings
from d2l import paddle as d2l

warnings.filterwarnings("ignore")
import paddle

img = d2l.plt.imread('F:/code/deeplearningfollowlimu/d2l-zh/pytorch/img/catdog.jpg')
h, w = img.shape[:2]
print(h, w)


#在特征图（fmap）上生成锚框（anchors），每个单位（像素）作为锚框的中心
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = paddle.zeros(shape=[1, 10, fmap_h, fmap_w])
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = paddle.to_tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
    

#探测小目标
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
d2l.plt.show()

#将特征图的高度和宽度减半，然后使用较大的锚框来检测较大的目标
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
d2l.plt.show()


#高宽减半，尺度增加到0.8
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
d2l.plt.show()
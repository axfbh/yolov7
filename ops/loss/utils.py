def smooth_BCE(eps=0.1):
    """用在ComputeLoss类中
    标签平滑操作  [1, 0]  =>  [0.95, 0.05]
    https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    :params eps: 平滑参数
    :return positive, negative label smoothing BCE targets  两个值分别代表正样本和负样本的标签取值
            原先的正样本=1 负样本=0 改为 正样本=1.0 - 0.5 * eps  负样本=0.5 * eps
    """
    return 1.0 - 0.5 * eps, 0.5 * eps

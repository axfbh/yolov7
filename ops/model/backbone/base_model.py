class BasicModel:
    def __init__(self):
        pass

    def activations_hook(self, grad):
        # 获取梯度的钩子
        self.gradients = grad

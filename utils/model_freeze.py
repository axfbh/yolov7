progress_plot = {
    0: '',
    1: '▏',
    2: '▏',
    3: '▎',
    4: '▍',
    5: '▌',
    6: '▋',
    7: '▋',
    8: '▊',
    9: '▉',
    10: '█',
}


def progress_bar(titles, start, end):
    print("", end="\n")
    prop_start = start / end * 100
    print(titles + ": {:.2f}%|".format(prop_start),
          '█' * int(prop_start / 10),
          progress_plot[int(prop_start / 10)],
          ' ' * (int((100 - prop_start) / 10)),
          '| {}/{}'.format(start, end), sep='', end="\n")


class FreezeLayer:
    def __init__(self,
                 layer,
                 freeze_start,
                 freeze_end,
                 last_epoch=0):
        """

        :param layer: 冻结层
        :param freeze_start: 冻结开始时间
        :param freeze_end: 冻结结束时间
        """

        self.last_epoch = last_epoch
        if freeze_end < freeze_start:
            raise ValueError('freeze_start must be greater than freeze_end.')
        self.freeze_end = freeze_end
        self.layer = layer
        self.freeze_start = freeze_start
        if self.freeze_start == -1 or self.freeze_end == -1:
            print("不执行冻结和解冻操作")
        else:
            if self.freeze_start <= (last_epoch + 1) <= self.freeze_end:
                self.freeze_status = 1  # 冻结状态 0: 未开始， 1：正在冻结，-1：结束冻结
            elif (last_epoch + 1) > self.freeze_end:
                self.freeze_status = -1
            else:
                self.freeze_status = 0

    def step(self, epoch):

        if self.freeze_start != -1 or self.freeze_end != -1:
            if self.freeze_status == 0:

                if self.freeze_start >= epoch:
                    progress_bar('开始冻结 layer 还剩', epoch, self.freeze_start)
                else:
                    self.layer.requires_grad_(False)
                    self.freeze_status = 1

            if self.freeze_status == 1:

                if self.freeze_end >= epoch:
                    progress_bar('结束冻结 layer 还剩', epoch, self.freeze_end)
                else:
                    self.layer.requires_grad_(True)
                    self.freeze_status = -1

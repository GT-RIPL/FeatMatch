import bisect


class SupConvScheduler:
    def __init__(self, optimizer, pretrain_iters, cycle_iters, end_iters, max_lr, max_mom, last_iter=-1):
        self.optimizer = optimizer
        self.milestones = [pretrain_iters, pretrain_iters+cycle_iters, pretrain_iters+2*cycle_iters]
        self.max_lr = max_lr
        self.start_lr = max_lr/100.
        self.base_lr = max_lr/10.
        self.end_lr = self.base_lr/1000.
        self.start_slope = (self.base_lr - self.start_lr) / pretrain_iters \
            if pretrain_iters != 0 and pretrain_iters is not None else None
        self.ramp_slope = (max_lr - self.base_lr) / cycle_iters
        self.end_slope = (self.base_lr - self.end_lr) / end_iters

        self.max_mom = max_mom
        self.min_mom = max_mom - 0.1
        self.mom_slope = (max_mom - self.min_mom) / cycle_iters

        self.last_iter = last_iter
        self.step()

    def compute_params(self, current_iter):
        stage = bisect.bisect_right(self.milestones, current_iter)

        if stage == 0:  # pretrain
            lr = self.start_lr + current_iter * self.start_slope
            mom = self.max_mom
        elif stage == 1:  # ramp-up
            lr = self.base_lr + (current_iter - self.milestones[0]) * self.ramp_slope
            mom = self.max_mom - (current_iter - self.milestones[0]) * self.mom_slope
        elif stage == 2:  # ramp-down
            lr = self.max_lr - (current_iter - self.milestones[1]) * self.ramp_slope
            mom = self.min_mom + (current_iter - self.milestones[1]) * self.mom_slope
        elif stage == 3:  # ending
            lr = self.base_lr - (current_iter - self.milestones[2]) * self.end_slope
            mom = self.max_mom
        else:
            raise ValueError

        return lr, mom

    def step(self, current_iter=None):
        if current_iter is None:
            current_iter = self.last_iter + 1
        lr, mom = self.compute_params(current_iter)
        self.optimizer.param_groups[0]['lr'] = lr
        self.optimizer.param_groups[0]['mom'] = mom

        self.last_iter = current_iter

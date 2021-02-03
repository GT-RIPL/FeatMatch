import numpy as np
import git
from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter
import pprint
import shutil
from termcolor import cprint
import torch
from torch import optim
from torch.cuda import amp
from util import scheduler, metric, misc


class Trainer(object):
    """
    The base class for training a model. This class implements the fundamental building blocks of the training pipeline.
    """
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.root_dir = Path(args.save_root)
        self.record()
        self.scaler = amp.GradScaler(enabled=args.amp)
        self.init_rand_seed()
        self.default_device = self.init_device()

        self.dataloader_train, self.dataloader_val, self.dataloader_test, self.Ttrain, self.Tval, self.Tnorm = \
            self.init_dataloader()
        self.model = self.init_model().to(self.default_device)
        self.optimizer = self.init_optimizer()
        self.scheduler, self.total_iters = self.init_scheduler()

        self.logger_train = SummaryWriter(logdir=self.root_dir/'log'/'train')
        self.logger_val = SummaryWriter(logdir=self.root_dir/'log'/'val')
        self.metric = metric.AccMetric()

        """
        Append the name of additional variables you want to record.
        :param state_objs: objects that have a member function of state_dict() such as pytorch model, optimizer, etc.
        :param attr_objs: objects that do not have a member function of state_dict() such as np array, torch tensor, etc.
        If either state_objs or attr_objs is changed in subclass' init function, don't forget to call self.load(args.mode) again.
        """
        self.state_objs = ['model', 'optimizer']
        self.attr_objs = []
        self.curr_iter, self.curr_result, self.best_result = self.load(args.mode)

    def record(self):
        """
        record some info of the experiment for reproducibility: (1) commit, (2) config, (3) running args
        """
        with open(self.root_dir/'record.txt', 'a+') as file:
            pp = pprint.PrettyPrinter(stream=file)
            pp.pprint(('commit', git.Repo().head.object.hexsha))
            pp.pprint(('config', self.config))
            pp.pprint(('arg', vars(self.args)))
            pp.pprint('')

    def init_rand_seed(self):
        """
        initialize random seeds of numpy and PyTorch for reproducibility
        :return: None
        """
        np.random.seed(self.args.rand_seed)
        torch.manual_seed(self.args.rand_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def init_device(self):
        """
        Initialize default running devices.
        :return: torch.device. default device
        """
        if self.args.devices is not None:
            default_device = torch.device('cuda', self.args.devices[0])
            torch.cuda.set_device(default_device)
        else:
            default_device = torch.device('cpu')

        return default_device

    def init_dataloader(self):
        """
        Create (1) dataloader of train, val, and test sets,
        (2) data augmentation of train and val (test),
        and (3) data normalizer.
        :return: dataloader_train, dataloader_val, dataloader_test, Ttrain, Tval, Tnorm
        """
        raise NotImplementedError

    def init_model(self):
        """
        Initialize deep model of type torch.nn.Module
        :return model: a subclass of torch.nn.Module
        """
        raise NotImplementedError

    def init_optimizer(self):
        """
        Initialize optimizer
        :return: torch.optim.Optimizer object. optimizer
        """
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.config['train']['lr'],
                              momentum=self.config['train']['mom'],
                              weight_decay=self.config['train']['weight_decay'],
                              nesterov=True)

        return optimizer

    def init_scheduler(self):
        """
        Initialize learning rate scheduler
        :return: learning rate scheduler
        :return: int. total training iterations
        """
        sc = scheduler.SupConvScheduler(optimizer=self.optimizer,
                                        pretrain_iters=self.config['train']['pretrain_iters'],
                                        cycle_iters=self.config['train']['cycle_iters'],
                                        end_iters=self.config['train']['end_iters'],
                                        max_lr=self.config['train']['lr'],
                                        max_mom=self.config['train']['mom'])

        total_iters = self.config['train']['pretrain_iters'] + \
                      2 * self.config['train']['cycle_iters'] + \
                      self.config['train']['end_iters']

        return sc, total_iters

    def save(self):
        # update best result
        better_result = False
        if self.curr_result > self.best_result:
            (self.root_dir / f'{self.best_result*100:.2f}').unlink()
            (self.root_dir / f'{self.curr_result*100:.2f}').touch()
            self.best_result = float(self.curr_result)
            better_result = True

        # save latest checkpoint
        ckpt = dict()
        for obj in self.state_objs:
            ckpt[obj] = getattr(self, obj).state_dict()
        for obj in self.attr_objs:
            ckpt[obj] = getattr(self, obj)
        ckpt["amp"] = self.scaler.state_dict()
        ckpt['last_iter'] = self.curr_iter
        ckpt['curr_result'] = self.curr_result
        ckpt['best_result'] = self.best_result
        torch.save(ckpt, self.root_dir / 'curr_ckpt')

        # save result
        with open(self.root_dir / 'results.txt', 'a+') as file:
            file.write(f'{self.curr_result}\n')

        # save best checkpoint
        if better_result:
            print(f'[{self.curr_iter}] Save best model with result = {self.best_result * 100:.2f} %')
            shutil.copyfile(self.root_dir / 'curr_ckpt', self.root_dir / 'best_ckpt')

        return

    def load(self, mode):
        """
        load from checkpoint. supported modes: new, resume, test
        :return curr_iter: int. current iteration.
        :return curr_result: float. current result.
        :return best_result: float. best result so far.
        """
        if mode == 'new':
            curr_iter, curr_result, best_result = 0, 0.0, 0.0
        else:
            if mode == 'resume':
                ckpt_file = self.root_dir / 'curr_ckpt'
            elif mode == 'test':
                ckpt_file = self.root_dir / 'best_ckpt'
            else:
                raise KeyError
            checkpoint = torch.load(ckpt_file, map_location=self.default_device)

            for obj in self.state_objs:
                getattr(self, obj).load_state_dict(checkpoint[obj])
            for obj in self.attr_objs:
                setattr(self, obj, checkpoint[obj])
            self.scaler.load_state_dict(checkpoint["amp"])
            curr_iter = checkpoint['last_iter'] + 1
            curr_result = checkpoint['curr_result']
            best_result = checkpoint['best_result']
            cprint(f'Load from iteration [{checkpoint["last_iter"]}], '
                   f'with current result = {checkpoint["curr_result"]*100:.2f} %, '
                   f'best result = {checkpoint["best_result"]*100:.2f} %',
                   color='blue', attrs=['bold'])

        (self.root_dir / f'{best_result * 100:.2f}').touch()
        return curr_iter, curr_result, best_result

    def forward_train(self, data):
        """
        Forward pass for training loop
        :param data: batch of data from the DataLoader
        :return loss: torch float tensor of scalar. total loss for optimization
        :return results: dict of results. key: str. name; val: float. value
        """
        raise NotImplementedError

    def forward_eval(self, data):
        """
        forward pass for evaluation loop
        :param data: batch of data from the DataLoader
        :return loss: torch float tensor of scalar. total loss for optimization
        :return results: dict of results. key: str. name; val: float. value
        """
        raise NotImplementedError

    def train(self):
        """
        Starting point of training and validation.
        :return best_result: best validation result.
        """
        pbar = tqdm(total=self.total_iters, initial=self.curr_iter, dynamic_ncols=True)

        while self.curr_iter < self.total_iters:
            for _, data in enumerate(self.dataloader_train):
                if not (self.curr_iter < self.total_iters):
                    break

                # train update
                self.scheduler.step(self.curr_iter)
                self.optimizer.zero_grad()

                with amp.autocast(enabled=self.args.amp):
                    loss, results = self.forward_train(data)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # training log
                curr_acc = self.metric.record(results.pop('y_true'), results.pop('y_pred'), clear=True)
                self.logger_train.add_scalar('acc/', curr_acc, self.curr_iter)
                for c, results_c in results.items():
                    for k, v in results_c.items():
                        self.logger_train.add_scalar(f'{c}/{k}', v, self.curr_iter)

                # evaluate trained model
                if (self.curr_iter + 1) % self.config['train']['update_interval'] == 0:
                    val_iters = np.linspace(self.curr_iter + 1 - self.config['train']['update_interval'],
                                            self.curr_iter + 1, len(self.dataloader_val), endpoint=False, dtype=int)
                    with torch.no_grad():
                        for i, data in enumerate(self.dataloader_val):
                            with amp.autocast(enabled=self.args.amp):
                                results = self.forward_eval(data)
                            self.metric.record(results.pop('y_true'), results.pop('y_pred'), clear=False)
                            for c, results_c in results.items():
                                for k, v in results_c.items():
                                    self.logger_val.add_scalar(f'{c}/{k}', v, val_iters[i])
                    self.curr_result = self.metric.average(clear=True)
                    self.logger_val.add_scalar('acc/', self.curr_result, self.curr_iter)
                    self.save()

                # update training status
                self.curr_iter += 1
                pbar.update()

        pbar.close()
        self.logger_train.export_scalars_to_json(self.logger_train.logdir/'train.json')
        self.logger_val.export_scalars_to_json(self.logger_val.logdir/'val.json')

        return self.best_result

    def test(self):
        """
        Starting point of testing on the held-out test set
        :return val_acc: validation accuracy
        :return test_acc: testing accuracy
        """
        # reload the weights of the best model on val set so far
        self.curr_iter, _, val_acc = self.load('test')

        with torch.no_grad():
            for _, data in enumerate(self.dataloader_test):
                with amp.autocast(enabled=self.args.amp):
                    results = self.forward_eval(data)
                self.metric.record(results['y_true'], results['y_pred'], clear=False)
        test_acc = self.metric.average(clear=True)

        return val_acc, test_acc

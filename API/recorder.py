import numpy as np
import torch

class Recorder:
    '''
    保存模型的中间结果
    '''
    def __init__(self, verbose=False, delta=0):
        # verbose 打开时会打印 loss 的详细信息
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        '''
        该特殊方法可使类实例像普通函数那样调用
        function:
        如果 val_loss 降低,则保存模型,并更新 self.val_loss_min
        '''
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss
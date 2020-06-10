from tensorboardX import SummaryWriter

class Logger(SummaryWriter):

    def __init__(self, logdir):
        super().__init__(logdir)


    def logging(self, epoch, train_loss, valid_loss, lr):

        self.add_scalar('train_loss', train_loss, epoch)
        self.add_scalar('valid_loss', valid_loss, epoch)
        self.add_scalar('learning_rate', lr, epoch)

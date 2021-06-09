from data import *
import os
os.chdir("models")
from dann import *
from svm_base import *

ds = SeedDataset() #initiatlize data
flag_train = False
flag_logging=False


def train_model(subject = 0):
    trainer = Trainer(max_epochs=4, progress_bar_refresh_rate=20, fast_dev_run=False,
                    #tpu_cores=8,
                    #   ,val_check_interval=0.25
                    gpus=1)
    sdm = SeedDataModule(ds,batch_size=64,leave_out_idx=subject)
    m = DANN(lr1=1e-4,lr2=1e-4)
    trainer.fit(m,sdm)
    trainer.test(m,datamodule=sdm)
    results= m.test_acc
    model_accuracy_same_lr.append(results)
    
    

def train_model(subject = 0):
    trainer = Trainer(max_epochs=4, progress_bar_refresh_rate=20, fast_dev_run=False,
                    gpus=1)
    sdm = SeedDataModule(ds,batch_size=64,leave_out_idx=subject)
    m = DANN(lr1=1e-3,lr2=1e-15)
    trainer.fit(m,sdm)
    trainer.test(m,datamodule=sdm)
    results= m.test_acc
    model_accuracy_seperate_lr.append(results)

def train_model(subject = 0):
    trainer = Trainer(max_epochs=5, progress_bar_refresh_rate=20, fast_dev_run=False,
                    #tpu_cores=8,
                    #   ,val_check_interval=0.25
                    gpus=1)
    m = SVM_linear(lr1=1e-3)
    sdm = SeedDataModule(ds,batch_size=64,leave_out_idx=subject)
    trainer.fit(m,sdm)
    trainer.test(m,datamodule=sdm)
    results= m.test_acc
    model_accuracy_svm.append(results)

if __name__ == "__main__":
    
    model_accuracy_svm = list()

    for i in range(5):
        train_model(i)

    print(model_accuracy_svm)
    print(f'mean: {torch.mean(torch.stack(model_accuracy_svm))}')
    print(f'std: {torch.std(torch.stack(model_accuracy_svm))}')
    
    
    model_accuracy_seperate_lr =list()

    # Training loop (with same learning rate)
    for i in range(5):
        train_model(i)
    print(model_accuracy_seperate_lr)
    print(f'mean: {torch.mean(torch.stack(model_accuracy_seperate_lr))}')
    print(f'std: {torch.std(torch.stack(model_accuracy_seperate_lr))}')

    # Training loop (with same learning rate)
    model_accuracy_same_lr =list()
    for i in range(5):
        train_model(i)

    print(model_accuracy_same_lr)
    print(f'mean: {torch.mean(torch.stack(model_accuracy_same_lr))}')
    print(f'std: {torch.std(torch.stack(model_accuracy_same_lr))}')

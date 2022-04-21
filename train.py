import torch, config 

from training.training_utils import training_epochs
from training.validation_utils import validation_epochs
## Load data
from data.pytorch_dataset import DatasetGeneration, pytorch_dataset
from torch.utils.data import DataLoader
from eval_with_extract import HPatchesExtractAndEvaluate
## Network architecture & loss / optimizer
from training.model.REKD import REKD
import torch.optim as optim
## Logging
from utils.logger import Logger, Recorder



def init_dataloader(args, logtime):
    print('Start training REKD Architecture')

    dataset_generation = DatasetGeneration(args)

    dataset_train = pytorch_dataset(dataset_generation.get_training_data(), mode='train')
    dataset_val = pytorch_dataset(dataset_generation.get_validation_data(), mode='val')

    dataloader_train  = DataLoader(dataset_train, batch_size=args.batch_size, drop_last=True, shuffle=True)
    dataloader_val  = DataLoader(dataset_val, batch_size=1, drop_last=True, shuffle=False)

    hpatches_val = HPatchesExtractAndEvaluate(args, args.exp_name + logtime, split=args.eval_split)

    return dataloader_train, dataloader_val, hpatches_val

def init_model(args, device, verbose=False):

    model = REKD(args, device)
    model = model.to(device) ## use GPU

    if args.load_dir != '':
        model.load_state_dict(torch.load(args.load_dir))  ## Load the PyTorch learnable model parameters.
        Logger.info("Model paramter : {} is loaded.".format( args.load_dir ))
    
    # print("MSIP hyperparameters : ", args.MSIP_sizes, args.MSIP_factor_loss)
    print("Succeed to initialize model group {} with layer channel {} {} {}.".format(args.group_size, args.dim_first, args.dim_second, args.dim_third))
    # count_model_parameters(model)

    return model


def validate(epoch, dataloader_val, model, total_loss, nms_size, device, recorder, model_save=True):
    with torch.no_grad():
        model.eval()
        rep_s, ori_acc, ori_approx_acc = validation_epochs(epoch, dataloader_val, model, nms_size, device, num_points=25)
        is_best = recorder.update(epoch, {'repeatability':rep_s, 'ori_acc':ori_acc, 'ori_apx_acc':ori_approx_acc, 'total_loss': total_loss})

        if model_save:
            Logger.save_model(model, epoch, rep_s)

            if is_best:
                Logger.save_best_model(model, rep_s)

                hpatches_val.run(model)
        model.train()

    return recorder

def finish(recorder, logpath):
    best_epoch, result = recorder.get_results()

    msg = "Best results at epoch {}   ".format(best_epoch)
    for k, v in result.items():
        msg += "{} : {:.2f}    ".format(k.upper(), v)
    Logger.info(msg + " at {}".format(logpath))



if __name__ == "__main__":
    
    args = config.get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logtime, logpath = Logger.initialize(args)   
    dataloader_train, dataloader_val, hpatches_val = init_dataloader(args, logtime)
    model = init_model(args, device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_initial_learning_rate, weight_decay=0.1) 
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5) 

    recorder = Recorder()

    validate(0, dataloader_val, model, float("inf"), args.nms_size, device, recorder, False)    

    for epoch in range(1, args.num_epochs+1):
        keynet_loss, ori_loss, total_loss = training_epochs(epoch, dataloader_train, model, optimizer, args, device)

        validate(epoch, dataloader_val, model, total_loss, args.nms_size, device, recorder, True)   

        if epoch == 10:
            scheduler.step()

    finish(recorder, logpath)

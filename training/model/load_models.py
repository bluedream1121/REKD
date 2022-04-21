import torch
from .REKD import REKD
from .hardnet_pytorch import HardNet
from .sosnet_pytorch import SOSNet32x32
from .hynet_pytorch import HyNet

from collections import OrderedDict

def load_detector(args, device):
    model1 = None
     
    if args.load_dir != '':
        args.group_size, args.dim_first, args.dim_second, args.dim_third = model_parsing(args)
        model1 = REKD(args, device)
        model1.load_state_dict(torch.load(args.load_dir))
        model1.export()
        model1.eval()
        model1.to(device) ## use GPU

    return model1

def load_descriptor(args, device):
    # Define Pytorch HardNet
    if args.descriptor == "hardnet":
        model2 = HardNet()
        checkpoint = torch.load('trained_models/pretrained_nets/HardNet++.pth')
        model2.load_state_dict(checkpoint['state_dict'])
        model2.eval()
        model2.to(device)        
    # # Define Pytorch SOSNet
    elif args.descriptor == "sosnet":
        model2 = SOSNet32x32()
        net_name = "liberty"
        model2.load_state_dict(torch.load(os.path.join('trained_models/pretrained_nets/sosnet-weights',"sosnet-32x32-"+net_name+".pth")))
        model2.eval()
        model2.to(device)
    # # Define Pytorch HyNet
    elif args.descriptor == "hynet":
        model2 = HyNet()
        model2.load_state_dict(torch.load(os.path.join('trained_models/pretrained_nets/HyNet_LIB.pth')))
        model2.eval()
        model2.to(device)
    else:
        raise NotImplementedError
        
    return model2



## Load our model
def model_parsing(args):
    group_size = args.load_dir.split('_group')[1].split('_')[0]
    dim_first = args.load_dir.split('_f')[1].split('_')[0]
    dim_second = args.load_dir.split('_s')[1].split('_')[0]
    dim_third = args.load_dir.split('_t')[1].split('.log')[0]        

    return int(group_size), int(dim_first), int(dim_second), int(dim_third)




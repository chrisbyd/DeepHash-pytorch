import DFH
import DPN
import CNNH
import HashNet
import GreedyHash
import PCDH
import ADSH
import sys
import CSQ
from utils.tools import *
import torch
import LCDSH
import IDHN
import DSH
import DPSH
import DCH
import DSDH
modules = {'DFH': DFH, 'DPN': DPN, 'CNNH': CNNH, 'HashNet': HashNet, 'GreedyHash': GreedyHash, 'PCDH': PCDH, 'ADSH' :ADSH, 'CSQ': CSQ,
           'LCDSH' : LCDSH, 'IDHN': IDHN, 'DSH': DSH, "DPSH": DPSH , 'DCH':DCH , "DSDH":DSDH}

epoch = int(sys.argv[2])
model_name = sys.argv[1]
bit_list = sys.argv[3]
batch_size = sys.argv[4]
dataset = sys.argv[5]
gpu_device = sys.argv[6]
machine_name = sys.argv[7]
test_map = sys.argv[8]
model = modules[model_name]
test_batch_size = sys.argv[9]
#dataset cifar10 imagenet nuswide81

if __name__ == '__main__':
    bit_list = bit_list[1:-1]
    bit_list = bit_list.split(',')
    bit_list = [int(bit) for bit in bit_list]
    print(bit_list)
    config = model.get_config()
    config['epoch'] = epoch
    config['batch_size'] = int(batch_size)
    config['dataset'] = dataset
    config['device'] = torch.device("cuda:{}".format(gpu_device))
    config['machine_name'] = str(machine_name)
    config['test_batch_size'] = int(test_batch_size)
    config['test_map'] = int(test_map)
    print(config["test_batch_size"])
    config = config_dataset(config)
    for bit in bit_list:
        print("start training bit {}".format(bit))
        model.train_val(config,bit)







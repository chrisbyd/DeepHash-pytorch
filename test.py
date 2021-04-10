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
import os
import numpy as np
from validate import validate
modules = {'DFH': DFH, 'DPN': DPN, 'CNNH': CNNH, 'HashNet': HashNet, 'GreedyHash': GreedyHash, 'PCDH': PCDH, 'ADSH' :ADSH, 'CSQ': CSQ,
           'LCDSH' : LCDSH, 'IDHN': IDHN, 'DSH': DSH, "DPSH": DPSH , 'DCH':DCH , "DSDH":DSDH}

epoch = int(sys.argv[2])
model_name = sys.argv[1]
bit_num = sys.argv[3]
batch_size = sys.argv[4]
dataset = sys.argv[5]
gpu_device = sys.argv[6]
machine_name = sys.argv[7]
test_map = sys.argv[8]
model = modules[model_name]
test_batch_size = sys.argv[9]
if_save_code =int( sys.argv[10])
print(test_batch_size)


if __name__ == '__main__':
    bit_num = int(bit_num)
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
    device = config["device"]
    net = config["net"](bit_num).to(device)
    saved_model_name = dataset + '_'+ str(bit_num) + '-model.pt'
    path = os.path.join(config["save_path"],saved_model_name)
    net.load_state_dict(torch.load(path))
    precompute_codes = np.load('./save/[HashNet]_imagenet_64-code.npy').item()
    validate(config,bit_num,0,100,net = net, if_save_code= 0, precomputed_codes= precompute_codes)

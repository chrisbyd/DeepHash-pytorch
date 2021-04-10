import torch
import os
from utils.tools import compute_result
from utils.metric import MAPs
from utils.log_to_excel import results_to_excel
from utils.log_to_txt import results_to_txt
from utils.metric import MAPs
from utils.tools import CalcTopMap, get_data
import numpy as np

def validate(config, bit, epoch_num, best_map, net =None, if_save_code =1, precomputed_codes = None):
    device = config["device"]
    if net is None:
        net = config["net"](bit).to(device)
        path = os.path.join(config["save_path"], config["dataset"] + '-' + str(bit) + '-model.pt')
        net.load_state_dict(torch.load(path))
    net.eval()
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    print("The number of gallery points is",num_dataset)

    if precomputed_codes is None:
        query_codes, query_labels = compute_result(test_loader, net, device=device)

        # print("calculating dataset binary code.......")\
        gallery_codes, gallery_labels = compute_result(dataset_loader, net, device=device)
        fname = config['info'] + '_' + config['dataset'] + '_' + str(bit) +'-'+'code.npy'
        if if_save_code == 1:
            codes = {"q_codes": query_codes,
                     'q_labels': query_labels,
                     'g_codes' : gallery_codes,
                     "g_labels" : gallery_labels}
            np.save("./save/"+ fname, codes)


    else:
        query_codes, query_labels, gallery_codes, gallery_labels = precomputed_codes['q_codes'], precomputed_codes['q_labels'], \
                                                                   precomputed_codes['g_codes'], precomputed_codes['g_labels']


        # print("calculating map.......")
    mAP, cum_prec, cum_recall = CalcTopMap(query_codes.numpy(), gallery_codes.numpy(), query_labels.numpy(), gallery_labels.numpy(),
                               config["topK"])

    metric = MAPs(config['topK'])
    top_k_map = metric.get_mAPs_after_sign(query_codes.numpy(), query_labels.numpy(), gallery_codes.numpy(),
                                           gallery_labels.numpy())
    prec, recall, all_map = metric.get_precision_recall_by_Hamming_Radius_All(query_codes.numpy(),
                                                                              query_labels.numpy(),
                                                                              gallery_codes.numpy(),
                                                                              gallery_labels.numpy())
    file_name = config['machine_name'] +'_' +config['dataset']
    model_name = config['info'] + '_' + str(bit) + '_' + str(epoch_num)
    index_range = num_dataset // 100
    index = [i * 100 - 1 for i in range(1, index_range+1)]

    max_index = max(index)
    overflow = num_dataset - index_range * 100
    index = index + [max_index + i  for i in range(1,overflow + 1)]

    c_prec = cum_prec[index].tolist()
    c_recall = cum_recall[index].tolist()


    results_to_txt([mAP], filename=file_name, model_name=model_name, sheet_name='map')
    results_to_txt(c_prec, filename=file_name, model_name=model_name, sheet_name='prec_cum')
    results_to_txt(c_recall, filename=file_name, model_name=model_name, sheet_name='recall_cum')
    results_to_txt(prec.tolist(), filename=file_name, model_name=model_name, sheet_name='prec')
    results_to_txt(recall.tolist(), filename=file_name, model_name=model_name, sheet_name='recall')



    if mAP > best_map :

        if "save_path" in config:
            if not os.path.exists(config["save_path"]):
                os.makedirs(config["save_path"])
            print("save in ", config["save_path"])
            torch.save(net.state_dict(),
                       os.path.join(config["save_path"], config["dataset"] + "-" + str(mAP) +'_' + str(bit) + "-model.pt"))
        else:
            raise NotImplementedError("Needed to offer a save_path in the config")

    return mAP

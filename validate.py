import torch
import os
from utils.tools import compute_result
from utils.metric import MAPs
from utils.log_to_excel import results_to_excel
from utils.metric import MAPs
from utils.tools import CalcTopMap, get_data

def validate(config, bit, epoch_num, best_map, net =None):
    device = config["device"]
    if net is None:
        net = config["net"](bit).to(device)
        path = os.path.join(config["save_path"], config["dataset"] + '-' + str(bit) + '-model.pt')
        net.load_state_dict(torch.load(path))
    net.eval()
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)

    query_codes, query_labels = compute_result(test_loader, net, device=device)

    # print("calculating dataset binary code.......")\
    gallery_codes, gallery_labels = compute_result(dataset_loader, net, device=device)

    # print("calculating map.......")
    mAP, cum_prec = CalcTopMap(query_codes.numpy(), gallery_codes.numpy(), query_labels.numpy(), gallery_labels.numpy(),
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
    index = [i * 100 - 1 for i in range(1, 21)]
    c_prec = cum_prec[index]
    res = c_prec.tolist() + [mAP]
    results_to_excel(res, filename=file_name, model_name=model_name, sheet_name='prec_map')
    results_to_excel(prec, filename=file_name, model_name=model_name, sheet_name='prec')
    results_to_excel(recall, filename=file_name, model_name=model_name, sheet_name='recall')



    if mAP > best_map:

        if "save_path" in config:
            if not os.path.exists(config["save_path"]):
                os.makedirs(config["save_path"])
            print("save in ", config["save_path"])
            torch.save(net.state_dict(),
                       os.path.join(config["save_path"], config["dataset"] + "-" + str(mAP) +'_' + str(bit) + "-model.pt"))
        else:
            raise NotImplementedError("Needed to offer a save_path in the config")

    return mAP

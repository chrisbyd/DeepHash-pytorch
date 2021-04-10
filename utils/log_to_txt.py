import  os


def results_to_txt(res,filename, model_name,sheet_name):
    filename = './{}_{}_results.txt'.format(filename,sheet_name)
    res = [str(item) + ' ' if index != len(res) - 1 else str(item) + '\n' for index, item in enumerate(res)]
    res = ''.join(res)
    res = model_name + ' '+ res
    with open(filename, 'a') as f:
        f.write(res)

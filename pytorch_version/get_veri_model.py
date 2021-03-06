import torch
from mfcc_model import Cnn_model as v1
from veri_model_mfcc import Cnn_model as v2
import os
import constants as c

def get_model(model_load_path,weight_path,model_save_path):

    iden_model = torch.load(model_load_path)
    state = {
        'state': iden_model.state_dict()
    }
    torch.save(state, weight_path)

    model_v2 = v2()
    v1_dic = torch.load(weight_path)
    model_v1 = (v1_dic['state'])
    # print(model_v1)
    v2_dic = model_v2.state_dict()

    model_dict = {k : v for k,v in model_v1.items() if k in v2_dic}
    # print(model_dict)
    v2_dic.update(model_dict)

    model_v2.load_state_dict(v2_dic)
    print(model_v2.state_dict())
    torch.save(model_v2,model_save_path)

if __name__ == "__main__":
    get_model(c.IDEN_MODEL_PATH,c.WEIGHT_PATH,c.VERI_MODEL_PATH)
import torch
import torch.optim as opt
import datetime
import yaml
from pathlib import Path
import numpy as np
from .Model import FeatureMining,MultiTask,Vgg16,TransformerEncoder as Transformer


class ModelUser():
    def __init__(self,configure_path,timestamp,uav_num):
        self.load_config(configure_path,uav_num)
        self.timestamp = timestamp

    def load_config(self,configure_path,uav_num):
        self.configure_path = configure_path 
        f = open(configure_path,"r",encoding="UTF-8")
        configure = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

        self.device = "cuda:0" if configure["use_GPU"] and torch.cuda.is_available() else "cpu" 
        self.alpha = configure["alpha"]

        self.FM_config = configure["feature_mining"]
        self.MT_config = configure["multi_task"] 
        self.loss_F1_name = configure["loss_function1"] 
        if self.loss_F1_name == "CE":
            self.loss_F1 = torch.nn.CrossEntropyLoss()
        elif self.loss_F1_name == "BCE": 
            self.loss_F1 = torch.nn.BCELoss()
        else:
            raise RuntimeError("No matching loss_function1")
        self.loss_F2_name = configure["loss_function2"]
        if self.loss_F2_name == "MSE": 
            self.loss_F2 = torch.nn.MSELoss()
        else:
            raise RuntimeError("No matching loss_function2")
        self.init_model(uav_num)

    def init_model(self,uav_num): 
        if self.FM_config["type"] == "cnn_lstm": 
            self.FM_model = FeatureMining(
                hidden_size=self.FM_config["hidden_size"],
                layer=self.FM_config["layer"],
                dropout=self.FM_config["dropout"],
                uav_num=uav_num
            ) 
            self.FM_model.to(self.device)
            self.FM_model_opt = opt.Adam(filter(lambda p: p.requires_grad, self.FM_model.parameters()),
                                     lr=self.FM_config["lr"])
        else:
            raise RuntimeError("No matching FeatureMining type")


        if self.FM_config["load_switch"] and len(self.FM_config["load_path"]) > 0:
            # new_state_dict = torch.load(self.FM_config["load_path"])
            # new_state_dict = torch.load(self.FM_config["load_path"])
            # self.FM_model.load_state_dict(new_state_dict)
            new_model = torch.load(self.FM_config["load_path"])
            new_state_dict = new_model.state_dict()
            self.FM_model.load_state_dict(new_state_dict)
        self.FM_model.eval()


        self.MT_model = MultiTask( 
            input_channel=self.MT_config["input_channel"], 
            uav_num=uav_num,  
            dropout=self.MT_config["dropout"],  
        )
        self.MT_model.to(self.device)
        self.MT_model_opt = opt.Adam(filter(lambda p: p.requires_grad, self.MT_model.parameters()),
                                   lr=self.MT_config["lr"])
        if self.MT_config["load_switch"] and len(self.MT_config["load_path"]) > 0:
            # new_state_dict = torch.load(self.MT_config["load_path"])
            # self.MT_model.load_state_dict(new_state_dict)
            new_model = torch.load(self.MT_config["load_path"])
            new_state_dict = new_model.state_dict()
            self.MT_model.load_state_dict(new_state_dict)
        self.MT_model.eval() 
        return

    def model_to_train(self):
        self.FM_model.train()
        self.MT_model.train()

    def model_to_eval(self):
        self.FM_model.eval()
        self.MT_model.eval()

    def forward(self, X):


        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X = X.to(self.device)


        self.FM_model_opt.zero_grad()
        self.MT_model_opt.zero_grad()
      

        if self.FM_config["type"] == "cnn_lstm":
           
            multi_label_X = self.FM_model(X)
        else:
            multi_label_X = self.FM_model(X)

        self.pre_label1, self.pre_label2, self.pre_label3, self.pre_compensation = self.MT_model(multi_label_X)
        return

    def calc_loss(self,tar_label1,tar_label2,tar_label3,tar_compensation): 

        def get_attack_indexes(target_label1, perdict_label1): 
            indexes = torch.where(((target_label1==1) | (perdict_label1>=self.alpha)) == True)
            return indexes

        def get_downlink_attack_indexes(target_label1, target_label3, perdict_label1, perdict_label3): 
            indexes = torch.where((((target_label1==1) | (perdict_label1>=self.alpha)) &
                                  ((target_label3==1) | (perdict_label3>=self.alpha))) == True)
            return indexes

        loss_value = [0,0,0,0]
        sample_cnt = [0,0,0,0]
        if self.loss_F1_name == "BCE": 
            self.loss_label1 = self.loss_F1(
                self.pre_label1.flatten(),
                tar_label1.flatten()
            ) 
            loss_value[0] += self.loss_label1.item() * self.pre_label1.flatten().shape[0]
            sample_cnt[0] += self.pre_label1.flatten().shape[0]
            attack_indexes = get_attack_indexes(tar_label1, self.pre_label1)
            if attack_indexes[0].shape[0] > 0: 
                self.loss_label2 = self.loss_F1(
                    self.pre_label2[attack_indexes[0],attack_indexes[1]],
                    tar_label2[attack_indexes[0],attack_indexes[1]],
                )
                self.loss_label3 = self.loss_F1(
                    self.pre_label3[attack_indexes[0],attack_indexes[1]],
                    tar_label3[attack_indexes[0],attack_indexes[1]],
                )
                loss_value[1] += self.loss_label2.item() * attack_indexes[0].shape[0]
                loss_value[2] += self.loss_label3.item() * attack_indexes[0].shape[0]
                sample_cnt[1] += attack_indexes[0].shape[0]
                sample_cnt[2] += attack_indexes[0].shape[0]
            else:
                self.loss_label2 = None
                self.loss_label3 = None

        elif self.loss_F1_name == "CE": 
            self.loss_label1 = self.loss_F1(
                self.pre_label1,
                tar_label1
            )
            loss_value[0] += self.loss_label1.item() * self.pre_label1.shape[0] 
            sample_cnt[0] += self.pre_label1.shape[0] 
            attack_indexes = get_attack_indexes(tar_label1, self.pre_label1) 
            if attack_indexes[0].shape[0] > 0: 
                self.loss_label2 = self.loss_F1(
                    self.pre_label2[attack_indexes[0]],
                    tar_label2[attack_indexes[0]],
                )
                self.loss_label3 = self.loss_F1(
                    self.pre_label3[attack_indexes[0]],
                    tar_label3[attack_indexes[0]],
                )

                loss_value[1] += self.loss_label2.item() * attack_indexes[0].shape[0]
                loss_value[2] += self.loss_label3.item() * attack_indexes[0].shape[0]
                sample_cnt[1] += attack_indexes[0].shape[0]
                sample_cnt[2] += attack_indexes[0].shape[0]

            else:
                self.loss_label2 = None
                self.loss_label3 = None
        else:
            raise RuntimeError("No matching loss_function1")

        downlink_attack_indexes = get_downlink_attack_indexes(tar_label1, tar_label3,
                                                              self.pre_label1, self.pre_label3)
        if downlink_attack_indexes[0].shape[0] > 0:
            self.loss_compensation = self.loss_F2(
                self.pre_compensation[downlink_attack_indexes[0], downlink_attack_indexes[1]],
                tar_compensation[downlink_attack_indexes[0], downlink_attack_indexes[1]]
            )
            loss_value[3] += self.loss_compensation.item() * downlink_attack_indexes[0].shape[0]
            sample_cnt[3] += downlink_attack_indexes[0].shape[0]
        else:
            self.loss_compensation = None
        return loss_value, sample_cnt

    def backward(self):
        self.loss_label1.backward(retain_graph=True) 
        if not self.loss_label2 is None:
            self.loss_label2.backward(retain_graph=True)
        if not self.loss_label3 is None:
            self.loss_label3.backward(retain_graph=True)
        if not self.loss_compensation is None:
            self.loss_compensation.backward(retain_graph=True)
        self.FM_model_opt.step()
        self.MT_model_opt.step()

   

    def save_model(self, save_root, step_num=None):  
    
        save_root_obj = Path(save_root)

        save_root_obj.mkdir(parents=True, exist_ok=True)
        self.save_FM(save_root_obj, step_num)
        self.save_MT(save_root_obj, step_num)
        return

    def save_FM(self, save_root_obj, step_num):  
        if step_num is None:
            FM_save_name = f"{self.timestamp}_FM.p"
        else:

            FM_save_name = f"{self.timestamp}_step[{step_num}]_FM.p"
        FM_save_path = save_root_obj / FM_save_name 
        torch.save(self.FM_model, str(FM_save_path))  
        print(f"FM模型保存至：{str(FM_save_path)}")
        return

    def save_MT(self, save_root_obj, step_num): 
        if step_num is None:
            MT_save_name = f"{self.timestamp}_MT.p"
        else:
  
            MT_save_name = f"{self.timestamp}_step[{step_num}]_MT.p"
        MT_save_path = save_root_obj / MT_save_name 
        torch.save(self.MT_model, str(MT_save_path)) 
        print(f"MT模型保存至：{str(MT_save_path)}")
        return








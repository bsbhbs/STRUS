import numpy as np

from .Test import test

class Compensator(): 
    def __init__(self,batch_size,ModelUser):
        self.ModelUser = ModelUser
        self.batch_size = batch_size
        self.compensated_flag = None

    def execute(self,SW): # SW: sliding window
        pre_label1,pre_label2,pre_label3,self.pre_compensation = test(
            SW, self.batch_size, self.ModelUser,
            show_progress=False
        )
        self.get_compensated_indexes(pre_label1,pre_label3)
        return

    def get_compensated_indexes(self,pre_label1,pre_label3): 
        if self.compensated_flag is None:
            self.compensated_flag = np.zeros_like(pre_label1) 
        else:
            self.compensated_flag[:] = 0 
        temp = pre_label1 & pre_label3 
        compensated_indexes = np.where(temp == 1) 
        self.compensated_flag[compensated_indexes] = 1
        return

    def update_compensation_to_window(self,SW,compensated_flag,pre_compensation):
        compensated_indexes = np.where(compensated_flag==1)
        SW[compensated_indexes[0],compensated_indexes[1],compensated_indexes[2],
            -1,compensated_indexes[3],-3] = \
            pre_compensation[compensated_indexes[0],compensated_indexes[1],
            compensated_indexes[2],compensated_indexes[3],0]
        SW[compensated_indexes[0],compensated_indexes[1],compensated_indexes[2],
            -1,compensated_indexes[3],-2] = \
            pre_compensation[compensated_indexes[0],compensated_indexes[1],
            compensated_indexes[2],compensated_indexes[3],1]
        return
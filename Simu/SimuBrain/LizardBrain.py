# -*- coding: utf-8 -*-
# @Time    : 2023/2/18 17:06
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

import torch
import torch.nn as nn

"""
爬行脑分区
"""


class Flight(nn.Module):
    """
    Brainstem & cerebellum
    脑干和小脑
    (信号)底层自动导航
    """

    def __init__(self, dim_in, max_flight=100):
        super().__init__()
        self.max_flight = max_flight
        self.loop_times = 0
        #
        self.liner1 = nn.Linear(dim_in, dim_in)
        self.relu = nn.ReLU(True)
        self.liner2 = nn.Linear(dim_in, 4 * dim_in)
        #
        self.liner3 = nn.Linear(4 * dim_in, dim_in)
        # signal call type
        self.liner4 = nn.Linear(dim_in, 5)
        # signal call content
        self.liner5 = nn.Linear(dim_in, 1000)

    def forward(self, sig_in):
        sig_out = self.liner1(sig_in)
        sig_out = self.relu(sig_out)
        sig_out = self.liner2(sig_out)
        sig_out = self.relu(sig_out)
        sig_out = self.liner3(sig_out)
        sig_out = self.relu(sig_out)

        sig_flag = self.relu(self.liner4(sig_out))
        sig_content = self.relu(self.liner5(sig_out))

        self.AutoPilot(sig_flag, sig_content)
        return sig_flag, sig_content

    def AutoPilot(self, sig_flag, sig_content):
        """
        负责信息自动驾驶
        """

        print("called AutoPilot.")
        if self.loop_times > self.max_flight:
            exit("Brainstem broken!")

        sigCallType = sig_flag.argmax(axis=1)

        for eachSig in sigCallType.data:
            self.loop_times += 1

            if eachSig == 0:
                self._Sig_Call_Nop("id_0000")
            elif eachSig == 1:
                self._Sig_Call_Clear("id_1111")
            elif eachSig == 2:
                self._Sig_Call_Speak(sig_content[eachSig].data, "id_2222")
            elif eachSig == 3:
                self._Sig_Call_Image(sig_content[eachSig].data, "id_3333")
            elif eachSig == 4:
                self._Sig_Call_Write(sig_content[eachSig].data, "id_4444")

    def _Sig_Call_Nop(self, sig_id):
        print(f"signal loop:{self.loop_times}, signal id:{sig_id}. called Nop.")

    def _Sig_Call_Clear(self, sig_id):
        print(f"signal loop:{self.loop_times}, signal id:{sig_id}. called Clear.")
        self.loop_times = 0

    def _Sig_Call_Speak(self, sig_content, sig_id):
        print(f"signal loop:{self.loop_times}, signal id:{sig_id}. called Speak. sig_content={sig_content[0:5]}")

    def _Sig_Call_Image(self, sig_content, sig_id):
        print(f"signal loop:{self.loop_times}, signal id:{sig_id}. called Image. sig_content={sig_content[0:5]}")

    def _Sig_Call_Write(self, sig_content, sig_id):
        print(f"signal loop:{self.loop_times}, signal id:{sig_id}. called Write. sig_content={sig_content[0:5]}")

    # extend more control to machine


def test():
    batch_size = 32
    dim_in = 1000
    # signal_seq 信号来源于杏仁核
    signal_seq = torch.randn(batch_size, dim_in)
    # Brainstem
    netFlight = Flight(dim_in=dim_in)
    # sig_out 传送至杏仁核
    sig_flag, sig_content = netFlight(signal_seq)

    print("in test():", sig_flag.shape, sig_content.shape)


if __name__ == '__main__':
    test()

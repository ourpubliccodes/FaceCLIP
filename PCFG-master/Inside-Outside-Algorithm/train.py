import math

import numpy as np

from CFG import CFG
from PCFG import PCFG
from PCFG_EM import PCFG_EM
import os
import sys


def train(cfg_file,pcfg_file,attribute_file,iter_num=20):
    cfg = CFG(cfg_file=cfg_file)
    pcfg_tmp = PCFG(pcfg_file=pcfg_file)
    pcfg = PCFG_EM(pcfg_file=pcfg_file,CFG=cfg,PP=pcfg_tmp,attribute_file=attribute_file)
    (name,ext) = os.path.splitext(cfg_file)
    # state = pcfg.EM(iter_num=iter_num)
    # with open(name+'.pcfg','w') as f:
    #     for (A,B,C) in cfg.binary_rules:
    #         f.writelines(A+' -> '+B+' '+C+' '+str(state.get((A,B,C)))+'\n')
    #
    #     for (A,w) in cfg.unary_rules:
    #         f.writelines(A+' -> '+w+' '+str(state.get((A,w)))+'\n')
    xxx = 0
    for img in pcfg.attribute_anno:
        temp_unary_rules = cfg.unary_rules.copy()
        temp_binary_rules = cfg.binary_rules.copy()
        temp_q = pcfg.q.copy()
        shanchu_count = 0
        WearAttributes_count = 7
        IsAttributes_count = 12
        HaveAttributes_count = 25

        ## 确定性别  ###
        if int(img[1]) == 1:  # 说明他是男性，-1为女性
            del temp_unary_rules[6]
            shanchu_count += 1
            del temp_unary_rules[3]
            shanchu_count += 1
        if int(img[1]) == -1:  # 说明他是女性，-1为女性
            del temp_unary_rules[5]
            shanchu_count += 1
            del temp_unary_rules[2]
            shanchu_count += 1
        ##  确定其他属性  ###
        for i in range(2, len(img)):
            if int(img[i]) == -1:  # 没有该属性词，删除
                attr = pcfg.attribute[i]
                for j in range(7, len(temp_unary_rules)):
                    if temp_unary_rules[j][1] == attr:
                        ##  以确定最终剩余属性的个数  ##
                        if temp_unary_rules[j][0] == 'WearAttributes':
                            WearAttributes_count -= 1
                        if temp_unary_rules[j][0] == 'IsAttributes':
                            IsAttributes_count -= 1
                        if temp_unary_rules[j][0] == 'HaveAttributes':
                            HaveAttributes_count -= 1
                        ############################
                        del temp_unary_rules[j]
                        shanchu_count += 1
                        break

        ##  使最终确定好的属性概率相等，同一属性类别的概率和为1  ##
        if WearAttributes_count > 0:
            WearAttributes_P = math.floor((1.0/WearAttributes_count) * 1000)/1000
        if IsAttributes_count > 0:
            IsAttributes_P = math.floor((1.0/IsAttributes_count) * 1000)/1000
        if HaveAttributes_count > 0:
            HaveAttributes_P = math.floor((1.0/HaveAttributes_count) * 1000)/1000
        if WearAttributes_count == 0:
            if IsAttributes_count == 0:
                if HaveAttributes_count == 0:
                    print("attr is error!!")
                else: # 只有have属性
                    temp_binary_rules[3] = ('VP', 'HaveWith')
                    for k in range(0, 7):
                        del temp_binary_rules[4]
                    temp_q[temp_binary_rules[3]] = 1.0
            else: # is属性不为0
                if HaveAttributes_count == 0: # 只有is属性
                    temp_binary_rules[3] = ('VP', 'Are')
                    del temp_binary_rules[11]
                    for k in range(0, 6):
                        del temp_binary_rules[4]
                    temp_q[temp_binary_rules[3]] = 1.0
                else: # 有have和is属性
                    temp_binary_rules[3] = ('VP', 'Are', 'PN', 'HaveWith')
                    temp_binary_rules[4] = ('VP', 'HaveWith', 'PN', 'Are')
                    for k in range(0, 5):
                        del temp_binary_rules[5]
                    temp_q[temp_binary_rules[3]] = 0.5
                    temp_q[temp_binary_rules[4]] = 0.5
        else: # wear属性不为0
            if IsAttributes_count == 0:
                if HaveAttributes_count == 0: # 只有wear属性
                    temp_binary_rules[3] = ('VP', 'Wearing')
                    del temp_binary_rules[11]
                    del temp_binary_rules[10]
                    for k in range(0, 5):
                        del temp_binary_rules[4]
                    temp_q[temp_binary_rules[3]] = 1.0
                else: # 有wear和have属性
                    temp_binary_rules[3] = ('VP', 'Wearing', 'PN', 'HaveWith')
                    temp_binary_rules[4] = ('VP', 'HaveWith', 'PN', 'Wearing')
                    del temp_binary_rules[10]
                    for k in range(0, 4):
                        del temp_binary_rules[5]
                    temp_q[temp_binary_rules[3]] = 0.5
                    temp_q[temp_binary_rules[4]] = 0.5
            else: # is属性不为0
                if HaveAttributes_count == 0:  # 有wear和is属性
                    temp_binary_rules[3] = ('VP', 'Are', 'PN', 'Wearing')
                    temp_binary_rules[4] = ('VP', 'Wearing', 'PN', 'Are')
                    del temp_binary_rules[11]
                    for k in range(0, 4):
                        del temp_binary_rules[5]
                    temp_q[temp_binary_rules[3]] = 0.5
                    temp_q[temp_binary_rules[4]] = 0.5
                else:  # 有wear，have和is属性，则不做任何删除和更改操作temp_unary_rules = {list: 13} [('Det', 'a'), ('Det', 'this'), ('Gender', 'man'), ('Gender', 'person'), ('PN', 'he'), ('WearVerb', 'wears'), ('WearVerb', 'is wearing'), ('IsVerb', 'is'), ('IsAttributes', 'young'), ('IsAttributes', 'neutral'), ('HaveVerb', 'has'), ('HaveAttributes', 'ban… View
                    pass

        for j in range(7, len(temp_unary_rules)):
            if WearAttributes_count > 0 and temp_unary_rules[j][0] == 'WearAttributes':
                temp_q[temp_unary_rules[j]] = WearAttributes_P
            if IsAttributes_count > 0 and temp_unary_rules[j][0] == 'IsAttributes':
                temp_q[temp_unary_rules[j]] = IsAttributes_P
            if HaveAttributes_count > 0 and temp_unary_rules[j][0] == 'HaveAttributes':
                temp_q[temp_unary_rules[j]] = HaveAttributes_P

        pcfg.temp_binary_rules = temp_binary_rules
        pcfg.temp_unary_rules = temp_unary_rules
        pcfg.temp_q = temp_q

        ### 生成句子  #####
        # with open(name+'.gen','w') as f:
        temp_name = img[0].split(".")[0] + '.txt'
        temp_path = "../data/affectnet/text/"+temp_name
        with open(temp_path, 'w') as f:
           for i in range(10):
               n = np.random.normal(loc=4.0, scale=1.0, size=1)
               n = round(n[0])
               pcfg.n = n
               f.writelines(pcfg.gen_sentence('S')+'\n')
        xxx = xxx + 1
        print("finished ", img[0])


if __name__ == '__main__':
    train('test/celeba.cfg','test/celeba.pcfg',"../data/affectnet/affect_attributes.xlsx")






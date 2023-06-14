__author__ = 'zhouyang'


class PCFG:
    def __init__(self, pcfg_file):
        self.pcfg = self.read_pcfg(pcfg_file)
        self.p_noterminals = self.get_p_noterminals()
        self.p_unary_rules = self.get_p_unary_rules()
        self.p_binary_rules = self.get_p_binary_rules()

    def read_pcfg(self,file_name):
        '''
        Read rules from file
        :param file_name:
        :return: all rules
        '''
        PCFG = []
        with open(file_name) as f:
            for line in f:
                line = line.strip().split('->')
                gailv = float(line[1].split(" ")[-1])
                temp = temp = line[1].split(" "+line[1].split(" ")[-1])[0]
                PCFG.append((line[0],temp,gailv))
        return PCFG

    def get_p_noterminals(self):
        '''
        Get noterminals in the pcfg
        :return:
        '''
        p_noterminal = set()
        for rule in self.pcfg:
            p_noterminal.add(rule[0])
        return tuple(p_noterminal)

    def get_p_unary_rules(self):
        '''
        Get the rules in form A->w
        where A is nonterminal and w is terminal
        :return: all rules in form A->w
        '''
        rules = []
        for rule in self.pcfg[12:]:
            rules.append(tuple([rule[0],rule[1],rule[2]]))

        return rules

    def get_p_binary_rules(self):
        '''
        Get the rules in form A->B C
        where A,B,C are all nonterminal
        :return: all rules in form A->B C
        '''
        rules = []
        for rule in self.pcfg[:12]:
            tmp = rule[1].split()
            if len(tmp) == 1:
                rules.append(tuple([rule[0], tmp[0], rule[2]]))
            if len(tmp) == 2:
                rules.append(tuple([rule[0], tmp[0], tmp[1], rule[2]]))
            if len(tmp) == 5:
                rules.append(tuple([rule[0], tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], rule[2]]))
        return rules

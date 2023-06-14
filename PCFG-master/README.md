# Implementation of algorithms in PCFG

## What is it

Implementation of the Expectation Maximation algorithm to calculate probabilities of rules in Context-Free Grammar (CFG) in order to create Probabilistic Context-Free Grammar (PCFG). It also generates new sentences in that grammar using a PCFG (.gen file).

## What do you need to run it

* .cfg file

A file with a context-free grammar in Chomsky normal form, but with "#" instead of an arrow (no spaces around).

Rule example:

```
S#NP VP
NP#Det Gender
NP#PN
VP#Wearing PN Are PN HaveWith
VP#Wearing PN HaveWith PN Are
VP#Are PN Wearing PN HaveWith
VP#Are PN HaveWith PN Wearing
VP#HaveWith PN Wearing PN Are
VP#HaveWith PN Are PN Wearing
Wearing#WearVerb WearAttributes
Are#IsVerb IsAttributes
HaveWith#HaveVerb HaveAttributes
Det#a
Det#this
Gender#man
Gender#woman
Gender#person
PN#he
PN#she
WearVerb#wears
WearVerb#is wearing
WearAttributes#earrings
WearAttributes#hat
WearAttributes#lipstick
WearAttributes#necklace
WearAttributes#necktie
WearAttributes#eyeglasses
WearAttributes#heavy makeup
IsVerb#is 1.0
IsAttributes#smiling
IsAttributes#attractive
IsAttributes#young
IsAttributes#chubby
IsAttributes#bald
IsAttributes#happy
IsAttributes#sad
IsAttributes#fear
IsAttributes#anger
IsAttributes#surprised
IsAttributes#contemptuous
IsAttributes#disgusted
IsAttributes#neutral
HaveVerb#has
HaveAttributes#pale skin
HaveAttributes#oval face
HaveAttributes#arched eyebrows
HaveAttributes#bushy eyebrows
HaveAttributes#bags under eyes
HaveAttributes#narrow eyes
HaveAttributes#high cheekbones
HaveAttributes#rosy cheeks
HaveAttributes#pointy nose
HaveAttributes#big nose
HaveAttributes#mouth slightly open
HaveAttributes#big lips
HaveAttributes#beard
HaveAttributes#goatee
HaveAttributes#mustache
HaveAttributes#double chin
HaveAttributes#receding hairline
HaveAttributes#bangs
HaveAttributes#sideburns
HaveAttributes#straight hair
HaveAttributes#wavy hair
HaveAttributes#blond hair
HaveAttributes#red hair
HaveAttributes#black hair
HaveAttributes#brown hair
HaveAttributes#gray hair
```

* .pcfg file

A file with a context-free grammar in Chomsky normal form, but with "#" instead of an arrow (no spaces around).

Rule example:

```
S->NP VP 1.0
NP->Det Gender 0.5
NP->PN 0.5
VP->Wearing PN Are PN HaveWith 0.166
VP->Wearing PN HaveWith PN Are 0.166
VP->Are PN Wearing PN HaveWith 0.166
VP->Are PN HaveWith PN Wearing 0.166
VP->HaveWith PN Wearing PN Are 0.166
VP->HaveWith PN Are PN Wearing 0.166
Wearing->WearVerb WearAttributes 1.0
Are->IsVerb IsAttributes 1.0
HaveWith->HaveVerb HaveAttributes 1.0
Det->a 0.5
Det->this 0.5
Gender->man 0.75  # 男人
Gender->woman 0.75  # 女人
Gender->person 0.25
PN->he 1.0   #
PN->she 1.0   #
WearVerb->wears 0.5
WearVerb->is wearing 0.5
WearAttributes->earrings 1.0   # 耳饰
WearAttributes->hat 1.0   # 帽子
WearAttributes->lipstick 1.0   # 唇膏，口红
WearAttributes->necklace 1.0   # 项链
WearAttributes->necktie 1.0   # 领带
WearAttributes->eyeglasses 1.0   # 眼镜
WearAttributes->heavy makeup 1.0   # 浓妆
IsVerb->is 1.0
IsAttributes->smiling 1.0   # 微笑
IsAttributes->attractive 1.0   # 吸引人的
IsAttributes->young 1.0   # 年轻的
IsAttributes->chubby 1.0   # 胖乎乎的
IsAttributes->bald 1.0   # 秃的
IsAttributes->happy 1.0   # 高兴的
IsAttributes->sad 1.0   # 伤心的
IsAttributes->fear 1.0   # 害怕的
IsAttributes->anger 1.0   # 生气的
IsAttributes->surprised 1.0   # 惊奇的
IsAttributes->contemptuous 1.0   # 蔑视的
IsAttributes->disgusted 1.0   # 厌恶的
IsAttributes->neutral 1.0   # 中性表情的，无表情的
HaveVerb->has 1.0
HaveAttributes->pale skin 1.0   # 白皮肤
HaveAttributes->oval face 1.0   # 鹅蛋脸
HaveAttributes->arched eyebrows 1.0   # 拱形眉，柳叶眉
HaveAttributes->bushy eyebrows 1.0   # 浓密的眉毛
HaveAttributes->bags under eyes 1.0   # 眼袋
HaveAttributes->narrow eyes 1.0   # 细长的眼睛
HaveAttributes->high cheekbones 1.0   # 高颧骨
HaveAttributes->rosy cheeks 1.0   # 粉红的脸颊
HaveAttributes->pointy nose 1.0   # 尖鼻子
HaveAttributes->big nose 1.0   # 大鼻子
HaveAttributes->mouth slightly open 1.0   # 嘴微张
HaveAttributes->big lips 1.0   # 大嘴唇
HaveAttributes->beard 1.0   # 胡子，总称
HaveAttributes->goatee 1.0   # 山羊胡，下巴上的胡子
HaveAttributes->mustache 1.0   # 嘴唇上较长的胡子，八字胡
HaveAttributes->double chin 1.0   # 双下巴
HaveAttributes->receding hairline 1.0  # 高发际线
HaveAttributes->bangs 1.0   # 刘海
HaveAttributes->sideburns 1.0   # 鬓角
HaveAttributes->straight hair 1.0   # 直发,
HaveAttributes->wavy hair 1.0   # 卷发
HaveAttributes->blond hair 1.0   # 金发
HaveAttributes->red hair 1.0   # 红发
HaveAttributes->black hair 1.0   # 黑发
HaveAttributes->brown hair 1.0   # 棕发
HaveAttributes->gray hair 1.0   # 灰白发
```

* .train file

A text file with .train extension with sentences you want to train your grammar on. Sentences should be in separate lines.


## How to run it

In the command line, type:

```bash
python train.py
```

You can use some examples in the test folder, or try your own.


## References
+    [Inside–outside algorithm](https://en.wikipedia.org/wiki/Inside–outside_algorithm)

+    [The Inside-Outside Algorithm](http://www.cs.columbia.edu/~mcollins/io.pdf)

+    [Note on the Inside-Outside Algorithm](https://www.cs.jhu.edu/~jason/465/iobasics.pdf)

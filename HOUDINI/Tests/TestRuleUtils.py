from HOUDINI.Library.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer.Utils import ReprUtils, RuleUtils
from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.AST import mkFuncSort, mkListSort, mkRealTensorSort, \
    mkBoolTensorSort, mkIntTensorSort
from HOUDINI.Synthesizer.Utils.ReprUtils import *


def printTerm(t):
    # print('++++++++++++++')
    # print(ReprUtils.repr_py_ann(t))
    # print('')
    printCodeGen(t)


def printCodeGen(t):
    print('assert(t == %s)' % str(t))
    print('printTerm(t)')
    print('#%s' % ReprUtils.repr_py_ann(t))
    print('')


def func(*lst):
    return mkFuncSort(*lst)


def lst(t):
    return mkListSort(t)


def getLib():
    libSynth = FnLibrary()
    A = PPSortVar('A')
    B = PPSortVar('B')
    C = PPSortVar('C')

    tr5 = mkRealTensorSort([5])
    tb5 = mkBoolTensorSort([5])
    ti5 = mkIntTensorSort([5])
    ppint = PPInt()

    cnts = PPEnumSort(2, 50)

    libSynth.addItems([
        PPLibItem('map', func(func(A, B), func(lst(A), lst(B))), None),
        PPLibItem('fold', func(func(B, A, B), B, func(lst(A), B)), None),
        PPLibItem('conv', func(func(A, lst(A), A),
                               func(lst(A), lst(A))), None),
        PPLibItem('compose', func(func(B, C), func(A, B), func(A, C)), None),
        PPLibItem('repeat', func(cnts, func(A, A), func(A, A)), None),
        PPLibItem('zeros', func(PPDimVar('a'),
                                mkRealTensorSort([1, 'a'])), None),
        PPLibItem('nn_fun_0', func(tr5, tr5), None),
        PPLibItem('nn_fun_1', func(tr5, tb5), None),
        PPLibItem('nn_fun_2', func(tb5, ti5), None),
    ])
    return libSynth


tr5 = mkRealTensorSort([5])
tb5 = mkBoolTensorSort([5])
ti5 = mkIntTensorSort([5])


def test_repeat():
    libSynth = getLib()
    t = PPTermNT("Z", func(tr5, tr5))
    assert (t == PPTermNT(name='Z',
                          sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                             shape=[PPDimConst(value=5)])],
                                          rtpe=PPTensorSort(param_sort=PPReal(),
                                                            shape=[PPDimConst(value=5)]))))
    printTerm(t)

    t1 = RuleUtils.expandToFuncApp(libSynth, t, 1, 'repeat')
    assert (t1 == PPFuncApp(fn=PPVar(name='lib.repeat'), args=[PPTermNT(name='Z',
                                                                        sort=PPEnumSort(start=2, end=50)),
                                                               t]))
    printTerm(t1)

    t2 = RuleUtils.expandEnum(t1, 1, PPIntConst(5))
    assert (t2 == PPFuncApp(fn=PPVar(name='lib.repeat'),
                            args=[PPIntConst(value=5), t]))

    printTerm(t2)


def test_conv():
    libSynth = getLib()
    t = PPTermNT("Z", func(lst(tr5), lst(tr5)))
    printTerm(t)

    t1 = RuleUtils.expandToFuncApp(libSynth, t, 1, 'conv')
    assert (t1 == PPFuncApp(fn=PPVar(name='lib.conv'),
                            args=[PPTermNT(name='Z',
                                           sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                              shape=[PPDimConst(value=5)]),
                                                                 PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                                                                                    shape=[PPDimConst(value=5)]))],
                                                           rtpe=PPTensorSort(param_sort=PPReal(),
                                                                             shape=[PPDimConst(value=5)])))]))
    printTerm(t1)

    t2 = RuleUtils.expandToUnk(t1, 1)
    assert (t2 == PPFuncApp(fn=PPVar(name='lib.conv'),
                            args=[PPTermUnk(name='Unk',
                                            sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                               shape=[PPDimConst(value=5)]),
                                                                  PPListSort(param_sort=PPTensorSort(param_sort=PPReal(),
                                                                                                     shape=[PPDimConst(value=5)]))],
                                                            rtpe=PPTensorSort(param_sort=PPReal(),
                                                                              shape=[PPDimConst(value=5)])))]))
    printTerm(t2)


def test_map():
    libSynth = getLib()
    # map
    t = PPTermNT("Z", func(lst(tr5), lst(tr5)))
    printTerm(t)

    t1 = RuleUtils.expandToFuncApp(libSynth, t, 1, 'map')
    assert (t1 == PPFuncApp(fn=PPVar(name='lib.map'),
                            args=[PPTermNT(name='Z',
                                           sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                              shape=[PPDimConst(value=5)])],
                                                           rtpe=PPTensorSort(param_sort=PPReal(),
                                                                             shape=[PPDimConst(value=5)])))]))
    printTerm(t1)

    t2 = RuleUtils.expandToVar(libSynth, t1, 1, 'nn_fun_0')
    assert (t2 == PPFuncApp(fn=PPVar(name='lib.map'),
                            args=[PPVar(name='lib.nn_fun_0')]))
    printTerm(t2)


def test_fold():
    tr15 = mkRealTensorSort([1, 5])
    tb51 = mkBoolTensorSort([1, 5])
    ti15 = mkIntTensorSort([1, 5])

    libSynth = getLib()

    t = PPTermNT("Z", func(lst(tr15), tr15))
    printTerm(t)
    # (Z: (List[Tensor[real][1,5]] --> Tensor[real][1,5]))
    t1 = RuleUtils.expandToFuncApp(libSynth, t, 1, 'fold')
    assert (t1 == PPFuncApp(fn=PPVar(name='lib.fold'), args=[PPTermNT(name='Z',
                                                                      sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                                                         shape=[PPDimConst(value=1),
                                                                                                                PPDimConst(value=5)]),
                                                                                            PPTensorSort(param_sort=PPReal(),
                                                                                                         shape=[PPDimConst(value=1),
                                                                                                                PPDimConst(value=5)])],
                                                                                      rtpe=PPTensorSort(param_sort=PPReal(),
                                                                                                        shape=[PPDimConst(value=1),
                                                                                                               PPDimConst(value=5)]))),
                                                             PPTermNT(name='Z',
                                                                      sort=PPTensorSort(param_sort=PPReal(),
                                                                                        shape=[PPDimConst(value=1),
                                                                                               PPDimConst(value=5)]))]))

    printTerm(t1)
    # lib.fold((Z: ((Tensor[real][1,5], Tensor[real][1,5]) --> Tensor[real][1,5])), (Z: Tensor[real][1,5]))
    t2 = RuleUtils.expandToFuncApp(libSynth, t1, 2, 'zeros')

    assert (t2 == PPFuncApp(fn=PPVar(name='lib.fold'),
                            args=[PPTermNT(name='Z',
                                           sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                              shape=[PPDimConst(value=1),
                                                                                     PPDimConst(value=5)]),
                                                                 PPTensorSort(param_sort=PPReal(),
                                                                              shape=[PPDimConst(value=1),
                                                                                     PPDimConst(value=5)])],
                                                           rtpe=PPTensorSort(param_sort=PPReal(),
                                                                             shape=[PPDimConst(value=1),
                                                                                    PPDimConst(value=5)]))),
                                  PPFuncApp(fn=PPVar(name='lib.zeros'),
                                            args=[PPTermNT(name='Z',
                                                           sort=PPDimConst(value=5))])]))

    printTerm(t2)
    # lib.fold((Z: ((Tensor[real][1,5], Tensor[real][1,5]) --> Tensor[real][1,5])), lib.zeros((Z: 5)))
    t3 = RuleUtils.expandDimConst(t2, 2)
    assert (t3 == PPFuncApp(fn=PPVar(name='lib.fold'),
                            args=[PPTermNT(name='Z',
                                           sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                              shape=[PPDimConst(value=1),
                                                                                     PPDimConst(value=5)]),
                                                                 PPTensorSort(param_sort=PPReal(),
                                                                              shape=[PPDimConst(value=1),
                                                                                     PPDimConst(value=5)])],
                                                           rtpe=PPTensorSort(param_sort=PPReal(),
                                                                             shape=[PPDimConst(value=1),
                                                                                    PPDimConst(value=5)]))),
                                  PPFuncApp(fn=PPVar(name='lib.zeros'),
                                            args=[PPIntConst(value=5)])])
            )

    printTerm(t3)
    # lib.fold((Z: ((Tensor[real][1,5], Tensor[real][1,5]) --> Tensor[real][1,5])), lib.zeros(5))
    t4 = RuleUtils.expandToUnk(t3, 1)
    assert (t4 == PPFuncApp(fn=PPVar(name='lib.fold'),
                            args=[PPTermUnk(name='Unk',
                                            sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                               shape=[PPDimConst(value=1),
                                                                                      PPDimConst(value=5)]),
                                                                  PPTensorSort(param_sort=PPReal(),
                                                                               shape=[PPDimConst(value=1),
                                                                                      PPDimConst(value=5)])],
                                                            rtpe=PPTensorSort(param_sort=PPReal(),
                                                                              shape=[PPDimConst(value=1),
                                                                                     PPDimConst(value=5)]))),
                                  PPFuncApp(fn=PPVar(name='lib.zeros'),
                                            args=[PPIntConst(value=5)])]))

    printTerm(t)
    # lib.fold((Unk: ((Tensor[real][1,5], Tensor[real][1,5]) --> Tensor[real][1,5])), lib.zeros(5))


def test_unk():
    t = PPTermNT("Z", func(lst(tr5), tr5))
    printTerm(t)
    t1 = RuleUtils.expandToUnk(t, 1)
    assert(t1 == PPTermNT("Unk", func(lst(tr5), tr5)))
    printTerm(t1)


def test_compose():
    libSynth = getLib()
    t = PPTermNT("Z", func(tr5, ti5))
    printTerm(t)
    t1 = RuleUtils.expandToFuncApp(libSynth, t, 1, 'compose')
    assert (t1 == PPFuncApp(fn=PPVar(name='lib.compose'),
                            args=[PPTermNT(name='Z',
                                           sort=PPFuncSort(args=[PPSortVar(name='B')],
                                                           rtpe=PPTensorSort(param_sort=PPInt(),
                                                                             shape=[PPDimConst(value=5)]))),
                                  PPTermNT(name='Z',
                                           sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                              shape=[PPDimConst(value=5)])],
                                                           rtpe=PPSortVar(name='B')))]))
    printTerm(t1)
    # lib.compose((Z: (B --> Tensor[int][5])), (Z: (Tensor[real][5] --> B)))
    t2 = RuleUtils.expandToVar(libSynth, t1, 1, 'nn_fun_2')
    assert (t2 == PPFuncApp(fn=PPVar(name='lib.compose'),
                            args=[PPVar(name='lib.nn_fun_2'),
                                  PPTermNT(name='Z',
                                           sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(),
                                                                              shape=[PPDimConst(value=5)])],
                                                           rtpe=PPTensorSort(param_sort=PPBool(),
                                                                             shape=[PPDimConst(value=5)])))]))
    printTerm(t2)
    # lib.compose(lib.nn_fun_2, (Z: (Tensor[real][5] --> Tensor[bool][5])))

    t3 = RuleUtils.expandToVar(libSynth, t2, 1, 'nn_fun_1')
    assert (t3 == PPFuncApp(fn=PPVar(name='lib.compose'),
                            args=[PPVar(name='lib.nn_fun_2'),
                                  PPVar(name='lib.nn_fun_1')]))
    printTerm(t3)
    # lib.compose(lib.nn_fun_2, lib.nn_fun_1)


def main():
    test_compose()


if __name__ == '__main__':
    main()

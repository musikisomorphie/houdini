from typing import List, Union
from HOUDINI.Synthesizer import AST


def mkListSort(sort: AST.PPSort) -> AST.PPListSort:
    return AST.PPListSort(sort)


def mkGraphSort(sort: AST.PPSort) -> AST.PPGraphSort:
    return AST.PPGraphSort(sort)


def mkUnk(sort: AST.PPSort) -> AST.PPTermUnk:
    return AST.PPTermUnk('Unk', sort)


def mkFuncSort(*sortlist) -> AST.PPFuncSort:
    return AST.PPFuncSort(list(sortlist[:-1]), sortlist[-1])


def mkTensorSort(sort: AST.PPSort, 
                 rdims: Union[str, int]) -> AST.PPTensorSort:
    dims = []
    for rdim in rdims:
        if type(rdim) == str:
            dims.append(AST.PPDimVar(rdim))
        elif type(rdim) == int:
            dims.append(AST.PPDimConst(rdim))
        else:
            raise Exception("Unhandled dimension")

    return AST.PPTensorSort(sort, dims)


def mkIntTensorSort(rdims) -> AST.PPTensorSort:
    return mkTensorSort(AST.PPInt(), rdims)


def mkRealTensorSort(rdims) -> AST.PPTensorSort:
    return mkTensorSort(AST.PPReal(), rdims)


def mkBoolTensorSort(rdims) -> AST.PPTensorSort:
    return mkTensorSort(AST.PPBool(), rdims)

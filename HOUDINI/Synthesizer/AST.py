from typing import NamedTuple, Union, List

# Custom Sort
PPInt = NamedTuple('PPInt')
PPReal = NamedTuple('PPReal')
PPBool = NamedTuple('PPBool')

PPFuncSort = NamedTuple('PPFuncSort', [('args', List['PPSort']),
                                       ('rtpe', 'PPSort')])

PPTensorSort = NamedTuple('PPTensorSort', [('param_sort', 'PPSort'),
                                           ('shape', List['PPDim'])])

PPEnumSort = NamedTuple('PPEnumSort', [('start', int),
                                       ('end', int)])

PPListSort = NamedTuple('PPListSort', [('param_sort', 'PPSort')])
PPGraphSort = NamedTuple('PPGraphSort', [('param_sort', 'PPSort')])
PPImageSort = NamedTuple('PPImageSort')

PPSortVar = NamedTuple('PPSortVar', [('name', str)])

PPSortTypes = (PPInt, PPReal, PPBool,
               PPFuncSort, PPTensorSort, PPEnumSort,
               PPListSort, PPGraphSort, PPImageSort,
               PPSortVar)

PPSort = Union[PPSortTypes]

# Custom Dim
PPDimVar = NamedTuple('PPDimVar', [('name', str)])
PPDimConst = NamedTuple('PPDimConst', [('value', int)])

PPDimTypes = (PPDimVar, PPDimConst)
PPDim = Union[PPDimTypes]

# Custom Decl
PPSymbol = NamedTuple('PPSymbol', [('value', str)])
PPVarDecl = NamedTuple('PPVarDecl', [('name', str),
                                     ('sort', PPSort)])

PPFuncDecl = NamedTuple('PPFuncDecl', [('fname', PPSymbol),
                                       ('sort', PPFuncSort)])

PPDeclTypes = (PPVarDecl, PPFuncDecl)
PPDecl = Union[PPDeclTypes]

# Custom Term
PPIntConst = NamedTuple('PPIntConst', [('value', int)])
PPRealConst = NamedTuple('PPRealConst', [('value', float)])
PPBoolConst = NamedTuple('PPBoolConst', [('value', bool)])

PPVar = NamedTuple('PPVar', [('name', str)])
PPListTerm = NamedTuple('PPListTerm', [('items', List['PPTerm'])])
PPTermNT = NamedTuple('PPTermNT', [('name', str), ('sort', PPSort)])
PPTermUnk = NamedTuple('PPTermUnk', [('name', str), ('sort', PPSort)])

PPFuncApp = NamedTuple('PPFuncApp', [('fn', 'PPTerm'),
                                     ('args', List['PPTerm'])])

PPLambda = NamedTuple('PPLambda', [('params', List['PPVarDecl']),
                                   ('body', 'PPTerm')])

PPTermTypes = (PPIntConst, PPRealConst, PPBoolConst,
               PPVar, PPListTerm,
               PPTermNT, PPTermUnk,
               PPFuncApp, PPLambda)
PPTerm = Union[PPTermTypes]

PPNTTypes = (PPTermNT)
PPNT = Union[PPTermNT]

PPSortOrDimVar = Union[PPSortVar, PPDimVar]
PPSortOrDim = Union[PPSort, PPDim]
PP = Union[PPSort, PPDim, PPDecl, PPTerm, PPNT]


if __name__ == '__main__':
    print(PP)

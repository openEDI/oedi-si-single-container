#!/usr/bin/python
import sys
import numpy as np
import pydantic
from pydantic import BaseModel
from typing import List
import json

###### Simulation Setup

class StaticConfig(BaseModel):
    name: str

## Sensor Error
class SensorError:
    def __init__(self):
        self.PN = 0.1700;
        self.QN = 0.1700;
        self.VN = 0.001;
        self.AN = 0.001;
        self.PL = 0.001;
        self.QL = 0.001;

class LabelledArray(BaseModel):
    array: List[float]
    unique_ids: List[str]


class Complex(BaseModel):
    real: float
    imag: float


def np_to_complex_list_of_lists(array):
    return [
        [Complex(real=element.real, imag=element.imag) for element in row]
        for row in array
    ]


def np_matrix_to_list_of_lists(array):
    return [
        [element for element in row]
        for row in array
    ]


def list_of_lists_to_np_mat(mat):
    np_mat = np.array([np.array(i) for i in mat], dtype=object)
    return np_mat


def list_of_lists_to_complex_np_mat (mat):
    np_mat = np.array([[x.real + 1j * x.imag for x in row] for row in mat])
    # np_mat = np.array([np.array(i.real +1j*i.imag) for i in mat], dtype=object)
    return np_mat


def list_of_lists_of_tuples_to_complex_np_mat (mat):
    np_mat = np.array([[x[0] + 1j * x[1] for x in row] for row in mat])
    return np_mat


################################################################################
class Base(BaseModel):
    # Parameters
    Sbase: float =  None # float
    Vbase: float =  None # float
    Ibase: float =  None # float
    Ybase: float =  None # float


#def update_Base(SlackBus, P_node, V_node):
#    ret_base = Base()
#    ret_base.Sbase = np.abs(P_node[SlackBus,0]) # float
#    ret_base.Vbase = np.abs(V_node[SlackBus,0]) # float
#    ret_base.Ibase = ret_base.Sbase/ret_base.Vbase;     # float
#    ret_base.Ybase = ret_base.Sbase/ret_base.Vbase**2   # float
#    return ret_base


def update_Base(sBase, vBase):
    ret_base = Base()
    ret_base.Sbase =  sBase     # float
    ret_base.Vbase =  vBase     # float
    ret_base.Ibase =  sBase/vBase       # float
    ret_base.Ybase =  sBase/vBase**2    # float
    return ret_base


################################################################################
class LoLF(BaseModel):
    vals: List[List[float]] = None # list(list(float))

def update_LoLF(xn):
    ret_LoLF = LoLF()
    ret_LoLF.vals = xn
    return ret_LoLF

def LoLF_to_send(xn):
    ret_lolf = LoLF()
    ret_lolf.vals = np_matrix_to_list_of_lists(xn)  # list(list(float))
    return ret_lolf

def LoLF_after_recv(xn):
    ret_lolf = LoLF()
    ret_lolf.vals = list_of_lists_to_np_mat (xn.vals)  # list(list(float))
    return ret_lolf.vals

################################################################################
class ListS(BaseModel):
    vals: List[str] = None # list(str)

def update_ListS(xn):
    ret_listS = ListS()
    ret_listS.vals = xn
    return ret_listS

def ListS_to_send(xn):
    ret_listS = ListS()
    ret_listS.vals = xn  # list(str)
    return ret_listS

def ListS_after_recv(xn):
    ret_listS = ListS()
    ret_listS.vals = xn.vals  # list(str)
    return ret_listS.vals

################################################################################
class ListF(BaseModel):
    vals: List[float] = None # list(float)

def update_ListF(xn):
    ret_ListF = ListF()
    ret_ListF.vals = xn
    return ret_ListF

def ListF_to_send(xn):
    ret_listf = ListF()
    ret_listf.vals = xn.tolist()  # list(float)
    return ret_listf

def ListF_after_recv(xn):
    ret_listf = ListF()
    ret_listf.vals = np.array(xn.vals)  # np.array(float)
    return ret_listf.vals


################################################################################
class Loc(BaseModel):
    Zeroinj: List[List[int]] = None # list(list(int))
    Zerovol: List[List[int]] = None # list(list(int))
    Line: List[List[int]] = None    # list(list(int))


def update_Loc(puTrue,Feeder):
    ret_loc = Loc()
    ret_loc.Zeroinj = [] # list(list(int))
    ret_loc.Zerovol = [] # list(list(int))

    for i in range(len(puTrue.IN)):
        for d in range(np.shape(puTrue.IN)[1]):
            if np.abs(puTrue.IN[i,d]) < 0.001:
                dum = [i,d]
                ret_loc.Zeroinj.append(dum)

    for i in range(len(puTrue.VN)):
        for d in range(np.shape(puTrue.VN)[1]):
            if np.abs(puTrue.VN[i,d]) < 0.001:
                dum = [i,d]
                ret_loc.Zerovol.append(dum)

    #print(Feeder['Topology'][0,0][:,0:2])
    ret_loc.Line = Feeder['Topology'][0,0][:,0:2] # list(list(int))
    return ret_loc


def Loc_to_send(loc):
    ret_loc = Loc()
    ret_loc.Zeroinj = np_matrix_to_list_of_lists(loc.Zeroinj)  # list(list(float))
    ret_loc.Zerovol = np_matrix_to_list_of_lists(loc.Zerovol)  # list(list(float))
    ret_loc.Line = np_matrix_to_list_of_lists(loc.Line)  # list(list(int))
    return ret_loc


def Loc_after_recv(loc):
    ret_loc = Loc()
    ret_loc.Zeroinj = list_of_lists_to_np_mat (loc.Zeroinj)  # list(list(float))
    ret_loc.Zerovol = list_of_lists_to_np_mat (loc.Zerovol)  # list(list(float))
    ret_loc.Line = list_of_lists_to_np_mat (loc.Line)  # list(list(float))
    return ret_loc


################################################################################
class Topology_addl(BaseModel):
    NumNodes: int = 0
    NumLines: int = 0
    Lines: List[List[int]] = None

def update_Topology_addl (num_nodes, num_lines, Lines):
    ret_topo = Topology_addl()
    ret_topo.NumNodes = num_nodes    # int
    ret_topo.NumLines = num_lines    # int
    ret_topo.Lines = Lines           # list(list(int))
    return ret_topo

def Topology_addl_to_send (topo):
    ret_topo = Topology_addl()
    ret_topo.NumNodes   =  int(topo.NumNodes)
    ret_topo.NumLines   =  int(topo.NumLines )
    ret_topo.Lines      =  np_matrix_to_list_of_lists(topo.Lines)  # list(list(int))
    return ret_topo


def Topology_addl_after_recv (topo):
    ret_topo = Topology_addl()
    ret_topo.NumNodes   =  np.int64(topo.NumNodes)
    ret_topo.NumLines   =  np.int64(topo.NumLines )
    ret_topo.Lines      =  list_of_lists_to_np_mat (topo.Lines)
    return ret_topo

################################################################################
class Topology_info(BaseModel):
    NumNodes: int = 0
    NumLines: int = 0
    Lines: List[List[int]] = None
    StateVar: int = 0
    SBase: float =  0
    VBase: float =  0
    IBase: float = 0.0
    YBase: float = 0.0
    Zeroinj: int = 0
    Zerovol: int = 0
    Zeroinj_loc: List[List[int]] = None
    Zerovol_loc: List[List[int]] = None
    InitStates: List[float] = None
    SlackBus: int = 0
    YBus: List[List[Complex]] = None


def update_Topology_info (num_nodes, num_lines, base, yBus, Lines,\
                slackBus, initStates,\
                zeroInj, zeroVol,\
                zeroInj_loc, zeroVol_loc):

    ret_topo = Topology_info()
    ret_topo.NumNodes = num_nodes    # int
    ret_topo.NumLines = num_lines    # int
    ret_topo.Lines = Lines           # list(list(int))
    ret_topo.SBase =  base.Sbase     # float
    ret_topo.VBase =  base.Vbase     # float
    ret_topo.IBase =  base.Ibase     # float
    ret_topo.YBase =  base.Ybase     # float

    # ret_topo.NumNodes = feeder['NumN'][0,0][0,0]   # int
    #ret_topo.NumLines = feeder['NumN'][0,0][0,0]   # int
    # ret_topo.Lines = feeder['Topology'][0,0][:,0:2] # list(list(int))
    # ret_topo.SBase =  sBase     # float
    # ret_topo.VBase =  vBase     # float
    # ret_topo.IBase =  sBase/vBase       # float
    # ret_topo.YBase =  sBase/vBase**2    # float

    ret_topo.StateVar = 3*2
    ret_topo.Zeroinj = zeroInj
    ret_topo.Zerovol = zeroVol
    ret_topo.Zeroinj_loc = zeroInj_loc
    ret_topo.Zerovol_loc = zeroVol_loc
    ret_topo.InitStates = initStates
    ret_topo.SlackBus = slackBus
    ret_topo.YBus = yBus
    return ret_topo


def Topology_info_to_send (topo):
    ret_topo = Topology_info()
    ret_topo.NumNodes   =  int(topo.NumNodes)
    ret_topo.NumLines   =  int(topo.NumLines )
    ret_topo.Lines      =  np_matrix_to_list_of_lists(topo.Lines)  # list(list(int))
    ret_topo.StateVar   =  int(topo.StateVar)
    ret_topo.SBase      =  float(topo.SBase)
    ret_topo.VBase      =  float(topo.VBase)
    ret_topo.IBase      =  float(topo.IBase)
    ret_topo.YBase      =  float(topo.YBase)
    ret_topo.Zeroinj    =  int(topo.Zeroinj)
    ret_topo.Zerovol    =  int(topo.Zerovol)
    ret_topo.Zeroinj_loc = np_matrix_to_list_of_lists(topo.Zeroinj_loc)
    ret_topo.Zerovol_loc = np_matrix_to_list_of_lists(topo.Zerovol_loc)
    ret_topo.InitStates  = topo.InitStates.tolist()
    #ret_topo.SlackBus    = topo.SlackBus
    #ret_topo.SlackBus    = topo.SlackBus.tolist()
    ret_topo.SlackBus    = int(topo.SlackBus)
    ret_topo.YBus= np_to_complex_list_of_lists(topo.YBus)  # list(list(float))
    return ret_topo


def Topology_info_after_recv (topo):
    ret_topo = Topology_info()
    ret_topo.NumNodes   =  np.int64(topo.NumNodes)
    ret_topo.NumLines   =  np.int64(topo.NumLines )
    ret_topo.Lines      =  list_of_lists_to_np_mat (topo.Lines)
    ret_topo.StateVar   =  np.int64(topo.StateVar)
    ret_topo.SBase      =  float(topo.SBase)
    ret_topo.VBase      =  float(topo.VBase)
    ret_topo.IBase      =  float(topo.IBase)
    ret_topo.YBase      =  float(topo.YBase)
    ret_topo.Zeroinj    =  np.int64(topo.Zeroinj)
    ret_topo.Zerovol    =  np.int64(topo.Zerovol)
    ret_topo.Zeroinj_loc = list_of_lists_to_np_mat (topo.Zeroinj_loc)
    ret_topo.Zerovol_loc = list_of_lists_to_np_mat (topo.Zerovol_loc )
    ret_topo.InitStates  = np.array(topo.InitStates)
    #ret_topo.SlackBus    = (topo.SlackBus)
    #ret_topo.SlackBus    = np.array(topo.SlackBus)
    ret_topo.SlackBus    = np.int64(topo.SlackBus)
    ret_topo.YBus= list_of_lists_to_complex_np_mat (topo.YBus)
    return ret_topo


################################################################################
#class Topology_info1(BaseModel):
#    Node: int = 0 # int
#    Line: int = 0 # int
#    StateVar: int = 0 # int
#    Zeroinj: int = 0 # int
#    Zerovol: int = 0 # int
#
#
#def update_Topology_info1 (Feeder):
#    ret_topo1 = Topology_info1()
#    ret_topo1.Node      = Feeder['NumN'][0,0][0,0]   # int
#    ret_topo1.Line      = Feeder['NumL'][0,0][0,0]   # int
#    ret_topo1.StateVar  = 3*2                        # int
#    ret_topo1.Zeroinj      = 50                      # int
#    ret_topo1.Zerovol      = 0                       # int
#    return ret_topo1
#
#
#def Topology_info1_to_send (topo1):
#    ret_topo1 = Topology_info1()
#    ret_topo1.Node      =  int(topo1.Node)
#    ret_topo1.Line      =  int(topo1.Line)
#    ret_topo1.StateVar  =  int(topo1.StateVar)
#    ret_topo1.Zeroinj   =  int(topo1.Zeroinj)
#    ret_topo1.Zerovol   =  int(topo1.Zerovol)
#    return ret_topo1
#
#def Topology_info1_after_recv (topo1):
#    ret_topo1 = Topology_info1()
#    ret_topo1.Node      =  np.int64(topo1.Node)
#    ret_topo1.Line      =  np.int64(topo1.Line)
#    ret_topo1.StateVar  =  np.int64(topo1.StateVar)
#    ret_topo1.Zeroinj   =  np.int64(topo1.Zeroinj)
#    ret_topo1.Zerovol   =  np.int64(topo1.Zerovol)
#    return ret_topo1
#
#
#################################################################################
#class Topology_info2(BaseModel):
#    StateVar: List[float] = None
#    SlackBus: int = None
#    YBus: List[List[Complex]] = None
#
#
#def update_Topology_info2 (StateVar, SlackBus, YBus):
#    ret_topo2 = Topology_info2()
#    ret_topo2.StateVar = StateVar
#    ret_topo2.SlackBus = SlackBus
#    ret_topo2.YBus = YBus
#    return ret_topo2
#
#
#def Topology_info2_to_send (topo2):
#    ret_topo2 = Topology_info2()
#    ret_topo2.StateVar = (topo2.StateVar).tolist() # list(float)
#    ret_topo2.YBus= np_to_complex_list_of_lists(topo2.YBus)  # list(list(float))
#    ret_topo2.SlackBus = int(topo2.SlackBus) # int
#    return ret_topo2
#
#
#def Topology_info2_after_recv (topo2):
#    ret_topo2 = Topology_info2()
#    ret_topo2.StateVar = np.array(topo2.StateVar) # list(float)
#    ret_topo2.YBus= list_of_lists_to_complex_np_mat (topo2.YBus)  # list(list(float))
#    ret_topo2.SlackBus =  np.int64(topo2.SlackBus) # int
#    return ret_topo2
#
#    ret_puTrue.VOL = list_of_lists_to_complex_np_mat (puTrue.VOL)         # list(list(complex))
################################################################################
class PuTrue(BaseModel):
    VOL: List[List[Complex]] = None
    PN : List[List[float]] = None
    QN : List[List[float]] = None
    IN : List[List[float]] = None
    PL : List[List[float]] = None
    QL : List[List[float]] = None
    IL : List[List[float]] = None
    Ve : List[List[float]] = None
    Vf : List[List[float]] = None
    VN : List[List[float]] = None
    AN : List[List[float]] = None


def update_PuTrue (Base, P_node, V_node, I_node, I_line,  P_line):
    ret_puTrue = PuTrue()
    # Parameters
    ret_puTrue.VOL = (V_node/Base.Vbase)         # list(list(complex))
    ret_puTrue.PN = (np.real(P_node)/Base.Sbase)  # list(list(float))
    ret_puTrue.QN = (np.imag(P_node)/Base.Sbase)  # flist(list(loat))
    ret_puTrue.IN = (np.abs(I_node)/Base.Ibase)   # list(list(float))
    ret_puTrue.PL = (np.real(P_line)/Base.Sbase)  # list(list(float))
    ret_puTrue.QL = (np.imag(P_line)/Base.Sbase)  # list(list(float))
    ret_puTrue.IL = (np.abs(I_line)/Base.Sbase)   # list(list(float))
    ret_puTrue.Ve = (np.real(ret_puTrue.VOL))     # list(list(float))
    ret_puTrue.Vf = (np.imag(ret_puTrue.VOL))     # list(list(float))
    ret_puTrue.VN = (np.absolute(ret_puTrue.VOL))      # list(list(float))
    ret_puTrue.AN = (np.angle(ret_puTrue.VOL))    # list(list(float))

    # data pre-processing
    ret_puTrue.PN[np.abs(ret_puTrue.PN) < 1e-8] = 0
    ret_puTrue.QN[np.abs(ret_puTrue.QN) < 1e-8] = 0
    ret_puTrue.IN[np.abs(ret_puTrue.IN) < 1e-8] = 0
    ret_puTrue.PL[np.abs(ret_puTrue.PL) < 1e-8] = 0
    ret_puTrue.QL[np.abs(ret_puTrue.QL) < 1e-8] = 0
    ret_puTrue.IL[np.abs(ret_puTrue.IL) < 1e-8] = 0
    return ret_puTrue


def PuTrue_to_send (puTrue):
    ret_puTrue = PuTrue()
    # Parameters
    ret_puTrue.VOL = np_to_complex_list_of_lists(puTrue.VOL)# list(list(complex))
    ret_puTrue.PN = np_matrix_to_list_of_lists(puTrue.PN)   # list(list(float))
    ret_puTrue.QN = np_matrix_to_list_of_lists(puTrue.QN)   # flist(list(loat))
    ret_puTrue.IN = np_matrix_to_list_of_lists(puTrue.IN)   # list(list(float))
    ret_puTrue.PL = np_matrix_to_list_of_lists(puTrue.PL)   # list(list(float))
    ret_puTrue.QL = np_matrix_to_list_of_lists(puTrue.QL)   # list(list(float))
    ret_puTrue.IL = np_matrix_to_list_of_lists(puTrue.IL)   # list(list(float))
    ret_puTrue.Ve = np_matrix_to_list_of_lists(puTrue.Ve)   # list(list(float))
    ret_puTrue.Vf = np_matrix_to_list_of_lists(puTrue.Vf)   # list(list(float))
    ret_puTrue.VN = np_matrix_to_list_of_lists(puTrue.VN)   # list(list(float))
    ret_puTrue.AN = np_matrix_to_list_of_lists(puTrue.AN)   # list(list(float))
    return ret_puTrue


def PuTrue_after_recv (puTrue):
    ret_puTrue = PuTrue()
    # Parameters
    ret_puTrue.VOL = list_of_lists_to_complex_np_mat (puTrue.VOL)         # list(list(complex))
    ret_puTrue.PN = list_of_lists_to_np_mat (puTrue.PN)  # list(list(float))
    ret_puTrue.QN = list_of_lists_to_np_mat (puTrue.QN)  # flist(list(loat))
    ret_puTrue.IN = list_of_lists_to_np_mat (puTrue.IN)   # list(list(float))
    ret_puTrue.PL = list_of_lists_to_np_mat (puTrue.PL)  # list(list(float))
    ret_puTrue.QL = list_of_lists_to_np_mat (puTrue.QL)  # list(list(float))
    ret_puTrue.IL = list_of_lists_to_np_mat (puTrue.IL)   # list(list(float))
    ret_puTrue.Ve = list_of_lists_to_np_mat (puTrue.Ve)     # list(list(float))
    ret_puTrue.Vf = list_of_lists_to_np_mat (puTrue.Vf)     # list(list(float))
    ret_puTrue.VN = list_of_lists_to_np_mat (puTrue.VN)      # list(list(float))
    ret_puTrue.AN = list_of_lists_to_np_mat (puTrue.AN)    # list(list(float))
    return ret_puTrue


################################################################################
class MeaPha(BaseModel):
    PN : List[List[int]] = None
    QN : List[List[int]] = None
    VN : List[List[int]] = None
    AN : List[List[int]] = None
    PL : List[int] = None
    QL : List[int] = None


def idx2pha(index,n):
    a = index/n

    a1 = np.where((a >= 0) & (a < 1))
    a2 = np.where((a >= 1) & (a < 2))
    a3 = np.where((a >= 2) & (a < 3))

    byphase = np.zeros((len(index),2))
    byphase[a1,0] = index[a1]
    byphase[a1,1] = 1
    byphase[a2,0] = index[a2]-n
    byphase[a2,1] = 2
    byphase[a3,0] = index[a3]-2*n
    byphase[a3,1] = 3
    return byphase


def update_MeaPha (meaIdx,topology_info):
    ret_meaPha = MeaPha ()
    ret_meaPha.PN = (idx2pha(meaIdx.PN,topology_info.NumNodes)) # list(list(int))
    ret_meaPha.QN = (idx2pha(meaIdx.QN,topology_info.NumNodes)) # list(list(int))
    ret_meaPha.VN = (idx2pha(meaIdx.VN,topology_info.NumNodes)) # list(list(int))
    ret_meaPha.AN = (idx2pha(meaIdx.AN,topology_info.NumNodes)) # list(list(int))
    ret_meaPha.PL = []                                     # list(list(int))
    ret_meaPha.QL = []                                     # list(list(int))
    return ret_meaPha


def MeaPha_to_send (meaPha):
    ret_meaPha = MeaPha ()
    ret_meaPha.PN = np_matrix_to_list_of_lists((meaPha.PN)) # list(list(int))
    ret_meaPha.QN = np_matrix_to_list_of_lists((meaPha.QN)) # list(list(int))
    ret_meaPha.VN = np_matrix_to_list_of_lists((meaPha.VN)) # list(list(int))
    ret_meaPha.AN = np_matrix_to_list_of_lists((meaPha.AN)) # list(list(int))
    ret_meaPha.PL = meaPha.PL # np_matrix_to_list_of_lists((meaPha.PL))                                     # list(list(int))
    ret_meaPha.QL = meaPha.QL # np_matrix_to_list_of_lists((meaPha.QL))                                     # list(list(int))
    return ret_meaPha


def MeaPha_after_recv (meaPha):
    ret_meaPha = MeaPha ()
    ret_meaPha.PN = list_of_lists_to_np_mat((meaPha.PN)) # list(list(int))
    ret_meaPha.QN = list_of_lists_to_np_mat((meaPha.QN)) # list(list(int))
    ret_meaPha.VN = list_of_lists_to_np_mat((meaPha.VN)) # list(list(int))
    ret_meaPha.AN = list_of_lists_to_np_mat((meaPha.AN)) # list(list(int))
    ret_meaPha.PL = meaPha.PL # Llist_of_lists_to_np_mat((meaPha.PL))                                     # list(list(int))
    ret_meaPha.QL = meaPha.QL # list_of_lists_to_np_mat((meaPha.QL))                                     # list(list(int))
    return ret_meaPha


################################################################################
class MeaIdx(BaseModel):
    PN : List[int] = None
    QN : List[int] = None
    VN : List[int] = None
    AN : List[int] = None
    PL : List[int] = None
    QL : List[int] = None

def update_MeaIdx (pn_index, qn_index, vn_slack_index, an_slack_index):
    ret_meaIdx = MeaIdx()
    ret_meaIdx.PN = pn_index
    ret_meaIdx.QN = qn_index
    ret_meaIdx.VN = vn_slack_index
    ret_meaIdx.AN = an_slack_index
    ret_meaIdx.PL = [] # list(int)
    ret_meaIdx.QL = [] # list(int)
    return ret_meaIdx


def MeaIdx_to_send (meaIdx):
    ret_meaIdx = MeaIdx ()
    ret_meaIdx.PN = (meaIdx.PN).tolist() # list(int)
    ret_meaIdx.QN = (meaIdx.QN).tolist() # list(int)
    ret_meaIdx.VN = (meaIdx.VN).tolist() # list(int)
    ret_meaIdx.AN = (meaIdx.AN).tolist() # list(int)
    ret_meaIdx.PL = (meaIdx.PL) # list(int)
    ret_meaIdx.QL = (meaIdx.QL) # list(int)
    return ret_meaIdx


def MeaIdx_after_recv (meaIdx):
    ret_meaIdx = MeaIdx ()
    ret_meaIdx.PN = np.array(meaIdx.PN) # list(int)
    ret_meaIdx.QN = np.array(meaIdx.QN) # list(int)
    ret_meaIdx.VN = np.array(meaIdx.VN) # list(int)
    ret_meaIdx.AN = np.array(meaIdx.AN) # list(int)
    ret_meaIdx.PL = meaIdx.PL #np.array(meaIdx.PL) # list(int)
    ret_meaIdx.QL = meaIdx.QL #np.array(meaIdx.QL) # list(int)
    return ret_meaIdx

################################################################################
## Create measure with noise
def update_Mea (PN, QN, VN, AN, PL, QL):
    ret_measure = Mea()
    ret_measure.PN = PN
    ret_measure.QN = QN
    ret_measure.VN = VN**2
    ret_measure.AN = AN
    ret_measure.PL = PL
    ret_measure.QL = QL

    ret_measure.PN = np.reshape(PN, -1, order='F')
    ret_measure.QN = np.reshape(QN, -1, order='F')
    ret_measure.VN = np.reshape(VN, -1, order='F')
    ret_measure.AN = np.reshape(AN, -1, order='F')
    ret_measure.PL = np.reshape(PL, -1, order='F')
    ret_measure.QL = np.reshape(QL, -1, order='F')
    return ret_measure

class Mea(BaseModel):
    PN : List[int] = None
    QN : List[int] = None
    VN : List[int] = None
    AN : List[int] = None
    PL : List[int] = None
    QL : List[int] = None

def Mea_to_send (mea):
    ret_mea = Mea ()
    ret_mea.PN = list(mea.PN) # list(int)
    ret_mea.QN = list(mea.QN) # list(int)
    ret_mea.VN = list(mea.VN) # list(int)
    ret_mea.AN = list(mea.AN) # list(int)
    ret_mea.PL = (mea.PL) # list(int)
    ret_mea.QL = (mea.QL) # list(int)
    return ret_meaIdx

def Mea_after_recv (mea):
    ret_mea = Mea()
    ret_mea.PN = np.array(mea.PN) # list(float)
    ret_mea.QN = np.array(mea.QN) # list(float)
    ret_mea.VN = np.array(mea.VN) # list(float)
    ret_mea.AN = np.array(mea.AN) # list(float)
    ret_mea.PL = mea.PL # llist_of_lists_to_np_mat((meaPha.PL)) # list(list(int))
    ret_mea.QL = mea.QL # list_of_lists_to_np_mat((meaPha.QL))  # list(list(int))
    return ret_mea

################################################################################
class SensorData(BaseModel):
    Rd : List[List[float]]  = None          # list(list(float))
    measure : List[List[float]]  = None     # list(list(float))
    # measure : List[float]  = None # list(list(float))

def update_SensorData(R, measure):
    ret_sensdata = SensorData()
    ret_sensdata.Rd = R
    ret_sensdata.measure = measure
    return ret_sensdata

def SensorData_to_send (sensdata):
    ret_sensdata = SensorData()
    ret_sensdata.Rd = np_matrix_to_list_of_lists(sensdata.Rd)
    ret_sensdata.measure = np_matrix_to_list_of_lists(sensdata.measure)
    # ret_sensdata.measure = (sensdata.measure).tolist()
    return ret_sensdata

def SensorData_after_recv (sensdata):
    ret_sensdata = SensorData()
    ret_sensdata.Rd = list_of_lists_to_np_mat (sensdata.Rd)
    ret_sensdata.measure = list_of_lists_to_np_mat (sensdata.measure)
    #ret_sensdata.measure = np.array(sensdata.measure)
    return ret_sensdata

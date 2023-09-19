
"""
SETO OEDI Project
Python OpenDSS Python Interaction for Henry's OPF code 

## From He Yin, UTK
## P,Q are the active and reactive power flow for all buses from Gadal
## load_id is the bus number where loads are deployed
## name_master is the name of the master.dss for the program
## name_parameter is the .csv file name for the control parameters such as the location of the PV, Cap, and their settings


Created on 09/26/2022

@author: Jin Dong, Boming Liu(ORNL)
"""
# import pandapower as pp
from array import array
import numpy as np
# import scipy.sparse as sp
# from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, identity
# from scipy.sparse.linalg import splu
from scipy.io import loadmat
# from makeYbus import makeYbus
# from makeSbus import makeSbus
# from newtonpf import newtonpf
########################################################################################################################
import opendssdirect as dss
import pandas as pd
import math
from timeit import default_timer as timer
import re
import mat4py as mt

import sys

# def PV_123bus_single_VWC_nocap_IrrTemp_4_v2_powerloss_PS_HELICS(PV_number,PV_bus, Cap_enable,VWC_y_1,VVC_y_1):
# def dss_command():
 
def main(PV_number, Cap_enable, VWC_y_1, VWC_y_2, VVC_y_1, VVC_y_2):
    ## descriptions for the output variables
    # loss: 1x1(double) the active circuit loss from the current power system
    # PV_current_output: 1x14 (double) voltage and current measurement from PV buses
    # PV_powera_output: 1x14 (double)active power flow from the PV buses
    # PV_powerr_output: 1x14 (double) reactive power flow from the PV buses
    # L_current_output: 1x98 (double) voltage and current measurement from load buses
    # L_powera_output: 1x98 (double)active power flow from the Load buses
    # L_powerr_output: 1x98 (double)reactive power flow from the Load buses
    # solve_converged: 1x1(double) whether the power flow is converged

    ## description for the input variables
    # PV_number: 1x1(double) how many PVs deployed in the system
    # PV_bus: the buses where PVs are depployed
    # Cap_enable: 1x4(double) reactive power control for the Capacitors
    # VWC_y_1: 1x14(double) active power control for the PVs
    # VVC_y_1: 1x14(double) reactive power control for the PVs

    ## loading local definitions
    ##load Load_curve.mat

    # Feeder = temp['Feeder']  # sy
    # I_line = temp['I_line']

    # data_PV_control = loadmat('PV_control.mat')  # error due to OCTAVE
    # PV_control_mode = 4
    # Objective_function = 1
    # data_PV_profile = loadmat('PV_profile.mat')
    
    input_data = pd.read_csv('IEEE123_input.csv')
    PV_control_mode = int(input_data[~input_data['PV_control_mode'].isnull()]['PV_control_mode'].values[0])
    print('PV_control_mode is:', PV_control_mode)
    Objective_function = int(input_data['Objective_function'][0])
    print('The objective values is:', Objective_function)
    PV_bus = input_data[~input_data['PV_bus'].isnull()]['PV_bus'].values
    PV_number =  len(PV_bus)
    PV_ratedpower =  input_data['PV_ratedpower'][0]
    Cap_bus = input_data[~input_data['Cap_bus'].isnull()]['Cap_bus'].values
    Cap_Phase = input_data[~input_data['phase'].isnull()]['phase'].values
    Cap_kvar = input_data[~input_data['kvar'].isnull()]['kvar'].values
    Cap_volt = input_data[~input_data['kv'].isnull()]['kv'].values 
    Y_irr = input_data[~input_data['Y_irr'].isnull()]['Y_irr'].values
    Y_temp = input_data[~input_data['Y_temp'].isnull()]['Y_temp'].values
    
   # str(list(Y_irr))
    # print("Hello coordinate")
   

    Dist_fileName = './master.dss'  # Distribution System File directory
    OpenDSSfileName = Dist_fileName

    # JD run OpenDSS
    dss.run_command('clear')  
    dss.run_command('Compile (' + OpenDSSfileName + ')')

    # Load P data
    # df_load_P = pd.read_csv('load_P_daily_123.csv',header=None)
    # load_P = df_load_P.to_numpy()[:,]
    load_PJ = np.array(np.load(os.path.join(baseDir,'data','load_P.npy')))
    load_P = np.transpose(load_PJ)
    # print(load_P)
    

    # Load Q data
    # df_load_Q = pd.read_csv('load_Q_daily_123.csv',header=None)
    # load_Q = df_load_Q.to_numpy()[:,]
    load_QJ = np.array(np.load(os.path.join(baseDir,'data','load_Q.npy')))
    load_Q = np.transpose(load_QJ)
    # print(load_Q)

    df_load_id = pd.read_csv('load_id_123.csv',header=None)
    load_id = df_load_id.to_numpy()
    

    Total_power = sum(sum(abs(load_P)))/96 + sum(sum(abs(load_Q)))/96
    PV_power = Total_power*(PV_ratedpower/PV_number+0.001) # %1% to 30% is a reasonable range
    

    Total_power_a = sum(load_P[0,:])#
    Total_power_r = sum(load_Q[0,:])#
    PV_power_a = PV_ratedpower*Total_power_a/PV_number/1000
    PV_power_r = PV_ratedpower*Total_power_r/PV_number/1000
    Cap_number = len(Cap_bus)
        
 
    for i in range(len(load_P.T)):
        
        # load_phase = data['Loads'][i][0][4][0][0]
        # load_name = re.search(r'\w+(?<=.)', str(load_id[i])).group(0)
        load_name = re.search(r'\w+(?<=.)', str(load_id[0,i])).group(0)
        load_phase = re.search(r'\.[0-9]', str(load_id[0,i])).group(0)
        

        if load_phase == '.1':
            
            loaddata_24_a=abs(load_P[:,i])
            loaddata_24_r=abs(load_Q[:,i])
            dss.run_command('New LoadShape.'+'L'+str(i+1)+'a npts = 96 interval=1 Pmult='+str(list(loaddata_24_a))+' Qmult='+str(list(loaddata_24_r)))
            temp = ['New Load.load_' + load_name + '_1  Bus1=' + load_name + '.1   Phases=1 Conn=Wye   Vminpu=0.8 Vmaxpu=1.2  Model=1 kV=2.4 daily=L'+str(i+1)+'a status=variable']#
            temp = ''.join(temp)
            dss.run_command(temp)#     
            
            temp = ['New Monitor.L_currents'+str(i+1)+'_1 element=Load.load_'+load_name+'_1 terminal=1 mode=0']
            temp = ''.join(temp)
            dss.run_command(temp)   
            
            if Objective_function == 1:
                temp = ['New Monitor.L_powers'+str(i+1)+'_1 element=Load.load_'+load_name+'_1 terminal=1 mode=9']; # mode=9 is to record the energy loss;
                temp = ''.join(temp)
                dss.run_command(temp)
            else:
                temp = ['New Monitor.L_powers'+str(i+1)+'_1 element=Load.load_'+load_name+'_1 terminal=1 mode=1']; # mode=9 is to record the energy loss
                temp = ''.join(temp)
                dss.run_command(temp)
                
        elif load_phase == '.2':
            
            loaddata_24_a=abs(load_P[:,i])
            loaddata_24_r=abs(load_Q[:,i])
            dss.run_command('New LoadShape.'+'L'+str(i+1)+'b npts = 96 interval=1 Pmult='+str(list(loaddata_24_a))+' Qmult='+str(list(loaddata_24_r)))
            temp = ['New Load.load_' + load_name + '_2  Bus1=' + load_name + '.2   Phases=1 Conn=Wye   Vminpu=0.8 Vmaxpu=1.2  Model=1 kV=2.4 daily=L'+str(i+1)+'b status=variable']#
            temp = ''.join(temp)
            dss.run_command(temp)#        
            
            temp = ['New Monitor.L_currents'+str(i+1)+'_2 element=Load.load_'+load_name+'_2 terminal=1 mode=0']
            temp = ''.join(temp)
            dss.run_command(temp)
            
            if Objective_function == 1:
                temp = ['New Monitor.L_powers'+str(i+1)+'_2 element=Load.load_'+load_name+'_2 terminal=1 mode=9']; # mode=9 is to record the energy loss;
                temp = ''.join(temp)
                dss.run_command(temp)
            else:
                temp = ['New Monitor.L_powers'+str(i+1)+'_2 element=Load.load_'+load_name+'_2 terminal=1 mode=1']; # mode=9 is to record the energy loss
                temp = ''.join(temp)
                dss.run_command(temp)

        elif load_phase == '.3':
            loaddata_24_a=abs(load_P[:,i])
            loaddata_24_r=abs(load_Q[:,i])
            dss.run_command('New LoadShape.'+'L'+str(i+1)+'c npts = 96 interval=1 Pmult='+str(list(loaddata_24_a))+' Qmult='+str(list(loaddata_24_r)))
            temp = ['New Load.load_' + load_name + '_3  Bus1=' + load_name + '.3   Phases=1 Conn=Wye   Vminpu=0.8 Vmaxpu=1.2  Model=1 kV=2.4 daily=L'+str(i+1)+'c status=variable']#
            temp = ''.join(temp)
            dss.run_command(temp)#        
            
            temp = ['New Monitor.L_currents'+str(i+1)+'_3 element=Load.load_'+load_name+'_3 terminal=1 mode=0']
            temp = ''.join(temp)
            dss.run_command(temp)
            
            # objective function tbd
            if Objective_function == 1:
                temp = ['New Monitor.L_powers'+str(i+1)+'_3 element=Load.load_'+load_name+'_3 terminal=1 mode=9']; # mode=9 is to record the energy loss;
                temp = ''.join(temp)
                dss.run_command(temp)
            else:
                temp = ['New Monitor.L_powers'+str(i+1)+'_3 element=Load.load_'+load_name+'_3 terminal=1 mode=1']; # mode=9 is to record the energy loss
                temp = ''.join(temp)
                dss.run_command(temp)

       
    # Loads=load_P # not necessary
    ## continuous initialation

    ## start to define the PV as generators
    dss.run_command('New XYCurve.Eff npts=4 xarray=[.1 .2 .4 1.0] yarray=[1 1 1 1]')
    dss.run_command('New XYCurve.FatorPvsT npts=4 xarray=[0 25 75 100] yarray=[1 1 1 1]')
    dss.run_command('New Loadshape.Irrad npts=96 interval=1 mult='+str(list(Y_irr)))
    # dss.run_command('~ mult='+str(list(Y_irr)))
    dss.run_command('New Tshape.Temp npts=96 interval=1 temp='+str(list(Y_temp)))
    # dss.run_command('~ temp='+str(list(Y_temp)))
    
    
    # VWC_y_1 =1.2
    # VWC_y_2 = 1.8
    # VVC_y_1 = 0.2
    # VVC_y_2 = 0.8
    
    VVC_x = [1,1,0,-1,-1]
    VVC_y = [VVC_y_1,VVC_y_2,1.0,2-VVC_y_2,2-VVC_y_1]
    VWC_x = [1,1,0]
    VWC_y = [1,VWC_y_1,VWC_y_2]
    
    VWC_y_all = [1,VWC_y_1,VWC_y_2,1]

    
    Cap_enable = [1,1,1,1] #temp
    
    # VWC_y_all = x
    
    if PV_control_mode == 1:
            temp = ['New XYcurve.generic1 npts=5 yarray='+str(VVC_x)+' xarray='+str(list(VVC_y))]
            temp = ''.join(temp)
            dss.run_command(temp)
    elif PV_control_mode == 2:
            temp = ['New XYcurve.generic2 npts=3 yarray='+str(VWC_x)+' xarray='+str(list(VWC_y))]
            temp = ''.join(temp)
            dss.run_command(temp)
    elif PV_control_mode == 3:
            temp = ['New XYcurve.generic1 npts=5 yarray='+str(VVC_x)+' xarray='+str(VVC_y)]
            temp = ''.join(temp)
            dss.run_command(temp)
            temp = ['New XYcurve.generic2 npts=3 yarray='+str(VWC_x)+' xarray='+str(VWC_y)]
            temp = ''.join(temp)
            dss.run_command(temp)


    for i in range(PV_number): ############ bml : change [i] to  to none for VWC_y_1 and VVC_Y_1 remove
        
        temp = ['New PVSystem.PV'+str(i+1)+' bus1='+str(PV_bus[i])+' Pmpp=100 kV =2.40 kVA='+str(PV_power)+' effcurve=Eff conn=wye']
        temp = ''.join(temp)
        dss.run_command(temp) 
        
        dss.run_command('~ P-TCurve=FatorPvsT %Pmpp=100 irradiance=1 daily=Irrad Tdaily=Temp wattpriority=no')
        
        if PV_control_mode == 1:
            temp = ['New InvControl.VoltVar'+str(i+1)+' mode=VOLTVAR vvc_curve=generic1 RefReactivePower=VARMAX']
            temp = ''.join(temp)
            dss.run_command(temp)        
        elif PV_control_mode == 2:
            temp = ['New InvControl.VoltWatt'+str(i+1)+' mode=VOLTWATT voltwatt_curve=generic2 VoltwattYAxis=PCTPMPPPU']
            temp = ''.join(temp)
            dss.run_command(temp)         
        elif PV_control_mode == 3:
            temp = ['New InvControl.InvCombiControl'+str(i+1)+' CombiMode=VV_VW voltage_curvex_ref=rated vvc_curve1=generic1 voltwatt_curve=generic2 VoltwattYAxis=PCTPMPPPU RefReactivePower=VARAVAL'];
            temp = ''.join(temp)
            dss.run_command(temp) 
            
    for i in range(PV_number):
        temp = ['New Monitor.PV_currents'+str(i+1)+' element=PVSystem.PV'+str(i+1)+' terminal=1 mode=0']
        temp = ''.join(temp)
        dss.run_command(temp)  
        if Objective_function == 1:
             temp = ['New Monitor.PV_powers'+str(i+1)+' element=PVSystem.PV'+str(i+1)+' terminal=1 mode=9'] # mode=9 is to record the energy loss
             temp = ''.join(temp)
             dss.run_command(temp)  
        else:
            temp = ['New Monitor.PV_powers'+str(i+1)+' element=PVSystem.PV'+str(i+1)+' terminal=1 mode=1'] # mode=1 is to record power flow
            temp = ''.join(temp)
            dss.run_command(temp)  
 
    dss.run_command("New Capacitor.C83       Bus1=83      Phases=3     kVAR=600     kV=4.16 enabled = yes NumSteps=5 states={0}")
    dss.run_command("New Capacitor.C88a      Bus1=88.1    Phases=1     kVAR=50      kV=2.402 enabled = yes NumSteps=5 states={0}") #states={0}
    dss.run_command("New Capacitor.C90b      Bus1=90.2    Phases=1     kVAR=50    kV=2.402 enabled = yes NumSteps=5 states={0}")
    dss.run_command("New Capacitor.C92c      Bus1=92.3    Phases=1     kVAR=50     kV=2.402 enabled = yes NumSteps=5 states={0}")
    
    dss.run_command('new capcontrol.CC83 capacitor=C83 element=line.L84 terminal=2 type=voltage PTratio=4160 ONsetting='+str(1-Cap_enable[0])+' OFFsetting='+str(1+Cap_enable[0]))  
    dss.run_command('new capcontrol.CC88 capacitor=C88a element=line.L87 terminal=2 type=voltage PTratio=2400 ONsetting='+str(1-Cap_enable[1])+' OFFsetting='+str(1+Cap_enable[1]))
    dss.run_command('new capcontrol.CC90 capacitor=C90b element=line.L89 terminal=2 type=voltage PTratio=2400 ONsetting='+str(1-Cap_enable[2])+' OFFsetting='+str(1+Cap_enable[2]))
    dss.run_command('new capcontrol.CC92 capacitor=C92c element=line.L91 terminal=2 type=voltage PTratio=2400 ONsetting='+str(1-Cap_enable[3])+' OFFsetting='+str(1+Cap_enable[3]))

    ## deploy energy meters
    for i in range(118):
      temp = ['New energymeter.m'+str(i+1),' Line.L'+str(i+1)]
      temp = ''.join(temp)
      dss.run_command(temp)



    #solve
    dss.run_command('set controlmode=Time')
    dss.run_command('Set mode = daily')
    dss.run_command('Set stepsize = 0.25h')
    dss.run_command('Set number = 96')
    dss.run_command('Set maxcontroliter = 250')
    dss.run_command('Solve')

    ## calculate the loss
    MytotalCircuitLosses= np.array(dss.Circuit.Losses())#
    loss = np.sqrt(MytotalCircuitLosses[0]**2 + MytotalCircuitLosses[1]**2)# #rms(MytotalCircuitLosses)#

    ## is the solution coverged?
    solve_converged = dss.Solution.Converged()

    ## get load infomration
    loadNames = dss.Loads.AllNames()#

    Loads = pd.DataFrame(data = loadNames, columns = ['name'])

    # Get voltages bases
    kVBases = np.array(dss.Settings.VoltageBases())#
    kVBases = [kVBases, kVBases/np.sqrt(3)]# # kvBases are only LL, adding LN
    
    L_current_output = np.zeros((len(Loads),len(load_P[0,:])))#
    L_powera_output = np.zeros((len(Loads),len(load_P[0,:])))#
    L_powerr_output= np.zeros((len(Loads),len(load_P[0,:])))#
    PV_current_output = np.zeros((PV_number,len(load_P)))#
    PV_powera_output = np.zeros((PV_number,len(load_P)))#
    PV_powerr_output = np.zeros((PV_number,len(load_P)))#
    
    ################################################################# PV_ _ size 96*PV_number
    for i in range(PV_number):
 
        dss.Monitors.Name = ['PV_currents'+str(i+1)] #Selects the monitor M1
        PV_current_output[i,:] = dss.Monitors.Channel(1) #reshape(VIMonitor, iMonitorDataSize+2, [])';
        
        dss.Monitors.Name = ['PV_powers'+str(i+1)] #Selects the monitor M1
        
        PV_powera_output[i,:] = dss.Monitors.Channel(1) #reshape(VIMonitor, iMonitorDataSize+2, [])'
        PV_powerr_output[i,:] = dss.Monitors.Channel(2) #eshape(VIMonitor, iMonitorDataSize+2, [])'


    k = 0
    if solve_converged:

        for i in range(len(load_P[0,:])):
            load_phase = re.search(r'\.[0-9]', str(load_id[0,i])).group(0)

        
            if load_phase == '.1':
                    dss.Monitors.Name=['L_currents'+str(i+1)+'_1']
                    L_current_output[:,k] = dss.Monitors.Channel(1)
                    dss.Monitors.Name = ['L_powers'+str(i+1),'_1']
                    L_powera_output[:,k] = dss.Monitors.Channel(1)
                    L_powerr_output[:,k] = dss.Monitors.Channel(2)
                    k=k+1
            elif load_phase == '.2':
                    dss.Monitors.Name=['L_currents'+str(i+1)+'_2']
                    L_current_output[:,k] = dss.Monitors.Channel(1)
                    dss.Monitors.Name = ['L_powers'+str(i+1),'_2']
                    L_powera_output[:,k] = dss.Monitors.Channel(1)
                    L_powerr_output[:,k] = dss.Monitors.Channel(2)
                    k=k+1
            elif load_phase == '.3':
                    dss.Monitors.Name=['L_currents'+str(i+1)+'_3']
                    L_current_output[:,k] = dss.Monitors.Channel(1)
                    dss.Monitors.Name = ['L_powers'+str(i+1),'_3']
                    L_powera_output[:,k] = dss.Monitors.Channel(1)
                    L_powerr_output[:,k] = dss.Monitors.Channel(2)
                    k=k+1
                    
    else:
        PV_current_output=[]
        PV_powera_output=[]
        PV_powerr_output=[]
        L_current_output=[]
        L_powera_output=[]
        L_powerr_output=[]

  
    print(L_current_output)
    print(PV_current_output)

 
    return loss,PV_current_output,PV_powera_output,PV_powerr_output,L_current_output,L_powera_output,L_powerr_output,solve_converged


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

# if __name__ == "__main__":
    
#     # PV_number = 14   
#     # Cap_enable =[16.565,   19.635,   13.979,   20.770,]  
#     # VWC_y_1= [16.170,   20.012,   19.885,   22.129,   18.434,   15.138,   14.845,   23.023,   15.702,   16.431,   23.263,   22.269,   15.723,   13.505]
#     # VVC_y_1=[22.375,   24.371,   24.779,   21.732,   22.142,  16.452,   22.160,   19.419,   18.727,   20.243,   18.011,   21.562,   21.823,   17.578]
    
#     PV_number = 10
#     Cap_enable = [1,1,1,1]
#     VWC_y_1 =1.2
#     VWC_y_2 = 0.8
#     VVC_y_1 = 0.2
#     VVC_y_2 = 0.8   
#     [loss,PV_current_output,PV_powera_output,PV_powerr_output,L_current_output,L_powera_output,L_powerr_output,solve_converged] = main(PV_number, Cap_enable, VWC_y_1, VWC_y_2, VVC_y_1, VVC_y_2)
#     print(solve_converged)


if __name__ == "__main__":       

#     #----------- Using the following default values --------------
    PV_number = 10
    Cap_enable = [1,1,1,1]

    # VWC_y_1 = 1.2
    # VWC_y_2 = 1.2    
    # VVC_y_1 = 0.2    
    # VVC_y_2 = 0.2

#     #----------- Pass these VVO curve coefficients -------------

    baseDir='/home/run/dopf_ornl'
    filename = os.path.join(baseDir,'data','Octave_to_Python_Inputs_data.txt')
    data_inputs = np.loadtxt(filename, dtype=float)
    # print(data_inputs)
    data_inputs = np.array(data_inputs)

    # if data_inputs[2] == 0.86:
    #     data_inputs[2] = 0.82

    # if (data_inputs[3] < data_inputs[2]) or (data_inputs[1] < data_inputs[0]) or (data_inputs[3] < 0.6):
    #     loss2  = 10000
    # else:

    # elif data_inputs[3] <= 0.6:
    #     data_inputs[3] = 0.6
    # data_inputs = np.transpose(data_inputs)
    # print(data_inputs[0])
    # print(data_inputs[1])
    # print(data_inputs[2])
    # print(data_inputs[3])
    # data_inputs = [1.43801,   1.32399,   0.46666,   0.30645]
    
    # [loss, PV_current_output, PV_powera_output, PV_powerr_output, L_current_output,L_powera_output,L_powerr_output,solve_converged] = main(PV_number, Cap_enable, VWC_y_1, VWC_y_2, VVC_y_1, VVC_y_2)
    [loss, PV_current_output, PV_powera_output, PV_powerr_output, L_current_output,L_powera_output,L_powerr_output,solve_converged] = main(PV_number, Cap_enable, data_inputs[0], data_inputs[1], data_inputs[2],data_inputs[3])
    # [loss, PV_current_output, PV_powera_output, PV_powerr_output, L_current_output,L_powera_output,L_powerr_output,solve_converged] = main(PV_number, Cap_enable, 1.2,   1.1,   0.82,   0.6)

    # print(PV_current_output.shape)
    # print(L_current_output.shape)

    if np.isnan(loss) or len(PV_current_output) == 0:
        loss = 1000000000
        temp_fake_output1 = np.ones([14, 96])
        PV_current_output = temp_fake_output1
        PV_powera_output = temp_fake_output1
        PV_powerr_output = temp_fake_output1
        temp_fake_output2 = np.ones([96, 96])
        L_current_output = temp_fake_output2
        L_powera_output = temp_fake_output2
        L_powerr_output = temp_fake_output2
    
    
    loss_2 = trunc(loss, decs=2)
        
    # Reduce to 2 decimals only
    PV_current_output_2 = trunc(PV_current_output, decs=2)
    PV_powera_output_2 = trunc(PV_powera_output, decs=2)
    PV_powerr_output_2 = trunc(PV_powerr_output, decs=2)
    L_current_output_2 = trunc(L_current_output, decs=2)
    L_powera_output_2 = trunc(L_powera_output, decs=2)
    L_powerr_output_2 = trunc(L_powerr_output, decs=2)

    print('loss is:',loss_2)
    print('Converged?', solve_converged)

    #### ------  This is where we saved the .npy files
    np.save('loss.npy', loss_2)
    
    np.save('PV_current_output.npy', PV_current_output_2)
    np.save('PV_powera_output.npy', PV_powera_output_2)
    np.save('PV_powerr_output.npy', PV_powerr_output_2)
    
    np.save('L_current_output.npy', L_current_output_2)
    np.save('L_powera_output.npy', L_powera_output_2)
    np.save('L_powerr_output.npy', L_powerr_output_2)

    #     outputs = [str(loss_2) + '=' + str(PV_current_output_2) + '=' +  str(PV_powera_output_2) + '=' + str(PV_powerr_output_2)  + '=' + str(L_current_output_2)  + '=' +  str(L_powera_output_2) + '=' +  str(L_powerr_output_2) + '=' +  str(solve_converged)]
    outputs = [str(solve_converged)]      
    outputs_with_brackets = ''.join(outputs)
    # outputs_with_brackets2 = re.sub(r"[\[\]]",'', outputs_with_brackets)
    sys.stdout.write(re.sub(r"[\[\]\n]",'', outputs_with_brackets))

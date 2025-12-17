import pandas as pd
import pandapower as pp
import numpy as np
import copy
from pandapower.plotting import simple_plotly, pf_res_plotly
import os
import tempfile
from random import uniform
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
import plotly.express as px


df = pd.read_excel('Real_demand_and_generation_data_oneday.xls')
# Rename Column
df.rename({'Unnamed: 0' : "id",
           "Unnamed: 1" : "name",
           "Unnamed: 2" : "geoid",
           "Unnamed: 3" : "geoname",
           "Unnamed: 4" : "value",
           "Unnamed: 5" : "datetime",
           }, axis='columns', inplace=True)
# Modify the value
df.drop(0, inplace=True)
# delete "."
for value in df.value:
    if type(value)==str:
        df.loc[df.value == value, "value"] = value.replace('.','') #第一个是查它的行号，因为每个值都是unique的
    elif type(value)==int:
        df.loc[df.value == value, "value"] = 0  # 必要！check solar data 没有日照的时候会蹦出来int值，实际应为0
# change ","
for value in df.value:
    if type(value)==str:
        df.loc[df.value == value, "value"] = value.replace(',','.')
# str to float
for value in df.value:
    if type(value)==str:
        df.loc[df.value == value, "value"] = df.loc[df.value == value, "value"].astype(float)

# Create 4 seperate profiles
demand_profile = df.loc[df['name'] == 'Demanda real nacional'].copy()
solar_profile = df.loc[df['name'] == 'Generación T.Real Solar'].copy()
wind_profile =  df.loc[df['name'] == 'Generación T.Real eólica'].copy()
nuclear_profile =  df.loc[df['name'] == 'Generación T.Real nuclear'].copy()
# Remove unnecessary column
demand_profile.drop(columns=['id', 'name', 'geoid', 'geoname', 'datetime'], inplace=True)
solar_profile.drop(columns=['id', 'name', 'geoid', 'geoname', 'datetime'], inplace=True)
wind_profile.drop(columns=['id', 'name', 'geoid', 'geoname', 'datetime'], inplace=True)
nuclear_profile.drop(columns=['id', 'name', 'geoid', 'geoname', 'datetime'], inplace=True)
# Reset index
demand_profile.reset_index(drop=True, inplace=True)
solar_profile.reset_index(drop=True, inplace=True)
wind_profile.reset_index(drop=True, inplace=True)
nuclear_profile.reset_index(drop=True, inplace=True)
# New column for p.u. value
demand_profile['value_pu'] = 0
solar_profile['value_pu'] = 0
wind_profile['value_pu'] = 0
nuclear_profile['value_pu'] = 0
# Normalization
for value in demand_profile.value:
    demand_profile.loc[demand_profile.value == value, "value_pu"] = float(value/demand_profile.max()[0]) # [0]的作用是取profile中的第一个列

for value in solar_profile.value:
    solar_profile.loc[solar_profile.value == value, "value_pu"] = float(value/solar_profile.max()[0])

for value in wind_profile.value:
    wind_profile.loc[wind_profile.value == value, "value_pu"] = float(value/wind_profile.max()[0])

for value in nuclear_profile.value:
    nuclear_profile.loc[nuclear_profile.value == value, "value_pu"] = float(value/nuclear_profile.max()[0])


project_net = pp.create_empty_network()
######## Buses
# creating HV buses
pp.create_bus(project_net, name="Terrasa", vn_kv=110, geodata=(2.01787, 41.56681))
pp.create_bus(project_net, name="Manresa", vn_kv=110, geodata=(1.81685, 41.72396))
pp.create_bus(project_net, name="Tarrega", vn_kv=110, geodata=(1.13954, 41.64687))
pp.create_bus(project_net, name="Montblanc", vn_kv=110, geodata=(1.16178, 41.37478))
pp.create_bus(project_net, name="Vandellos", vn_kv=110, geodata=(0.87562, 40.95624))
pp.create_bus(project_net, name="Lleida", vn_kv=110, geodata=(0.61613, 41.61538))
pp.create_bus(project_net, name="Igualada", vn_kv=110, geodata=(1.61912, 41.58074))
pp.create_bus(project_net, name="El Perellos", vn_kv=110, geodata=(0.71262, 40.87541))
pp.create_bus(project_net, name="Agramunt", vn_kv=110, geodata=(1.08664, 41.79154))
pp.create_bus(project_net, name="Valls", vn_kv=110, geodata=(1.24764, 41.28434))
pp.create_bus(project_net, name="Falset", vn_kv=110, geodata=(0.82771, 41.14390))
pp.create_bus(project_net, name="Conesa", vn_kv=110, geodata=(1.27159, 41.50215))

# creating MV buses
pp.create_bus(project_net, name="Manresa_LV", vn_kv=20, geodata=(1.81210, 41.71518))
pp.create_bus(project_net, name="Tarrega_LV", vn_kv=20, geodata=(1.15177, 41.64157))
pp.create_bus(project_net, name="Montblanc_LV", vn_kv=20, geodata=(1.15398, 41.37473))
pp.create_bus(project_net, name="Lleida_LV", vn_kv=20, geodata=(0.61427, 41.61262))

######## Transformers
pp.create_transformer_from_parameters(project_net, hv_bus = 1, lv_bus = 12, sn_mva = 100, vn_hv_kv =110, vn_lv_kv = 20, vk_percent = 12, vkr_percent = 0.26, pfe_kw = 55, i0_percent = 0.06, name = 'Trafo_Manresa')

pp.create_transformer_from_parameters(project_net, hv_bus = 2, lv_bus = 13, sn_mva = 160, vn_hv_kv = 110, vn_lv_kv = 20, vk_percent = 12.2, vkr_percent = 0.25, pfe_kw = 60, i0_percent = 0.06, name = 'Trafo_Tarrega')

pp.create_transformer_from_parameters(project_net, hv_bus = 3, lv_bus = 14, sn_mva = 100, vn_hv_kv = 110, vn_lv_kv = 20, vk_percent = 12, vkr_percent = 0.26, pfe_kw = 55, i0_percent = 0.06, name = 'Trafo_Montblanc')

pp.create_transformer_from_parameters(project_net, hv_bus = 5, lv_bus = 15, sn_mva = 100, vn_hv_kv = 110, vn_lv_kv = 20, vk_percent = 12, vkr_percent = 0.26, pfe_kw = 55, i0_percent = 0.06, name = 'Trafo_Lleida')

############# Lines
# Single lines
pp.create_line_from_parameters(project_net, from_bus=0, to_bus=1, length_km=23, r_ohm_per_km=0.062, x_ohm_per_km=0.41036, c_nf_per_km=8.792127, max_i_ka=0.88898, name='s0_1')
pp.create_line_from_parameters(project_net, from_bus=2, to_bus=6, length_km=40, r_ohm_per_km=0.062, x_ohm_per_km=0.41036, c_nf_per_km=8.792127, max_i_ka=0.88898, name='s2_6')
pp.create_line_from_parameters(project_net, from_bus=2, to_bus=5, length_km=44, r_ohm_per_km=0.062, x_ohm_per_km=0.41036, c_nf_per_km=8.792127, max_i_ka=0.88898, name='s2_5')
# Double lines
pp.create_line_from_parameters(project_net, from_bus=1, to_bus=2, length_km=58, r_ohm_per_km=0.031, x_ohm_per_km=0.193358009, c_nf_per_km=18.69669739, max_i_ka=0.88898 * 2, name='d1_2')
pp.create_line_from_parameters(project_net, from_bus=2, to_bus=3, length_km=30, r_ohm_per_km=0.031, x_ohm_per_km=0.193358009, c_nf_per_km=18.69669739, max_i_ka=0.88898 * 2, name='d2_3')
pp.create_line_from_parameters(project_net, from_bus=3, to_bus=4, length_km=52, r_ohm_per_km=0.031, x_ohm_per_km=0.193358009, c_nf_per_km=18.69669739, max_i_ka=0.88898 * 2, name='d3_4')
pp.create_line_from_parameters(project_net, from_bus=4, to_bus=5, length_km=76.4, r_ohm_per_km=0.031, x_ohm_per_km=0.193358009, c_nf_per_km=18.69669739, max_i_ka=0.88898 * 2, name='d4_5')

############ External grid
pp.create_ext_grid(project_net,0)

############# Loads
def get_reactive(P,PF): #everything the unit is in kX
    Q = P * np.tan(np.arccos(PF))
    return Q
# Bus 1 load - Type I
PF = 0.98
p_mw = 60
q_mvar = get_reactive(p_mw, PF)
pp.create_load(project_net,12,p_mw,q_mvar)
# Bus 3 load - Type I
PF = 0.98
p_mw = 60
q_mvar = get_reactive(p_mw, PF)
pp.create_load(project_net,14,p_mw,q_mvar)
# Bus 5 load - Type I
PF = 0.98
p_mw = 60
q_mvar = get_reactive(p_mw, PF)
pp.create_load(project_net,15,p_mw,q_mvar)
# Bus 2 load - Type II
PF = 0.98
p_mw = 145
q_mvar = get_reactive(p_mw, PF)
pp.create_load(project_net,13,p_mw,q_mvar)

######## Generators
pp.create_gen(project_net, 4, p_mw = 245, vm_pu = 1.0)


pp.to_excel(project_net, 'project_final_baseline.xlsx')

# import "net" and Create Profile DataFrame
net = pp.from_excel('project_final_baseline.xlsx')
# Create DataFrame for profiles
profiles = pd.DataFrame()
profiles.index = demand_profile.index # Time from 0-23
########## Load profiles
for value in demand_profile['value_pu']:
    demand_profile.loc[demand_profile['value_pu'] == value,'value_pu'] = value * uniform(0.85, 1.05)
    profiles['load0'] = demand_profile['value_pu'] * net.load.p_mw[0]

for value in demand_profile['value_pu']:
    demand_profile.loc[demand_profile['value_pu'] == value,'value_pu'] = value * uniform(0.85, 1.05)
    profiles['load1'] = demand_profile['value_pu'] * net.load.p_mw[1]

for value in demand_profile['value_pu']:
    demand_profile.loc[demand_profile['value_pu'] == value,'value_pu'] = value * uniform(0.85, 1.05)
    profiles['load2'] = demand_profile['value_pu'] * net.load.p_mw[2]

for value in demand_profile['value_pu']:
    demand_profile.loc[demand_profile['value_pu'] == value,'value_pu'] = value * uniform(0.85, 1.05)
    profiles['load3'] = demand_profile['value_pu'] * net.load.p_mw[3]

profiles['gen0'] = nuclear_profile['value_pu'] * net.gen.p_mw[0]


# Create ds and assign values from profile with controller
#ConstControl and profiles 之间语言不通，必须依靠 ds 来传递数据
ds = DFData(profiles)

for i in net.load.index:
    ConstControl(net, element='load', variable='p_mw', element_index=i, data_source=ds, profile_name=['load'+str(i)], scale_factor=1)

for j in net.gen.index:
    ConstControl(net, element='gen', variable='p_mw', element_index=j, data_source=ds, profile_name=['gen'+str(j)], scale_factor=1)


time_steps = range(0, profiles.shape[0])
output_dir = os.getcwd() + '/results_v1'
ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xlsx", log_variables=list())

ow.log_variable('res_load', 'p_mw') # 看load的有功功率 p_mw 是否符合定义
ow.log_variable('res_gen', 'p_mw') # 看有功输出是不是符合定义
ow.log_variable('res_sgen', 'q_mvar') #看无功输出是不是符合定义 (wind_profile)
ow.log_variable('res_bus', 'vm_pu')  # 看有没有在 0.9-1.1范围内
ow.log_variable('res_bus', 'p_mw')  #看哪条母线传递了很多有功功率
ow.log_variable('res_bus', 'q_mvar')  #看哪条母线传递了很多无功功率
ow.log_variable('res_line', 'loading_percent')  # 看每条线是不是都低于 80%
ow.log_variable('res_line', 'i_ka')  # 进一步看看线路上的电流
ow.log_variable('res_trafo', 'loading_percent')  #看看变压器负载率

run_timeseries(net, time_steps, run=pp.runpp)


#print('######### Results for voltage in p.u. ########')
vm_pu_file = output_dir + '/res_bus/vm_pu.xlsx'
vm_pu = pd.read_excel(vm_pu_file, index_col=0)  #索引为excel里的第0列

#print('#######  Results for line loading in % ########')
ll_file = output_dir + '/res_line/loading_percent.xlsx'
line_loading = pd.read_excel(ll_file, index_col=0)

#print('#######  Results for bus power in mw ########')
p_mw_file = output_dir + '/res_bus/p_mw.xlsx'
p_mw = pd.read_excel(p_mw_file, index_col=0)

#print('#######  Results for load power in MW ########')
load_file = output_dir + '/res_load/p_mw.xlsx'
load = pd.read_excel(load_file, index_col=0)

#print('#######  Results for generation power in MW ########')
gen_file = output_dir + '/res_gen/p_mw.xlsx'
gen = pd.read_excel(gen_file, index_col=0)


# plotting voltage results
# 反映无功 - bus
fig = px.line(vm_pu, y=vm_pu.columns, labels={"value": "count", "variable": "Bus ID"})
fig.update_xaxes(title='Hour', fixedrange=True)
fig.update_yaxes(title='Voltage [p.u.]')
fig.update_traces(mode="markers+lines", hovertemplate='Hour: %{x} <br>Voltage: %{y}')
fig.show()

#  plotting line results

fig = px.line(line_loading, y=line_loading.columns, labels={"value": "count", "variable": "Line ID"})
fig.update_xaxes(title='Hour', fixedrange=True)
fig.update_yaxes(title='Loading [%]')
fig.update_traces(mode="markers+lines", hovertemplate='Hour: %{x} <br>Loading: %{y}')
fig.show()

#  plotting bus power results

fig = px.line(p_mw, y=p_mw.columns, labels={"value": "count", "variable": "Bus ID"})
fig.update_xaxes(title='Hour', fixedrange=True)
fig.update_yaxes(title='Power [MW]')
fig.update_traces(mode="markers+lines", hovertemplate='Hour: %{x} <br>Loading: %{y}')
fig.show()

# plotting load results

fig = px.line(load*1000, y=load.columns, labels={"value": "count", "variable": "Load ID"})
fig.update_xaxes(title='Hour', fixedrange=True)
fig.update_yaxes(title='Power [kW]') # *1000, from MW to kW
fig.update_traces(mode="markers+lines", hovertemplate='Hour: %{x} <br>Power: %{y}')
fig.show()
# notice that load[0] is reflecting our control! we give the value of 'load0' in ds to it
# same for gen[0] of data from 'gen1' of ds
# the load and generation results are given by profile, not from runpp

# plotting generation results
fig = px.line(gen*1000, y=gen.columns, labels={"value": "count", "variable": "Gen ID"})
fig.update_xaxes(title='Hour', fixedrange=True)
fig.update_yaxes(title='Power [kW]')
fig.update_traces(mode="markers+lines", hovertemplate='Hour: %{x} <br>Power: %{y}')
fig.show()


#heavist hour - defined by line_loading sum
dn = line_loading
dn['Hourly Sum'] = dn.sum(axis=1)
max_loading_row = dn['Hourly Sum'].idxmax()
net2 = pp.from_excel('project_final_baseline.xlsx')
gen2 = gen.copy()
load2 = load.copy()

for i in gen2.columns:
    net2.gen.loc[i,'p_mw'] = gen2.loc[max_loading_row,i]

for i in load2.columns:
    net2.load.loc[i,'p_mw'] = load2.loc[max_loading_row,i]

pp.runpp(net2, numba=False)
pf_res_plotly(net2, on_map=True)
print()
print(gen2.columns)
print(load2.columns)
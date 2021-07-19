from numpy.core.fromnumeric import mean
from lib import common, envs, model
import argparse
import numpy as np
import drl
import matplotlib.pyplot as plt
import time

from tensorboardX import SummaryWriter

import torch
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd


def create_agent (env):
    action_dim = env.action_space.shape[0]

    eps = env.storage_capacity.copy() / 10
    Q = env.storage_capacity.copy()

    agent = model.ThresholdAgent(tuple(eps), tuple(Q), action_dim)

    return agent

NUM_STORE = 3


if __name__ == "__main__":
    st.title("Tối Ưu Hóa Chuỗi Cung Ứng Bằng Phương Pháp Học Tăng Cường Sâu")
    agent_type = st.selectbox("Chủ thể", ('(zeta, Q)','Cơ sở', 'Mô hình đề xuất', 'Đa chủ thể'))
    st.write("Tính chất đơn hàng")
    trend_demand = st.checkbox("Đơn hàng có xu hướng")
    var_demand = st.checkbox("Đơn hàng dao động mạnh")

    device = 'cuda' if torch.cuda.is_available() else "cpu"

    if agent_type == "(zeta, Q)":
        env = envs.supply_chain.SupplyChain(
        m_demand=trend_demand, 
        v_demand=var_demand)
        agent = create_agent(env)


    if agent_type == 'Cơ sở':
        env = envs.supply_chain.SupplyChain(
        m_demand=trend_demand, v_demand=var_demand)
        net = model.A2CModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        net.load_state_dict(torch.load('model/vpg/vpg_model.dat'))

    if agent_type == 'Mô hình đề xuất':
        env = envs.supply_chain.SupplyChain(
        m_demand=trend_demand, v_demand=var_demand)
        net = model.MatrixModel2(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        net.load_state_dict(torch.load('model/matrix_vpg/matrix_vpg_model.dat'))
    
    if agent_type == 'Đa chủ thể':
        env = envs.supply_chain.SupplyChainRetailer()
        retailer_net = model.A2CModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        retailer_net.load_state_dict(torch.load('model/2_agent/retailer_model.dat'))
        retailer_agent = model.NormalAgent(retailer_net, device=device)
        env = envs.supply_chain.SupplyChainWareHouse(
            m_demand=trend_demand, v_demand=var_demand,
            retailer_agent=retailer_agent,
            break_sp=True
        )
        net = model.A2CModel(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        net.load_state_dict(torch.load('model/2_agent/warhouse_model.dat'))

    

    total_reward = 0
    obs = env.reset()
    t = 0
    
    demand_df = pd.DataFrame([], columns=['Cửa hàng 1','Cửa hàng 2','Cửa hàng 3'])
    act_df = pd.DataFrame([], columns=["Sản xuất", "Vận chuyển 1", "Vận chuyển 2", "Vận chuyển 3"])
    inv_df = pd.DataFrame([], columns=["Nhà phân phối", "Đơn vị bán lẻ 1", "Đơn vị bán lẻ 2", "Đơn vị bán lẻ 3"])

    speed = st.slider("Tốc độ mô phỏng", min_value=0.1, max_value=2.0, step=0.1)
    col1, col2 = st.beta_columns(2)
    with col1:
        run_btn = st.button('Chạy mô phỏng')
    with col2:
        stop_btn = st.button('Dừng mô phỏng')
    
    st.write("Thời điểm")
    period_st = st.empty()
    period_st.write(t)
    
    st.header("Chuỗi Cung Ứng")
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
        st.write("Sản xuất")
        st.write("<font color=#ffffff>|</font>", unsafe_allow_html=True)
        plant = st.empty()
        plant.write("<font color=0>0</font>", unsafe_allow_html=True)
        st.write("<font color=#ffffff>|</font>", unsafe_allow_html=True)
    with col2:
        st.write("Nhà phân phối")
        st.write("<font color=#ffffff>|</font>", unsafe_allow_html=True)
        warehouse = st.empty()
        warehouse.write("<font color=0>0</font>", unsafe_allow_html=True)
        st.write("<font color=#ffffff>|</font>", unsafe_allow_html=True)
    with col3:
        st.write("Vận chuyển")
        deliveries = [st.empty() for i in range(3)]
        for i, delivery in enumerate(deliveries):
            delivery.write(f"0")
    with col4:
        st.write("Đơn vị bán lẻ")
        retailers = [st.empty() for i in range(3)]
        for i, retailer in enumerate(retailers):
            retailer.write("<font color=0>0</font>", unsafe_allow_html=True)
    with col5:
        st.write("Đơn hàng yêu cầu")
        demands = [st.empty() for i in range(3)]
        for i, demand in enumerate(demands):
            demand.write("<font color=0>0</font>", unsafe_allow_html=True)        

    cols = st.beta_columns(6)
    labels = ['Lợi nhuận', 'Doanh thu', "Sản xuất", 'Vận chuyển', 'Lưu trữ', 'Trễ hàng']
    cost_vals = []
    for i, col in enumerate(cols):
        with col:
            st.write(labels[i])
            temp = st.empty()
            temp.write(0)
            cost_vals.append(temp)
    
    st.header("Biểu đồ đơn hàng yêu cầu")
    demand_chart = st.empty()
    demand_chart.line_chart(demand_df)
    st.header("Biểu đồ hành động của chủ thể")
    act_chart = st.empty()
    act_chart.line_chart(act_df)
    st.header("Biểu đồ tình trạng kho hàng của chuỗi cung ứng")
    inv_chart = st.empty()
    inv_chart.line_chart(inv_df)
    
    def update_sc(inv, action=np.zeros(NUM_STORE+1), demand=np.zeros(NUM_STORE)):
        plant.write("<font color='green'>%+.2f</font>"%action[0], unsafe_allow_html=True)
        warehouse.write("<font color=0>%.2f</font>"%inv[0], unsafe_allow_html=True)
        for i in range(NUM_STORE):
            retailers[i].write("<font color=0>%.2f</font>"%inv[i+1], unsafe_allow_html=True)
            deliveries[i].write("<font color='blue'>%.2f</font>"%action[i+1], unsafe_allow_html=True)
            demands[i].write("<font color='red'>-%.2f</font>"%demand[i], unsafe_allow_html=True)
        time.sleep(2.1-speed)

    while run_btn:
        t += 1
        old_obs = obs.copy()
        period_st.write(t)
        update_sc(obs[:NUM_STORE+1])
        
        if agent_type == '(zeta, Q)':
            action = agent.get_action(obs)  
        else:
            obs_v = torch.FloatTensor([obs]).to(device)
            mu_v, _ = net(obs_v)
            action = mu_v.squeeze(dim=0).cpu().data.numpy()

        demand = env.demand              
        obs, reward, done, (action_t, inv) = env.step(action)
        
        update_sc(old_obs[:NUM_STORE+1], action_t)
        
        update_sc(old_obs[:NUM_STORE+1]+action_t)

        update_sc(old_obs[:NUM_STORE+1]+action_t, demand=demand)
        
        total_reward += reward
        color = 'green'
        if total_reward <= 0:
            color = 'red'
        cost_vals[0].write("<font color='%s'>%+.2f</font>"%(color, total_reward), unsafe_allow_html=True)
        costs = env.get_result()
        cost_vals[1].write("<font color='green'>%+.2f</font>"%costs[0], unsafe_allow_html=True)
        for i, cost in enumerate(costs[1:]):
            cost_vals[i+2].write("<font color='red'>-%.2f</font>"%cost, unsafe_allow_html=True)
            
        demand_df.loc[len(demand_df)] = demand
        demand_chart.line_chart(demand_df)

        act_df.loc[len(act_df)] = action_t
        act_chart.line_chart(act_df)

        inv_df.loc[len(inv_df)] = inv
        inv_chart.line_chart(inv_df)

        if done or stop_btn:
            
            break

    update_sc(obs[:NUM_STORE+1])
import math

import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable


from accelerating import soft_voltage_transform, hard_voltage_transform
from spike_neuron import act_fun_adp
from spike_activations import get_activation

import accelerating
import surrogate


def get_neuro(type, **kwargs):
    if type == "prelu":
        return ReluNode(**kwargs)
    # elif type == "lif":
    #     return LIFNeuronLayer1D(**nonlinear_config)
    elif type == "plif":
        return PLIFNode(**kwargs)
    elif type == "alif":
        return ALIFNode(**kwargs)


def multi_normal_initilization(param, means, stds):
    shape_list = param.shape
    if len(shape_list) == 1:
        num_total = shape_list[0]
    elif len(shape_list) == 2:
        num_total = shape_list[0]*shape_list[1]

    num_per_group = num_total // len(means)
    # if num_total%len(means) != 0: 
    num_last_group = num_total%len(means)
    a = []
    for i in range(len(means)):
        a = a + np.random.normal(means[i],stds[i],size=num_per_group).tolist()
        if i == len(means):
            a = a + np.random.normal(means[i],stds[i],size=num_per_group+num_last_group).tolist()
    p = np.array(a).reshape(shape_list)
    with torch.no_grad():
        param.copy_(torch.from_numpy(p).float())
    return param


class ReluNode(torch.nn.Module):
    def __init__(self, no_spiking=False):
        super(ReluNode, self).__init__()

        self.no_spiking = no_spiking
        self.relu = torch.nn.PReLU()

        self.neuro_states_init = False # neuro stats which are batch-wise
    
    def init_neuro_states(self, x):
        self.output = Variable(torch.zeros(*x.shape).type_as(x))
    
    def get_neuro_states(self, x, time_step):
        if time_step == 0 and self.neuro_states_init == False:
            self.init_neuro_states(x)
            self.neuro_states_init = True  # set to indicate it is true for current batch
        
        return self.output, None
    
    def forward(self, x, time_step):
        if time_step == 0:
            if self.neuro_states_init == False:
                self.init_neuro_states(x)
            else:
                self.neuro_states_init = False  # set to indicate init is needed for the next batch

        self.output = self.relu(x)
        if self.no_spiking:
            return self.output
        else:
            return self.output, None

class LIFNeuronLayer1D(torch.nn.Module):
    def __init__(self, size, leak, threshold, activation_name, no_spiking=False):
        super().__init__()
        self.size = size
        
        self.membrane = None
        self.last_spike_time = None

        self.leak = leak
        self.threshold = threshold
        self.activation_func = get_activation(activation_name)

        self.no_spiking = no_spiking

    def forward(self, input, time_step):
        batch_size = input.shape[0]
        if time_step == 0:
            self._init_neuros(batch_size, input)
        
        if self.no_spiking:
            self.membrane = self.leak * self.membrane + input
            return None, self.membrane
        else:
            self.membrane = self.leak * self.membrane + input - self.threshold * self.output
            self.output = self.activation_func(self.membrane, 
                                            self.threshold, 
                                            self.last_spike_time)
            self.last_spike_time = self.last_spike_time.masked_fill(self.output.bool(), time_step)
            return self.output, self.membrane
	
    def _init_neuros(self, batch_size, input):
        self.membrane = torch.zeros(batch_size, self.size).type_as(input)

        if not self.no_spiking:
            self.output = torch.zeros(batch_size, self.size).type_as(input)

            last_spike_time = torch.tensor([-1]*batch_size*self.size)
            self.last_spike_time = last_spike_time.view(batch_size, self.size).type_as(input)


class LIFNeuronLayer2D(torch.nn.Module):
    def __init__(self, leak, threshold, activation_name, no_spiking=False):
        super().__init__()
        
        self.membrane = None
        self.last_spike_time = None

        self.leak = leak
        self.threshold = threshold
        self.activation_func = get_activation(activation_name)

        self.no_spiking = no_spiking

    def forward(self, x, time_step):
        if time_step == 0:
            self._init_neuros(x)
        
        if self.no_spiking:
            self.membrane = self.leak * self.membrane + x
            return None, self.membrane
        else:
            self.membrane = self.leak * self.membrane + x - self.threshold * self.output
            self.output = self.activation_func(self.membrane, 
                                            self.threshold, 
                                            self.last_spike_time)
            self.last_spike_time = self.last_spike_time.masked_fill(self.output.bool(), time_step)
            
            return self.output, self.membrane
	
    def _init_neuros(self, x):
        self.membrane = torch.zeros(*x.shape).type_as(x)
        self.output = torch.zeros(*x.shape).type_as(x)

        self.last_spike_time = -1 * torch.ones(*x.shape).type_as(x)


class ALIFNode(torch.nn.Module):
    def __init__(self, input_dim, beta=0.184, b_j0=1.6, R_m=1.0,
                 tauM=20, tauM_inital_std=5,
                 tauAdp_inital=200, tauAdp_inital_std=50,
                 tau_initializer='normal',
                 dt=1, is_adaptive=1,
                 no_spiking=False):
        super(ALIFNode, self).__init__()

        self.b_j0 = b_j0
        self.R_m = R_m
        self.dt = dt

        self.is_adaptive = is_adaptive
        self.no_spiking = no_spiking

        self.tauM_init = tauM
        self.tauM_inital_std = tauM_inital_std
        self.tau_initializer = tau_initializer

        if self.no_spiking == False:
            self.beta = beta if self.is_adaptive else 0.
            self.tauAdp_inital = tauAdp_inital
            self.tauAdp_inital_std = tauAdp_inital_std

        self.tau_m = torch.nn.Parameter(torch.Tensor(input_dim))
        if self.no_spiking == False:
            self.tau_adp = torch.nn.Parameter(torch.Tensor(input_dim))

        if self.tau_initializer == 'normal':
            torch.nn.init.normal_(self.tau_m, self.tauM_init, self.tauM_inital_std)
            if self.no_spiking == False:
                torch.nn.init.normal_(self.tau_adp, self.tauAdp_inital, self.tauAdp_inital_std)
        elif self.tau_initializer == 'multi_normal':
            self.tau_m = multi_normal_initilization(self.tau_m, self.tauM_init, self.tauM_inital_std)
            if self.no_spiking == False:
                self.tau_adp = multi_normal_initilization(self.tau_adp, self.tauAdp_inital, self.tauAdp_inital_std)

        self.neuro_states_init = False  # neuro stats which are batch-wise

    def parameters(self):
        if self.no_spiking == False:
            return [self.tau_m,self.tau_adp]
        else:
            return [self.tau_m]
    
    def init_neuro_states(self, x):
        # self.mem = (torch.rand(batch_size,self.output_dim)*self.b_j0)
        self.mem = Variable(torch.zeros(*x.shape).type_as(x) * self.b_j0)
        self.spike = Variable(torch.zeros(*x.shape).type_as(x))
        if self.no_spiking == False:
            self.b = Variable(torch.ones(*x.shape).type_as(x) * self.b_j0)
    
    def get_neuro_states(self, x, time_step):
        if time_step == 0 and self.neuro_states_init == False:
            self.init_neuro_states(x)
            self.neuro_states_init = True  # set to indicate it is true for current batch

        return self.spike, self.mem
    
    def forward(self, x, step):
        # print(f"input_spike shape: {input_spike.shape}")
        if step == 0:
            if self.neuro_states_init == False:
                self.init_neuro_states(x)
            else:
                self.neuro_states_init = False  # set to indicate init is needed for the next batch

        alpha = torch.exp(-1. * self.dt / self.tau_m)
        
        if self.no_spiking:
            self.mem = self.mem * alpha + (1 - alpha) * self.R_m * x
            return self.mem
        else:
            ro = torch.exp(-1. * self.dt / self.tau_adp)
            self.b = ro * self.b + (1 - ro) * self.spike
            B = self.b_j0 + self.beta * self.b
            
            self.mem = self.mem * alpha + (1 - alpha) * self.R_m * x - B * self.spike * self.dt
            # print(f"inputs_ shape: {inputs_.shape}")
            # spike = F.relu(inputs_)  
            self.spike = act_fun_adp(self.mem-B)
            # print(f"self.spike shape: {self.spike.shape}")
            return self.spike, self.mem


class BaseNode(torch.nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=None, detach_reset=False, monitor_state=False):
        '''
        * :ref:`API in English <BaseNode.__init__-en>`
        .. _BaseNode.__init__-cn:
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :param detach_reset: 是否将reset过程的计算图分离
        :param monitor_state: 是否设置监视器来保存神经元的电压和释放的脉冲。
            若为 ``True``，则 ``self.monitor`` 是一个字典，键包括 ``v`` 和 ``s``，分别记录电压和输出脉冲。
            对应的值是一个链表。为了节省显存（内存），列表中存入的是原始变量转换为 ``numpy`` 数组后的值。
            还需要注意，``self.reset()`` 函数会清空这些链表
        可微分SNN神经元的基类神经元。
        * :ref:`中文API <BaseNode.__init__-cn>`
        .. _BaseNode.__init__-en:
        :param v_threshold: threshold voltage of neurons
        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :param detach_reset: whether detach the computation graph of reset
        :param detach_reset: whether detach the computation graph of reset 
        
        :param monitor_state: whether to set a monitor to recode voltage and spikes of neurons.
            If ``True``, ``self.monitor`` will be a dictionary with key ``v`` for recording voltage and ``s`` for
            recording spikes. And the value of the dictionary is lists. To save memory, the elements in lists are ``numpy``
            array converted from origin data. Besides, ``self.reset()`` will clear these lists in the dictionary
        This class is the base class of differentiable spiking neurons.
        '''
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        if self.v_reset is None:
            self.init_v = 0
        else:
            self.init_v = self.v_reset
        self.surrogate_function = surrogate_function
        if monitor_state:
            self.monitor = {'v': [], 's': []}
        else:
            self.monitor = False

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def set_monitor(self, monitor_state=True):
        '''
        * :ref:`API in English <BaseNode.set_monitor-en>`
        .. _BaseNode.set_monitor-cn:
        :param monitor_state: ``True`` 或 ``False``，表示开启或关闭monitor
        :return: None
        设置开启或关闭monitor。
        * :ref:`中文API <BaseNode.set_monitor-cn>`
        .. _BaseNode.set_monitor-en:
        :param monitor_state: ``True`` or ``False``, which indicates turn on or turn off the monitor
        :return: None
        Turn on or turn off the monitor.
        '''
        if monitor_state:
            self.monitor = {'v': [], 's': []}
        else:
            self.monitor = False

    def spiking(self):
        '''
        * :ref:`API in English <BaseNode.spiking-en>`
        .. _BaseNode.spiking-cn:
        :return: 神经元的输出脉冲
        根据当前神经元的电压、阈值、重置电压，计算输出脉冲，并更新神经元的电压。
        * :ref:`中文API <BaseNode.spiking-cn>`
        .. _BaseNode.spiking-en:
        :return: out spikes of neurons
        Calculate out spikes of neurons and update neurons' voltage by their current voltage, threshold voltage and reset voltage.
        '''
        
        spike = self.surrogate_function(self.v - self.v_threshold)
        if self.monitor:
            if self.monitor['v'].__len__() == 0:
                # 补充在0时刻的电压
                if self.v_reset is None:
                    self.monitor['v'].append(self.v.data.cpu().numpy().copy() * 0)
                else:
                    self.monitor['v'].append(self.v.data.cpu().numpy().copy() * self.v_reset)

            self.monitor['v'].append(self.v.data.cpu().numpy().copy())
            self.monitor['s'].append(spike.data.cpu().numpy().copy())

        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
            
        if self.v_reset is None:
            if self.surrogate_function.spiking:
                self.v = soft_voltage_transform(self.v, spike_d, self.v_threshold)
            else:
                self.v = self.v - spike_d * self.v_threshold
        else:
            if self.surrogate_function.spiking:
                self.v = hard_voltage_transform(self.v, spike_d, self.v_reset)
            else:
                self.v = self.v * (1 - spike_d) + self.v_reset * spike_d

        if self.monitor:
            self.monitor['v'].append(self.v.data.cpu().numpy().copy())

        return spike

    def forward(self, dv: torch.Tensor):
        '''
        * :ref:`API in English <BaseNode.forward-en>`
        .. _BaseNode.forward-cn:
        :param dv: 输入到神经元的电压增量
        :return: 神经元的输出脉冲
        子类需要实现这一函数。
        * :ref:`中文API <BaseNode.forward-cn>`
        .. _BaseNode.forward-en:
        :param dv: increment of voltage inputted to neurons
        :return: out spikes of neurons
        Subclass should implement this function.
        '''
        raise NotImplementedError

    def reset(self):
        '''
        * :ref:`API in English <BaseNode.reset-en>`
        .. _BaseNode.reset-cn:
        :return: None
        重置神经元为初始状态，也就是将电压设置为 ``v_reset``。
        如果子类的神经元还含有其他状态变量，需要在此函数中将这些状态变量全部重置。
        * :ref:`中文API <BaseNode.reset-cn>`
        .. _BaseNode.reset-en:
        :return: None
        Reset neurons to initial states, which means that set voltage to ``v_reset``.
        Note that if the subclass has other stateful variables, these variables should be reset by this function.
        '''
        if self.v_reset is None:
            self.v = 0
        else:
            self.v = self.v_reset
        if self.monitor:
            self.monitor = {'v': [], 's': []}


class PLIFNode(BaseNode):
    def __init__(self, init_tau=2.0, v_threshold=1.0, v_reset=0.0, 
                 detach_reset=True, surrogate_function=surrogate.ATan(), 
                 monitor_state=False, no_spiking=False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)
        init_w = - math.log(init_tau - 1)
        self.w = torch.nn.Parameter(torch.tensor(init_w, dtype=torch.float))
        self.no_spiking = no_spiking

        self.neuro_states_init = False

    def forward(self, dv: torch.Tensor, time_step):
        if time_step == 0:
            if self.neuro_states_init == False:
                self.init_neuro_states(dv)
            else:
                self.neuro_states_init = False  # set to indicate init is needed for the next batch
        
        if self.v_reset is None:
            # self.v += dv - self.v * self.w.sigmoid()
            self.v = self.v + (dv - self.v) * self.w.sigmoid()
        else:
            # self.v += dv - (self.v - self.v_reset) * self.w.sigmoid()
            self.v = self.v + (dv - (self.v - self.v_reset)) * self.w.sigmoid()
        
        if self.no_spiking:
            return self.v
        else:
            self.s = self.spiking()
            return self.s, self.v
            
    def get_neuro_states(self, dv: torch.Tensor, time_step):
        if time_step == 0 and self.neuro_states_init is False:
            self.init_neuro_states(dv)
            self.neuro_states_init = True  # set to indicate it is true for current batch

        return self.s, self.v

    def init_neuro_states(self, dv):
        self.v = Variable(torch.ones(*dv.shape).type_as(dv) * self.init_v)
        self.s = Variable(torch.zeros(*dv.shape).type_as(dv))

    def tau(self):
        return 1 / self.w.data.sigmoid().item()

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau()}'
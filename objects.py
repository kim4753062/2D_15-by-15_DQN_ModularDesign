import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from collections import deque

################################################################################################
############################## Definition: Data Structure Class ################################
################################################################################################

################################### Class: Placement sample ####################################
# WellPlacementSample Class contains information of One full sequence of simulation sample.
class WellPlacementSample:
    def __init__(self, args):
        self.args = args
        self.well_loc_map = [[[0 for i in range(0, args.gridnum_x)] for j in range(0, args.gridnum_y)]]
        self.well_loc_list = []
        self.PRESSURE_map = [[[args.initial_PRESSURE for i in range(0, args.gridnum_x)] for j in range(0, args.gridnum_y)]]
        self.SOIL_map = [[[args.initial_SOIL for i in range(0, args.gridnum_x)] for j in range(0, args.gridnum_y)]]
        self.income = []

################################### Class: Experience sample ###################################
# Experience Class contains One set of (s, a, r, s').
class Experience:
    def __init__(self, args):
        self.args = args
        self.current_state = list
        self.current_action = None
        self.reward = None
        self.next_state = list
        # 2023-10-10: Implementation of PER - Initial TD-error ==args.td_err_init
        if args.activate_PER == True:
            self.td_err = args.td_err_init

    def __transform__(self):
        self.current_state = torch.tensor(data=self.current_state, dtype=torch.float, device=self.args.device, requires_grad=False)
        self.current_action = torch.tensor(data=self.current_action, dtype=torch.float, device=self.args.device, requires_grad=False)
        self.reward = torch.tensor(data=self.reward, dtype=torch.float, device=self.args.device, requires_grad=False)
        self.next_state = torch.tensor(data=self.next_state, dtype=torch.float, device=self.args.device, requires_grad=False)

    def transform(self):
        self.__transform__()


###################### Class: Experience sample (Dataset for DataLoader) #######################
class Experience_list(Dataset):
    def __init__(self, args):
        self.args = args
        self.exp_list = deque()

    def __len__(self):
        return len(self.exp_list)

    def __getitem__(self, idx):
        return self.exp_list[idx].current_state


################################################################################################
################################## Definition: CNN Structure ###################################
################################################################################################

######################################### CNN Structure ########################################
# Modification of ResNet structure
# https://cryptosalamander.tistory.com/156
class BasicBlock(nn.Module):  # Residual block
    mul = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # x를 그대로 더해주기 위함
        self.shortcut = nn.Sequential()

        # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        if stride != 1:  # x와
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

        self.conv1.apply(self._init_weight)
        self.conv2.apply(self._init_weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 필요에 따라 layer를 Skip
        out = F.relu(out)
        return out

    def _init_weight(self, layer, init_type="Xavier"):
        if isinstance(layer, nn.Conv2d):
            if init_type == "Xavier":
                torch.nn.init.xavier_uniform_(layer.weight)
            elif init_type == "He":
                torch.nn.init.kaiming_uniform_(layer.weight)

class DQN(nn.Module):
    def __init__(self, args, block):
        '''
        이 두 코드는 형태가 조금 다르다.
        == super(MyModule,self).__init__()
        == super().__init__()
        super 안에 현재 클래스를 명시해준 것과 아닌 것으로 나눌 수 있는데 이는 기능적으론 아무런 차이가 없다.
        파생클래스와 self를 넣어서 현재 클래스가 어떤 클래스인지 명확하게 표시 해주는 용도이다.
        super(파생클래스, self).__init__()
        '''
        '''
        Structure of DQN (Basic CNN Structure)
        (1) input (Dim: (batch_size, len(state), gridnum_y, gridnum_x))
        (2) Conv1-BatchNorm-ReLU (In&Out Dim: (batch_size, 32, gridnum_y, gridnum_x))
        (3) Conv2-BatchNorm-ReLU (In&Out Dim: (batch_size, 32, gridnum_y, gridnum_x))
        (4) Conv3-BatchNorm-ReLU (In&Out Dim: (batch_size, 32, gridnum_y, gridnum_x))
        (5) Fully-Connected Layer (In Dim: (batch_size, gridnum_y * gridnum_x * self.out_channel), Out Dim: (batch_size, round(args.gridnum_y*args.gridnum_x*self.out_channel/2)))
        (5) Fully-Connected Layer (In Dim: (batch_size, round(args.gridnum_y*args.gridnum_x*self.out_channel/2)), Out Dim: (batch_size, round(args.gridnum_y*args.gridnum_x*1))) << Objective: Q-value of each action at current state
        Action masking will be done by modified Boltzmann policy
        '''
        super(DQN, self).__init__()
        if args.input_flag:
            self.num_of_channels = len(args.input_flag)
        else:
            self.num_of_channels = 3

        self.in_planes = 0
        self.out_channel = 32

        self.conv1 = nn.Conv2d(in_channels=self.num_of_channels, out_channels=self.out_channel, kernel_size=3,
                               stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(self.out_channel)

        # self.layer1 = self.make_layer(block=block, out_planes=self.out_channel, num_blocks=1, stride=1)

        self.conv2 = nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1,
                               padding='same')
        self.bn2 = nn.BatchNorm2d(self.out_channel)

        self.conv3 = nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1,
                               padding='same')
        self.bn3 = nn.BatchNorm2d(self.out_channel)

        self.linear1 = nn.Linear(in_features=args.gridnum_y * args.gridnum_x * self.out_channel,
                                 out_features=round(args.gridnum_y * args.gridnum_x * self.out_channel / 2))
        # # Fully-connected layer BatchNorm Test
        # self.lin_bn1 = nn.BatchNorm1d(num_features=round(args.gridnum_y*args.gridnum_x*self.out_channel/2))

        self.linear2 = nn.Linear(in_features=round(args.gridnum_y * args.gridnum_x * self.out_channel / 2),
                                 out_features=args.gridnum_y * args.gridnum_x * 1)

        self.conv1.apply(self._init_weight)
        self.conv2.apply(self._init_weight)
        self.conv3.apply(self._init_weight)

        self.linear1.apply(self._init_weight)
        self.linear2.apply(self._init_weight)

    def make_layer(self, block, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = F.relu(out)
        out = self.bn3(out)

        out = out.view(x.shape[0], -1)  # Flatten
        out = self.linear1(out)
        # # Fully-connected layer BatchNorm
        # out = self.lin_bn1(out)
        # out = F.relu(out)

        out = self.linear2(out)

        return out.reshape(-1, 1, x.shape[2], x.shape[3])

    # Weight initializer method
    def _init_weight(self, layer, init_type="He"):
        if isinstance(layer, nn.Conv2d):
            if init_type == "Xavier":
                torch.nn.init.xavier_uniform_(layer.weight)
            elif init_type == "He":
                torch.nn.init.kaiming_uniform_(layer.weight)

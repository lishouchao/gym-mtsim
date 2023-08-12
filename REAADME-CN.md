# gym-mtsim: OpenAI Gym - MetaTrader 5 Simulator

`MtSim` is a simulator for the [MetaTrader 5](https://www.metatrader5.com) trading platform alongside an [OpenAI Gym](https://github.com/openai/gym) environment for reinforcement learning-based trading algorithms. `MetaTrader 5` is a **multi-asset** platform that allows trading **Forex**, **Stocks**, **Crypto**, and Futures. It is one of the most popular trading platforms and supports numerous useful features, such as opening demo accounts on various brokers.

MtSim'是[MetaTrader 5]（https://www.metatrader5.com）交易平台的模拟器，以及用于强化学习的交易算法的[OpenAI Gym]（https://github.com/openai/gym）环境。“MetaTrader 5”是一个**多资产**平台，允许交易**外汇**，**股票**，**加密**和期货。它是最受欢迎的交易平台之一，支持许多有用的功能，例如在各种经纪人上开设模拟账户。

The simulator is separated from the Gym environment and can work independently. Although the Gym environment is designed to be suitable for RL frameworks, it is also proper for backtesting and classic analysis.

模拟器与Gym环境分开，可以独立工作。虽然 Gym 环境设计为适用于 RL 框架，但它也适用于回测和经典分析。

The goal of this project was to provide a *general-purpose*, *flexible*, and *easy-to-use* library with a focus on *code readability* that enables users to do all parts of the trading process through it from 0 to 100. So, `gym-mtsim` is not just a testing tool or a Gym environment. It is a combination of a **real-world** simulator, a **backtesting** tool with *high detail visualization*, and a **Gym environment** appropriate for RL/classic algorithms.

该项目的目标是提供一个*通用*，*灵活*和*易于使用*的库，重点是*代码可读性*，使用户能够通过它完成交易过程的所有部分，从0到100。因此，“gym-mtsim”不仅仅是一个测试工具或Gym环境。它是**真实世界**模拟器，具有*高细节可视化*的**回测**工具以及适用于RL/经典算法的**健身房环境**的组合。

**Note:** For beginners, it is recommended to check out the [gym-anytrading](https://github.com/AminHP/gym-anytrading) project.

**注意：**对于初学者，建议查看[gym-anytrading]（[https://github.com/AminHP/gym-anytrading）项目。](https://github.com/AminHP/gym-anytrading%EF%BC%89%E9%A1%B9%E7%9B%AE%E3%80%82)

## Prerequisites 先决条件

### Install MetaTrader 5

Download and install MetaTrader 5 software from [here](https://www.metatrader5.com/en/download).

Open a demo account on any broker. By default, the software opens a demo account automatically after installation.

Explore the software and try to get familiar with it by trading different symbols in both **hedged** and **unhedged** accounts.

从[这里]（https://www.metatrader5.com/en/download）下载并安装MetaTrader 5软件。

在任何经纪商上开设模拟账户。默认情况下，软件在安装后会自动打开一个模拟账户。

探索该软件，并尝试通过在**对冲**和**未对冲**账户中交易不同的符号来熟悉它。

### Install gym-mtsim

#### Via PIP

```bash
pip install gym-mtsim
```

#### From Repository

```bash
git clone https://github.com/AminHP/gym-mtsim
cd gym-mtsim
pip install -e .

## or

pip install --upgrade --no-deps --force-reinstall https://github.com/AminHP/gym-mtsim/archive/main.zip
```

### Install stable-baselines3

This package is required to run some examples. Install it from [here](https://github.com/DLR-RM/stable-baselines3#installation).

## Components

### 1. SymbolInfo 符号信息

This is a data class that contains the essential properties of a symbol. Try to get fully acquainted with [these properties](https://github.com/AminHP/gym-mtsim/blob/main/gym_mtsim/metatrader/symbol.py) in case they are unfamiliar. There are plenty of resources that provide good explanations.

这是一个包含符号基本属性的数据类。尝试完全熟悉[这些属性]，以防它们不熟悉。有很多资源可以提供很好的解释。

### 2. Order

This is another data class that consists of information of an order. Each order has the following properties:

这是另一个由订单信息组成的数据类。每个订单具有以下属性：

> `id`: A unique number that helps with tracking orders. 有助于跟踪订单的唯一编号。
> 
> `type`: An enum that specifies the type of the order. It can be either **Buy** or **Sell**. 指定订单类型的枚举。它可以是**买入**或**卖出**。
> 
> `symbol`: The symbol selected for the order.
> 
> `volume`: The volume chose for the order. It can be a multiple of *volume_step* between *volume_min* and *volume_max*. 为订单选择的量。它可以是 *volume_step* 的倍数，介于 *volume_min* 和 *volume_max* 之间。
> 
> `fee`: It is a tricky property. In MetaTrader, there is *no* such concept called fee. Each symbol has bid and ask prices, the difference between which represents the **fee**. Although MetaTrader API provides these bid/ask prices for the recent past, it is not possible to access them for the distant past. Therefore, the **fee** property helps to manage the mentioned difference. 这是一个棘手的属性。在MetaTrader中，*没有*这样的概念称为费用。每个交易品种都有买入价和卖出价，两者之间的差额代表**费用**。尽管 MetaTrader API 提供了近期的买入价/卖出价，但无法在遥远的过去访问它们。因此，**费用**属性有助于管理上述差异。
> 
> `entry_time`: The time when the order was placed. 下订单的时间。
> 
> `entry_price`: The **close** price when the order was placed. 下订单时的**收盘价。
> 
> `exit_time`: The time when the order was closed. 订单关闭的时间。
> 
> `exit_price`: The **close** price when the order was closed. 订单关闭时的 **收盘价。
> 
> `profit`: The amount of profit earned by this order so far. 到目前为止，此订单赚取的利润金额。
> 
> `margin`: The required amount of margin for this order. 此订单所需的保证金金额。
> 
> `closed`: A boolean that specifies whether this order is closed or not. 一个布尔值，它指定此订单是否已关闭。

### 3. MtSimulator

This is the core class that simulates the main parts of MetaTrader. Most of its public properties and methods are explained here. But feel free to take a look at the complete [source code](https://github.com/AminHP/gym-mtsim/blob/main/gym_mtsim/simulator/mt_simulator.py).

这是模拟 MetaTrader 主要部分的核心类。此处介绍了其大多数公共属性和方法。但是请随意查看完整的[源代码]。

- Properties:
  
  > `unit`: The unit currency. It is usually *USD*, but it can be anything the broker allows, such as *EUR*. 单位货币。它通常是*美元*，但它可以是经纪人允许的任何内容，例如*欧元*。
  > 
  > `balance`: The amount of money before taking into account any open positions. 考虑任何未平仓头寸之前的金额。
  > 
  > `equity`: The amount of money, including the value of any open positions. 金额，包括任何未平仓头寸的价值。
  > 
  > `margin`: The amount of money which is required for having positions opened. 开仓所需的金额。
  > 
  > `leverage`: The leverage ratio. 杠杆率。
  > 
  > `free_margin`: The amount of money that is available to open new positions. 可用于开立新头寸的资金金额。
  > 
  > `margin_level`: The ratio between **equity** and **margin**. **净值**和**保证金**之间的比率。
  > 
  > `stop_out_level`: If the **margin_level** drops below **stop_out_level**, the most unprofitable position will be closed automatically by the broker. 如果**margin_level**跌破**stop_out_level**，经纪商将自动平仓最无利可图的头寸。
  > 
  > `hedge`: A boolean that specifies whether hedging is enabled or not. 一个布尔值，用于指定是否启用对冲。
  > 
  > `symbols_info`: A dictionary that contains symbols' information. 包含符号信息的字典。
  > 
  > `symbols_data`: A dictionary that contains symbols' OHLCV data. 包含符号的 OHLCV 数据的字典。
  > 
  > `orders`: The list of open orders. 未结订单的列表。
  > 
  > `closed_orders`: The list of closed orders. 已关闭订单的列表。
  > 
  > `current_time`: The current time of the system. 系统的当前时间。
  
- Methods:
  
  > `download_data`: Downloads required data from MetaTrader for a list of symbols in a time range. This method can be overridden in order to download data from servers other than MetaTrader. *Note that this method only works on Windows, as the MetaTrader5 Python package is not available on other platforms.* 从 MetaTrader 下载所需数据，以获取某个时间范围内的交易品种列表。此方法可以被覆盖，以便从MetaTrader以外的服务器下载数据。*请注意，此方法仅适用于Windows，因为MetaTrader5 Python软件包在其他平台上不可用。
  > 
  > `save_symbols`: Saves the downloaded symbols' data to a file. 将下载的符号数据保存到文件中。
  > 
  > `load_symbols`: Loads the symbols' data from a file.
  > 
  > `tick`: Moves forward in time (by a delta time) and updates orders and other related properties. 按时间向前移动（按增量时间）并更新订单和其他相关属性。
  > 
  > `create_order`: Creates a **Buy** or **Sell** order and updates related properties. 创建**买入**或**卖出**订单并更新相关属性。
  > 
  > `close_order`: Closes an order and updates related properties. 关闭订单并更新相关属性。
  > 
  > `get_state`: Returns the state of the system. The result is similar to the *Trading tab* and *History tab* of the *Toolbox window* in MetaTrader software. 返回系统的状态。结果类似于 MetaTrader 软件中 *工具箱窗口* 的 *交易选项卡* 和 *历史选项卡*。
  

### 4. MtEnv

This is the Gym environment that works on top of the *MtSim*. Most of its public properties and methods are explained here. But feel free to take a look at the complete [source code](https://github.com/AminHP/gym-mtsim/blob/main/gym_mtsim/envs/mt_env.py).

这是在*MtSim*之上工作的Gym环境。此处介绍了其大多数公共属性和方法。但是请随意查看完整的[源代码]

- Properties:
  
  > `original_simulator`: An instance of **MtSim** class as a baseline for simulating the system. 作为模拟系统的基线的 **MtSim** 类的实例。
  > 
  > `simulator`: The current simulator in use. It is a copy of the **original_simulator**. 当前正在使用的模拟器。它是**original_simulator**的副本。
  > 
  > `trading_symbols`: The list of symbols to trade. 要交易的交易品种列表。
  > 
  > `time_points`: A list of time points based on which the simulator moves time. The default value is taken from the *pandas DataFrame.Index* of the first symbol in the **trading_symbols** list. 模拟器移动时间所基于的时间点列表。默认值取自 **trading_symbols** 列表中第一个符号的 *pandas DataFrame.Index*。
  > 
  > `hold_threshold`: A probability threshold that controls holding or placing a new order. 控制持有或下新订单的概率阈值。
  > 
  > `close_threshold`: A probability threshold that controls closing an order. 控制关闭订单的概率阈值。
  > 
  > `fee`: A constant number or a callable that takes a *symbol* as input and returns the **fee** based on that. 一个常量数字或可调用对象，它将 *symbol* 作为输入并基于此返回 ** 费用**。
  > 
  > `symbol_max_orders`: Specifies the maximum number of open positions per symbol in hedge trading. 指定对冲交易中每个交易品种的最大未平仓头寸数量。
  > 
  > `multiprocessing_processes`: Specifies the maximum number of processes used for parallel processing. 指定用于并行处理的最大进程数。
  > 
  > `prices`: The symbol prices over time. It is used to calculate signal features and render the environment. 交易品种随时间推移的价格。它用于计算信号特征和渲染环境。
  > 
  > `signal_features`: The extracted features over time. It is used to generate *Gym observations*. 随时间推移提取的特征。它用于生成*Gym观察*。
  > 
  > `window_size`: The number of time points (current and previous points) as the length of each observation's features. 时间点（当前点和先前点）的数量作为每个观测值要素的长度。
  > 
  > `features_shape`: The shape of a single observation's features. 单个观测点要素的形状。
  > 
  > `action_space`: The *Gym action_space* property. It has a complex structure since **stable-baselines** does not support *Dict* or *2D Box* action spaces. The action space is a 1D vector of size `count(trading_symbols) * (symbol_max_orders + 2)`.
  > 
  > *action_space*属性。它具有复杂的结构，因为**稳定基线**不支持*字典*或*2D Box*操作空间。动作空间是大小为“count（trading_symbols） * （symbol_max_orders + 2）”的一维向量。
  > 
  > For each symbol, two types of actions can be performed, closing previous orders and placing a new order. The former is controlled by the first *symbol_max_orders* elements and the latter is controlled by the last two elements. Therefore, the action for each symbol is ***[probability of closing order 1, probability of closing order 2, ..., probability of closing order symbol_max_orders, probability of holding or creating a new order, volume of the new order]***.
  > 
  > 对于每个交易品种，可以执行两种类型的操作，关闭以前的订单和下新订单。前者由前 *symbol_max_orders* 元素控制，后者由后两个元素控制。因此，每个交易品种的操作是 ***[平仓订单 1 的概率、平仓订单 2 的概率、...、平仓订单symbol_max_orders的概率、持有或创建新订单的概率、新订单的交易量]***。
  > 
  > The last two elements specify whether to hold or place a new order and the volume of the new order (positive volume indicates buy and negative volume indicates sell). These elements are a number in range (-∞, ∞), but the probability values must be in the range [0, 1]. This is a problem with **stable-baselines** as mentioned earlier. To overcome this problem, it is assumed that the probability values belong to the [logit](https://en.wikipedia.org/wiki/Logit) function. So, applying the [expit](https://en.wikipedia.org/wiki/Expit) function on them gives the desired probability values in the range [0, 1]. This function is applied in the **step** method of the environment.
  > 
  > 最后两个元素指定是持有还是下新订单以及新订单的交易量（正交易量表示买入，负交易量表示卖出）。这些元素是范围 （-∞， ∞） 中的数字，但概率值必须在 [0， 1] 范围内。如前所述，这是**稳定基线**的问题。为了克服这个问题，假设概率值属于 [logit]（https://en.wikipedia.org/wiki/Logit） 函数。因此，对它们应用 [expit]（https://en.wikipedia.org/wiki/Expit） 函数会给出 [0， 1] 范围内所需的概率值。此函数应用于环境的 **step** 方法。
  > 
  > `observation_space`: The *Gym observation_space* property. Each observation contains information about *balance*, *equity*, *margin*, *features*, and *orders*. The **features** is a window on the *signal_features* from index *current_tick - window_size + 1* to *current_tick*. The **orders** is a 3D array. Its first dimension specifies the symbol index in the *trading_symbols* list. The second dimension specifies the order number (each symbol can have more than one open order at the same time in hedge trading). The last dimension has three elements, *entry_price*, *volume*, and *profit* of corresponding order.
  > 
  > *健身房observation_space*属性。每个观察都包含有关*余额*、*净值*、*保证金*、*特征*和*订单*的信息。**功能**是*signal_features*上的一个窗口，从索引*current_tick - window_size + 1*到*current_tick*。**订单**是一个3D数组。其第一个维度指定 *trading_symbols* 列表中的符号索引。第二个维度指定订单号（在对冲交易中，每个交易品种可以同时有多个未平仓订单）。最后一个维度有三个元素，相应订单的 *entry_price*、*交易量*和*利润*。
  > 
  > `history`: Stores the information of all steps. 存储所有步骤的信息。
  
- Methods:
  
  > `seed`: The typical *Gym seed* method.
  > 
  > `reset`: The typical *Gym reset* method.
  > 
  > `step`: The typical *Gym step* method.
  > 
  > `render`: The typical *Gym render* method. It can render in three modes, **human**, **simple_figure**, and **advanced_figure**.
  > 
  > `close`: The typical *Gym close* method.
  
- Virtual Methods:
  
  > `_get_prices`: It is called in the constructor and calculates symbol **prices**.
  > 
  > `_process_data`: It is called in the constructor and calculates **signal_features**.
  > 
  > `_calculate_reward`: The reward function for the RL agent.
  

## A Simple Example

### MtSim

#### Create a simulator with custom parameters

```python
import pytz
from datetime import datetime, timedelta
from gym_mtsim import MtSimulator, OrderType, Timeframe, FOREX_DATA_PATH


sim = MtSimulator(
    unit='USD',
    balance=10000.,
    leverage=100.,
    stop_out_level=0.2,
    hedge=False,
)

if not sim.load_symbols(FOREX_DATA_PATH):
    sim.download_data(
        symbols=['EURUSD', 'GBPCAD', 'GBPUSD', 'USDCAD', 'USDCHF', 'GBPJPY', 'USDJPY'],
        time_range=(
            datetime(2021, 5, 5, tzinfo=pytz.UTC),
            datetime(2021, 9, 5, tzinfo=pytz.UTC)
        ),
        timeframe=Timeframe.D1
    )
    sim.save_symbols(FOREX_DATA_PATH)
```

#### Place some orders

```python
sim.current_time = datetime(2021, 8, 30, 0, 17, 52, tzinfo=pytz.UTC)

order1 = sim.create_order(
    order_type=OrderType.Buy,
    symbol='GBPCAD',
    volume=1.,
    fee=0.0003,
)

sim.tick(timedelta(days=2))

order2 = sim.create_order(
    order_type=OrderType.Sell,
    symbol='USDJPY',
    volume=2.,
    fee=0.01,
)

sim.tick(timedelta(days=5))

state = sim.get_state()

print(
    f"balance: {state['balance']}, equity: {state['equity']}, margin: {state['margin']}\n"
    f"free_margin: {state['free_margin']}, margin_level: {state['margin_level']}\n"
)
state['orders']
```

```
balance: 10000.0, equity: 10717.58118589908, margin: 3375.480933228619
free_margin: 7342.1002526704615, margin_level: 3.1751271592500743
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Symbol</th>
      <th>Type</th>
      <th>Volume</th>
      <th>Entry Time</th>
      <th>Entry Price</th>
      <th>Exit Time</th>
      <th>Exit Price</th>
      <th>Profit</th>
      <th>Margin</th>
      <th>Fee</th>
      <th>Closed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>USDJPY</td>
      <td>Sell</td>
      <td>2.0</td>
      <td>2021-09-01 00:17:52+00:00</td>
      <td>110.02500</td>
      <td>2021-09-06 00:17:52+00:00</td>
      <td>109.71200</td>
      <td>552.355257</td>
      <td>2000.000000</td>
      <td>0.0100</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>GBPCAD</td>
      <td>Buy</td>
      <td>1.0</td>
      <td>2021-08-30 00:17:52+00:00</td>
      <td>1.73389</td>
      <td>2021-09-06 00:17:52+00:00</td>
      <td>1.73626</td>
      <td>165.225928</td>
      <td>1375.480933</td>
      <td>0.0003</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>

#### Close all orders

```python
order1_profit = sim.close_order(order1)
order2_profit = sim.close_order(order2)

# alternatively:
# for order in sim.orders:
#     sim.close_order(order)

state = sim.get_state()

print(
    f"balance: {state['balance']}, equity: {state['equity']}, margin: {state['margin']}\n"
    f"free_margin: {state['free_margin']}, margin_level: {state['margin_level']}\n"
)
state['orders']
```

```
balance: 10717.58118589908, equity: 10717.58118589908, margin: 0.0
free_margin: 10717.58118589908, margin_level: inf
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Symbol</th>
      <th>Type</th>
      <th>Volume</th>
      <th>Entry Time</th>
      <th>Entry Price</th>
      <th>Exit Time</th>
      <th>Exit Price</th>
      <th>Profit</th>
      <th>Margin</th>
      <th>Fee</th>
      <th>Closed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>USDJPY</td>
      <td>Sell</td>
      <td>2.0</td>
      <td>2021-09-01 00:17:52+00:00</td>
      <td>110.02500</td>
      <td>2021-09-06 00:17:52+00:00</td>
      <td>109.71200</td>
      <td>552.355257</td>
      <td>2000.000000</td>
      <td>0.0100</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>GBPCAD</td>
      <td>Buy</td>
      <td>1.0</td>
      <td>2021-08-30 00:17:52+00:00</td>
      <td>1.73389</td>
      <td>2021-09-06 00:17:52+00:00</td>
      <td>1.73626</td>
      <td>165.225928</td>
      <td>1375.480933</td>
      <td>0.0003</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>

### MtEnv

#### Create an environment

```python
import gym
import gym_mtsim

env = gym.make('forex-hedge-v0')
# env = gym.make('stocks-hedge-v0')
# env = gym.make('crypto-hedge-v0')
# env = gym.make('mixed-hedge-v0')

# env = gym.make('forex-unhedge-v0')
# env = gym.make('stocks-unhedge-v0')
# env = gym.make('crypto-unhedge-v0')
# env = gym.make('mixed-unhedge-v0')
```

- This will create a default environment. There are eight default environments, but it is also possible to create environments with custom parameters.

#### Create an environment with custom parameters

```python
import pytz
from datetime import datetime, timedelta
import numpy as np
from gym_mtsim import MtEnv, MtSimulator, FOREX_DATA_PATH


sim = MtSimulator(
    unit='USD',
    balance=10000.,
    leverage=100.,
    stop_out_level=0.2,
    hedge=True,
    symbols_filename=FOREX_DATA_PATH
)

env = MtEnv(
    original_simulator=sim,
    trading_symbols=['GBPCAD', 'EURUSD', 'USDJPY'],
    window_size=10,
    # time_points=[desired time points ...],
    hold_threshold=0.5,
    close_threshold=0.5,
    fee=lambda symbol: {
        'GBPCAD': max(0., np.random.normal(0.0007, 0.00005)),
        'EURUSD': max(0., np.random.normal(0.0002, 0.00003)),
        'USDJPY': max(0., np.random.normal(0.02, 0.003)),
    }[symbol],
    symbol_max_orders=2,
    multiprocessing_processes=2
)
```

#### Print some information

```python
print("env information:")

for symbol in env.prices:
    print(f"> prices[{symbol}].shape:", env.prices[symbol].shape)

print("> signal_features.shape:", env.signal_features.shape)
print("> features_shape:", env.features_shape)
```

```
env information:
> prices[GBPCAD].shape: (88, 2)
> prices[EURUSD].shape: (88, 2)
> prices[USDJPY].shape: (88, 2)
> signal_features.shape: (88, 6)
> features_shape: (10, 6)
```

#### Trade randomly

```python
observation = env.reset()

while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        # print(info)
        print(
            f"balance: {info['balance']}, equity: {info['equity']}, margin: {info['margin']}\n"
            f"free_margin: {info['free_margin']}, margin_level: {info['margin_level']}\n"
            f"step_reward: {info['step_reward']}"
        )
        break
```

```
balance: 9376.891775198916, equity: 9641.936625205548, margin: 3634.1077619051393
free_margin: 6007.828863300409, margin_level: 2.6531785122131852
step_reward: 140.93306243685583
```

#### Render in *human* mode

```python
state = env.render()

print(
    f"balance: {state['balance']}, equity: {state['equity']}, margin: {state['margin']}\n"
    f"free_margin: {state['free_margin']}, margin_level: {state['margin_level']}\n"
)
state['orders']
```

```
balance: 9376.891775198916, equity: 9641.936625205548, margin: 3634.1077619051393
free_margin: 6007.828863300409, margin_level: 2.6531785122131852
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Symbol</th>
      <th>Type</th>
      <th>Volume</th>
      <th>Entry Time</th>
      <th>Entry Price</th>
      <th>Exit Time</th>
      <th>Exit Price</th>
      <th>Profit</th>
      <th>Margin</th>
      <th>Fee</th>
      <th>Closed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>119</td>
      <td>USDJPY</td>
      <td>Buy</td>
      <td>1.12</td>
      <td>2021-09-02 00:00:00+00:00</td>
      <td>109.93700</td>
      <td>2021-09-03 00:00:00+00:00</td>
      <td>109.71200</td>
      <td>-248.970123</td>
      <td>1120.000000</td>
      <td>0.018884</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>118</td>
      <td>EURUSD</td>
      <td>Buy</td>
      <td>0.24</td>
      <td>2021-09-02 00:00:00+00:00</td>
      <td>1.18744</td>
      <td>2021-09-03 00:00:00+00:00</td>
      <td>1.18772</td>
      <td>-4.355531</td>
      <td>284.985600</td>
      <td>0.000461</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>117</td>
      <td>USDJPY</td>
      <td>Sell</td>
      <td>1.94</td>
      <td>2021-09-01 00:00:00+00:00</td>
      <td>110.02500</td>
      <td>2021-09-03 00:00:00+00:00</td>
      <td>109.71200</td>
      <td>520.155098</td>
      <td>1940.000000</td>
      <td>0.018839</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>116</td>
      <td>GBPCAD</td>
      <td>Sell</td>
      <td>0.21</td>
      <td>2021-09-01 00:00:00+00:00</td>
      <td>1.73728</td>
      <td>2021-09-03 00:00:00+00:00</td>
      <td>1.73626</td>
      <td>-1.784594</td>
      <td>289.122162</td>
      <td>0.001126</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>113</td>
      <td>USDJPY</td>
      <td>Sell</td>
      <td>2.24</td>
      <td>2021-08-30 00:00:00+00:00</td>
      <td>109.91300</td>
      <td>2021-09-01 00:00:00+00:00</td>
      <td>110.02500</td>
      <td>-258.362674</td>
      <td>2240.000000</td>
      <td>0.014903</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>114</th>
      <td>6</td>
      <td>USDJPY</td>
      <td>Sell</td>
      <td>1.03</td>
      <td>2021-05-21 00:00:00+00:00</td>
      <td>108.94500</td>
      <td>2021-05-24 00:00:00+00:00</td>
      <td>108.74000</td>
      <td>173.893295</td>
      <td>1030.000000</td>
      <td>0.021416</td>
      <td>True</td>
    </tr>
    <tr>
      <th>115</th>
      <td>3</td>
      <td>EURUSD</td>
      <td>Buy</td>
      <td>0.86</td>
      <td>2021-05-19 00:00:00+00:00</td>
      <td>1.21744</td>
      <td>2021-05-24 00:00:00+00:00</td>
      <td>1.22150</td>
      <td>352.419311</td>
      <td>1046.998400</td>
      <td>-0.000038</td>
      <td>True</td>
    </tr>
    <tr>
      <th>116</th>
      <td>5</td>
      <td>GBPCAD</td>
      <td>Sell</td>
      <td>0.94</td>
      <td>2021-05-21 00:00:00+00:00</td>
      <td>1.70726</td>
      <td>2021-05-24 00:00:00+00:00</td>
      <td>1.70440</td>
      <td>174.119943</td>
      <td>1330.148695</td>
      <td>0.000629</td>
      <td>True</td>
    </tr>
    <tr>
      <th>117</th>
      <td>1</td>
      <td>GBPCAD</td>
      <td>Buy</td>
      <td>1.45</td>
      <td>2021-05-18 00:00:00+00:00</td>
      <td>1.71128</td>
      <td>2021-05-24 00:00:00+00:00</td>
      <td>1.70440</td>
      <td>-961.496723</td>
      <td>2056.809874</td>
      <td>0.001105</td>
      <td>True</td>
    </tr>
    <tr>
      <th>118</th>
      <td>2</td>
      <td>GBPCAD</td>
      <td>Sell</td>
      <td>0.58</td>
      <td>2021-05-19 00:00:00+00:00</td>
      <td>1.71211</td>
      <td>2021-05-21 00:00:00+00:00</td>
      <td>1.70726</td>
      <td>219.514676</td>
      <td>818.590377</td>
      <td>0.000284</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>119 rows × 12 columns</p>
</div>

#### Render in *simple_figure* mode

- Each *symbol* is illustrated with a separate color.
- The **green**/**red** triangles show successful **buy**/**sell** actions.
- The **gray** triangles indicate that the **buy**/**sell** action has encountered an **error**.
- The **black** vertical bars specify **close** actions.

```python
env.render('simple_figure')
```

![png](file://D:\LSC\gym-mtsim\doc\output_28_0.png?msec=1691858403022)

#### Render in *advanced_figure* mode

- Clicking on a symbol name will hide/show its plot.
- Hovering over points and markers will display their detail.
- The size of triangles indicates their relative volume.

```python
env.render('advanced_figure', time_format="%Y-%m-%d")
```

![png](file://D:\LSC\gym-mtsim\doc\output_30_0.png?msec=1691858403023)

### A Complete Example using `stable-baselines`

```python
import gym
from gym_mtsim import (
    Timeframe, SymbolInfo,
    MtSimulator, OrderType, Order, SymbolNotFound, OrderNotFound,
    MtEnv,
    FOREX_DATA_PATH, STOCKS_DATA_PATH, CRYPTO_DATA_PATH, MIXED_DATA_PATH,
)
from stable_baselines3 import A2C


# env = gym.make('forex-hedge-v0')

model = A2C('MultiInputPolicy', env, verbose=0)
model.learn(total_timesteps=1000)

observation = env.reset()
while True:
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    if done:
        break

env.render('advanced_figure', time_format="%Y-%m-%d")
```

![png](file://D:\LSC\gym-mtsim\doc\output_32_0.png?msec=1691858403025)

## References

- [https://www.mql5.com/en/docs/python_metatrader5](https://www.mql5.com/en/docs/python_metatrader5)
- [https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_forex](https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_forex)
- [https://admiralmarkets.com/education/articles/forex-basics/margin-in-forex-trading-margin-level-vs-margin-call](https://admiralmarkets.com/education/articles/forex-basics/margin-in-forex-trading-margin-level-vs-margin-call)
- [https://www.investopedia.com/articles/forex/12/calculating-profits-and-losses-of-forex-trades.asp](https://www.investopedia.com/articles/forex/12/calculating-profits-and-losses-of-forex-trades.asp)

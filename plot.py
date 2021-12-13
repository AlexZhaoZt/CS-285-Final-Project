import tensorboard
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

legends = ['DQN']

pacman = 'data/hw3_q1_MsPacman-v0_22-10-2021_03-35-40/events.out.tfevents.1634898940.bugting-desktop'

lunarlander_dqn = ['data/hw3_q2_dqn_1_LunarLander-v3_22-10-2021_02-29-15/events.out.tfevents.1634894955.bugting-desktop',
                    'data/hw3_q2_dqn_2_LunarLander-v3_22-10-2021_02-43-22/events.out.tfevents.1634895802.bugting-desktop',
                    'data/hw3_q2_dqn_3_LunarLander-v3_22-10-2021_02-55-08/events.out.tfevents.1634896508.bugting-desktop']
lunarlander_doubledqn = ['data/hw3_q2_doubledqn_1_LunarLander-v3_22-10-2021_02-37-49/events.out.tfevents.1634895469.bugting-desktop',
                        'data/hw3_q2_doubledqn_2_LunarLander-v3_22-10-2021_02-49-27/events.out.tfevents.1634896167.bugting-desktop',
                        'data/hw3_q2_doubledqn_3_LunarLander-v3_22-10-2021_03-05-31/events.out.tfevents.1634897131.bugting-desktop']
lunarlander_ddqn = ['data/ddqn_lunar_LunarLander-v3_13-12-2021_06-01-14/events.out.tfevents.1639404074.bugting-desktop',
                    'data/ddqn_lunar_LunarLander-v3_13-12-2021_06-37-10/events.out.tfevents.1639406230.bugting-desktop',
                    'data/ddqn_lunar_LunarLander-v3_13-12-2021_06-33-38/events.out.tfevents.1639406018.bugting-desktop']

hyperparam_legends = ['Original', '(0,1) to (500k,0.02)', '(0,1) to (500k,0.1)', '(0,1) to (50k,0.1)']

hyperparams = ['data/hw3_q2_dqn_1_LunarLander-v3_22-10-2021_02-29-15/events.out.tfevents.1634894955.bugting-desktop',
                'data/hw3_q3_hparam1_LunarLander-v3_22-10-2021_16-23-19/events.out.tfevents.1634944999.bugting-desktop',
                'data/hw3_q3_hparam2_LunarLander-v3_22-10-2021_16-23-58/events.out.tfevents.1634945038.bugting-desktop',
                'data/hw3_q3_hparam3_LunarLander-v3_22-10-2021_16-24-31/events.out.tfevents.1634945071.bugting-desktop']

cartpole_legends = ['Baseline', 'Method 1', 'Method 2']

cartpole = ['data/hw3_q4_ac_1_100_CartPole-v0_22-10-2021_17-26-51/events.out.tfevents.1634948811.bugting-desktop',
            'milestone_results/q4_100_10_doubleb_CartPole-v0_27-10-2021_21-26-22/events.out.tfevents.1635395182.Arwas-MacBook-Pro.local',
            'data/final_cartpole_n100_CartPole-v0_02-12-2021_16-18-44/events.out.tfevents.1638490724.bugting-desktop']

q5_legends = ['Baseline', 'Method 1', 'Method 2']

q5 = ['data/hw3_q5_1_100_InvertedPendulum-v2_22-10-2021_17-29-58/events.out.tfevents.1634948998.bugting-desktop',
    'milestone_results/q5_100_1_doubleb_InvertedPendulum-v2_27-10-2021_21-34-07/events.out.tfevents.1635395647.Arwas-MacBook-Pro.local',
    'data/q5_10_10_InvertedPendulum-v2_27-10-2021_22-06-23/events.out.tfevents.1635397583.bugting-desktop']

def plot_q1():
    ea = event_accumulator.EventAccumulator(pacman,
    size_guidance={ # see below regarding this argument
        event_accumulator.COMPRESSED_HISTOGRAMS: 500,
        event_accumulator.IMAGES: 4,
        event_accumulator.AUDIO: 4,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1,
    })

    events = ea.Reload()
    print(ea.Tags())
    avg_ret = ea.Scalars('Train_AverageReturn')
    best_ret = ea.Scalars('Train_BestReturn')
    df_avg_ret = pd.DataFrame(avg_ret)
    df_best_ret = pd.DataFrame(best_ret)
    x_axis = np.arange(0, 1e6, 10000)[:-1]
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ylabel('Return')
    plt.xlabel('Time steps')
    plt.plot(x_axis, df_avg_ret.value, label='average return')
    plt.plot(x_axis, df_best_ret.value, label='best return')

    plt.legend()
    plt.savefig('pacman.png')
    plt.clf()

def plot_lunarlander():
    lunarlander_doubledqn_array = []
    for exp in lunarlander_doubledqn:
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

        events = ea.Reload()
        eval_ret = ea.Scalars('Train_AverageReturn')
        df = pd.DataFrame(eval_ret)
        lunarlander_doubledqn_array.append(df.value)
        
    lunarlander_doubledqn_array = np.array(lunarlander_doubledqn_array)

    lunarlander_ddqn_array = []
    for exp in lunarlander_ddqn:
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

        events = ea.Reload()
        eval_ret = ea.Scalars('Train_AverageReturn')
        df = pd.DataFrame(eval_ret)
        lunarlander_ddqn_array.append(df.value)
        
    lunarlander_ddqn_array = np.array(lunarlander_ddqn_array)
    
    x_axis = np.arange(0, 5e5, 10000)[:-1]
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ylabel('Return')
    plt.xlabel('Time steps')
    plt.plot(x_axis, np.mean(lunarlander_doubledqn_array, 0), label='DQN')
    x_axis = np.arange(0, 5e5, 10000)
    plt.plot(x_axis, np.mean(lunarlander_ddqn_array, 0), label='DDQN')

    plt.legend()
    plt.savefig('q2.png')
    plt.clf()

def plot_q3():
    for exp, leg in zip(hyperparams, hyperparam_legends):
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })
        events = ea.Reload()
        # print(ea.Tags())
        eval_ret = ea.Scalars('Train_AverageReturn')
        df = pd.DataFrame(eval_ret)
        
        x_axis = np.arange(0, 5e5, 10000)[:-1]
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ylabel('Return')
        plt.xlabel('Time steps')
        plt.plot(x_axis, df.value, label=leg)

    plt.legend()
    plt.savefig('q3.png')
    plt.clf()


def plot_cartpole():
    for exp, leg in zip(cartpole, cartpole_legends):
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })
        events = ea.Reload()
        # print(ea.Tags())
        eval_ret = ea.Scalars('Train_AverageReturn')
        df = pd.DataFrame(eval_ret)
        
        x_axis = np.arange(0, 100, 10)
        print(df.value.shape)
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ylabel('Return')
        plt.xlabel('Iterations')
        plt.plot(x_axis, df.value, label=leg)

    plt.legend()
    plt.savefig('cartpole.png')
    plt.clf()

# def plot_cheetah():
#     # halfcheetah
#     exp = q5[0]
#     leg = q5_legends[0]
    
#     ea = event_accumulator.EventAccumulator(exp,
#     size_guidance={ # see below regarding this argument
#         event_accumulator.COMPRESSED_HISTOGRAMS: 500,
#         event_accumulator.IMAGES: 4,
#         event_accumulator.AUDIO: 4,
#         event_accumulator.SCALARS: 0,
#         event_accumulator.HISTOGRAMS: 1,
#     })
#     events = ea.Reload()
#     # print(ea.Tags())
#     eval_ret = ea.Scalars('Train_AverageReturn')
#     df = pd.DataFrame(eval_ret)
    
#     x_axis = np.arange(0, df.value.shape[0])
#     print(df.value.shape)
#     # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#     plt.ylabel('Return')
#     plt.xlabel('Iterations')
#     plt.plot(x_axis, df.value, label=leg)

#     plt.legend()
#     plt.savefig('q5a.png')
#     plt.clf()

#     # inverted
#     exp = q5[1]
#     leg = q5_legends[1]
    
#     ea = event_accumulator.EventAccumulator(exp,
#     size_guidance={ # see below regarding this argument
#         event_accumulator.COMPRESSED_HISTOGRAMS: 500,
#         event_accumulator.IMAGES: 4,
#         event_accumulator.AUDIO: 4,
#         event_accumulator.SCALARS: 0,
#         event_accumulator.HISTOGRAMS: 1,
#     })
#     events = ea.Reload()
#     # print(ea.Tags())
#     eval_ret = ea.Scalars('Train_AverageReturn')
#     df = pd.DataFrame(eval_ret)
    
#     x_axis = np.arange(0, df.value.shape[0]) * 10
#     print(df.value.shape)
#     # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#     plt.ylabel('Return')
#     plt.xlabel('Iterations')
#     plt.plot(x_axis, df.value, label=leg)

#     plt.legend()
#     plt.savefig('q5b.png')
#     plt.clf()

def plot_cheetah():
    for exp, leg in zip(q5, q5_legends):
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })
        events = ea.Reload()
        # print(ea.Tags())
        eval_ret = ea.Scalars('Train_AverageReturn')
        df = pd.DataFrame(eval_ret)
        
        x_axis = np.arange(0, 100, 10)
        print(df.value.shape)
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ylabel('Return')
        plt.xlabel('Iterations')
        plt.plot(x_axis, df.value, label=leg)

    plt.legend()
    plt.savefig('halfcheetah.png')
    plt.clf()

# plot_q1()
plot_lunarlander()
# plot_q3()
# plot_cartpole()
# plot_cheetah()
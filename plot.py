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

cartpole_doubledqn = ['data/dqn_cartpole_CartPole-v0_14-12-2021_20-30-32/events.out.tfevents.1639542632.bugting-desktop',
                        'data/dqn_cartpole_3_CartPole-v0_15-12-2021_21-06-12/events.out.tfevents.1639631172.bugting-desktop',
                        'data/dqn_cartpole_2_CartPole-v0_15-12-2021_21-06-07/events.out.tfevents.1639631167.bugting-desktop']
cartpole_ddqn = ['data/ddqn_cartpole_2_CartPole-v0_15-12-2021_18-42-25/events.out.tfevents.1639622545.bugting-desktop',
                    'data/ddqn_cartpole_3_CartPole-v0_15-12-2021_18-42-40/events.out.tfevents.1639622560.bugting-desktop',
                    'data/ddqn_cartpole_CartPole-v0_14-12-2021_20-20-40/events.out.tfevents.1639542040.bugting-desktop']

bonuscartpole_ddqn = ['data/ddqn_cartpole_1_bonus_CartPole-v0_15-12-2021_21-22-44/events.out.tfevents.1639632164.bugting-desktop',
                    'data/ddqn_cartpole_2_bonus_CartPole-v0_15-12-2021_21-22-56/events.out.tfevents.1639632176.bugting-desktop',
                    'data/ddqn_cartpole_3_bonus_CartPole-v0_15-12-2021_21-22-50/events.out.tfevents.1639632170.bugting-desktop']


mountaincar_doubledqn = ['data/dqn_mountaincar_1_MountainCar-v0_15-12-2021_18-53-23/events.out.tfevents.1639623203.bugting-desktop',
                        'data/dqn_mountaincar_2_MountainCar-v0_15-12-2021_23-05-48/events.out.tfevents.1639638348.bugting-desktop']
mountaincar_ddqn = ['data/ddqn_mountaincar_1_MountainCar-v0_15-12-2021_18-45-36/events.out.tfevents.1639622736.bugting-desktop',
                    'data/ddqn_mountaincar_3_MountainCar-v0_15-12-2021_23-12-33/events.out.tfevents.1639638753.bugting-desktop']


cartpole_a2c = ['data_old/q4_ac_10_10_CartPole-v0_22-10-2021_17-26-39/events.out.tfevents.1634948799.bugting-desktop',
            'data/a2c_cartpole2_CartPole-v0_15-12-2021_23-39-15/events.out.tfevents.1639640355.bugting-desktop',
            'data/a2c_cartpole3_CartPole-v0_15-12-2021_23-41-35/events.out.tfevents.1639640495.bugting-desktop']
cartpole_dac = ['data/dac_cartpole_CartPole-v0_15-12-2021_16-19-22/events.out.tfevents.1639613962.bugting-desktop']
bonuscartpole_dac = ['data/dac_cartpole_bonus_CartPole-v0_15-12-2021_22-54-36/events.out.tfevents.1639637676.bugting-desktop',
            'data/dac_cartpole_bonus2_CartPole-v0_15-12-2021_23-26-36/events.out.tfevents.1639639596.bugting-desktop',
            'data/dac_cartpole_bonus3_CartPole-v0_15-12-2021_23-27-18/events.out.tfevents.1639639638.bugting-desktop']

cheetah_dac = ['data/ac_halfcheetah_HalfCheetah-v2_15-12-2021_16-30-28/events.out.tfevents.1639614628.bugting-desktop']
cheetah_ac = ['data_old/q5_1_100_HalfCheetah-v2_22-10-2021_17-30-47/events.out.tfevents.1634949047.bugting-desktop']

inverted_dac = ['data/ac_inverted_InvertedPendulum-v2_15-12-2021_16-30-04/events.out.tfevents.1639614604.bugting-desktop']
inverted_ac = ['data_old/q5_1_100_InvertedPendulum-v2_22-10-2021_17-29-58/events.out.tfevents.1634948998.bugting-desktop']

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
    plt.plot(x_axis, np.mean(lunarlander_doubledqn_array, 0), label='DQN', color='darkorange')
    plt.fill_between(x_axis, np.mean(lunarlander_doubledqn_array, 0)-np.std(lunarlander_doubledqn_array, 0), np.mean(lunarlander_doubledqn_array, 0)+np.std(lunarlander_doubledqn_array, 0), alpha=0.2, color='darkorange')
    x_axis = np.arange(0, 5e5, 10000)
    plt.plot(x_axis, np.mean(lunarlander_ddqn_array, 0), label='DDQN', color='cornflowerblue')
    plt.fill_between(x_axis, np.mean(lunarlander_ddqn_array, 0)-np.std(lunarlander_ddqn_array, 0), np.mean(lunarlander_ddqn_array, 0)+np.std(lunarlander_ddqn_array, 0), alpha=0.2, color='cornflowerblue')

    plt.legend()
    plt.savefig('lunar.png')
    plt.clf()


def plot_cartpole():
    cartpole_doubledqn_array = []
    for exp in cartpole_doubledqn:
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

        events = ea.Reload()
        eval_ret = ea.Scalars('Train_BestReturn')
        df = pd.DataFrame(eval_ret)
        cartpole_doubledqn_array.append(df.value[:20])
        
    cartpole_doubledqn_array = np.array(cartpole_doubledqn_array)

    cartpole_ddqn_array = []
    for exp in cartpole_ddqn:
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

        events = ea.Reload()
        eval_ret = ea.Scalars('Train_BestReturn')
        df = pd.DataFrame(eval_ret)
        cartpole_ddqn_array.append(df.value[:20])
        
    cartpole_ddqn_array = np.array(cartpole_ddqn_array)
    
    x_axis = np.arange(0, 5e5, 10000)[:20]
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ylabel('Return')
    plt.xlabel('Time steps')
    plt.plot(x_axis, np.mean(cartpole_doubledqn_array, 0), label='DQN', color='darkorange')
    plt.fill_between(x_axis, np.mean(cartpole_doubledqn_array, 0)-np.std(cartpole_doubledqn_array, 0), np.mean(cartpole_doubledqn_array, 0)+np.std(cartpole_doubledqn_array, 0), alpha=0.2, color='darkorange')
    x_axis = np.arange(0, 5e5, 10000)[:20]
    plt.plot(x_axis, np.mean(cartpole_ddqn_array, 0), label='DDQN', color='cornflowerblue')
    plt.fill_between(x_axis, np.mean(cartpole_ddqn_array, 0)-np.std(cartpole_ddqn_array, 0)/2, np.mean(cartpole_ddqn_array, 0)+np.std(cartpole_ddqn_array, 0)/2, alpha=0.2, color='cornflowerblue')

    plt.legend()
    plt.savefig('cartpole_best.png')
    plt.clf()


def plot_bonuscartpole():
    cartpole_doubledqn_array = []
    for exp in cartpole_doubledqn:
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
        cartpole_doubledqn_array.append(df.value[:20])
        
    cartpole_doubledqn_array = np.array(cartpole_doubledqn_array)
    bonuscartpole_ddqn_array = []
    for exp in bonuscartpole_ddqn:
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
        bonuscartpole_ddqn_array.append(df.value[:20])
        print(df.value.shape)

    bonuscartpole_ddqn_array = np.array(bonuscartpole_ddqn_array)
    
    x_axis = np.arange(0, 5e5, 10000)[:20]
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ylabel('Return')
    plt.xlabel('Time steps')
    plt.plot(x_axis, np.mean(cartpole_doubledqn_array, 0), label='DQN', color='darkorange')
    plt.fill_between(x_axis, np.mean(cartpole_doubledqn_array, 0)-np.std(cartpole_doubledqn_array, 0), np.mean(cartpole_doubledqn_array, 0)+np.std(cartpole_doubledqn_array, 0), alpha=0.2, color='darkorange')
    x_axis = np.arange(0, 5e5, 10000)[:20]
    plt.plot(x_axis, np.mean(bonuscartpole_ddqn_array, 0), label='DDQN', color='cornflowerblue')
    plt.fill_between(x_axis, np.mean(bonuscartpole_ddqn_array, 0)-np.std(bonuscartpole_ddqn_array, 0)/2, np.mean(bonuscartpole_ddqn_array, 0)+np.std(bonuscartpole_ddqn_array, 0)/2, alpha=0.2, color='cornflowerblue')

    plt.legend()
    plt.savefig('bonus_cartpole.png')
    plt.clf()

def plot_bonuscartpole_a2c():
    cartpole_a2c_array = []
    for exp in cartpole_a2c:
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

        events = ea.Reload()
        eval_ret = ea.Scalars('Eval_AverageReturn')
        df = pd.DataFrame(eval_ret)
        cartpole_a2c_array.append(df.value[:10])
        
    cartpole_a2c_array = np.array(cartpole_a2c_array)
    bonuscartpole_dac_array = []
    for exp in bonuscartpole_dac:
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

        events = ea.Reload()
        eval_ret = ea.Scalars('Eval_AverageReturn')
        df = pd.DataFrame(eval_ret)
        bonuscartpole_dac_array.append(df.value[:10])
        print(df.value.shape)

    bonuscartpole_dac_array = np.array(bonuscartpole_dac_array)
    
    x_axis = np.arange(0, 5e5, 10000)[:10]
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ylabel('Return')
    plt.xlabel('Time steps')
    plt.plot(x_axis, np.mean(cartpole_a2c_array, 0), label='A2C', color='darkorange')
    plt.fill_between(x_axis, np.mean(cartpole_a2c_array, 0)-np.std(cartpole_a2c_array, 0), np.mean(cartpole_a2c_array, 0)+np.std(cartpole_a2c_array, 0), alpha=0.2, color='darkorange')
    plt.plot(x_axis, np.mean(bonuscartpole_dac_array, 0), label='DA2C', color='cornflowerblue')
    plt.fill_between(x_axis, np.mean(bonuscartpole_dac_array, 0)-np.std(bonuscartpole_dac_array, 0)/2, np.mean(bonuscartpole_dac_array, 0)+np.std(bonuscartpole_dac_array, 0)/2, alpha=0.2, color='cornflowerblue')

    plt.legend()
    plt.savefig('cartpole_a2c_bonus.png')
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

def plot_cheetah():
    cheetah_ac_array = []
    for exp in cheetah_ac:
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

        events = ea.Reload()
        eval_ret = ea.Scalars('Eval_AverageReturn')
        df = pd.DataFrame(eval_ret)
        cheetah_ac_array.append(df.value)
        
    cheetah_ac_array = np.array(cheetah_ac_array)

    cheetah_dac_array = []
    for exp in cheetah_dac:
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

        events = ea.Reload()
        eval_ret = ea.Scalars('Eval_AverageReturn')
        df = pd.DataFrame(eval_ret)
        cheetah_dac_array.append(df.value)
        
    cheetah_dac_array = np.array(cheetah_dac_array)

    x_axis = np.arange(0, 150)
    
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ylabel('Return')
    plt.xlabel('Time steps')
    plt.plot(x_axis, np.mean(cheetah_ac_array, 0), label='A2C', color='darkorange')
    plt.fill_between(x_axis, np.mean(cheetah_ac_array, 0)-np.std(cheetah_ac_array, 0), np.mean(cheetah_ac_array, 0)+np.std(cheetah_ac_array, 0), alpha=0.2, color='darkorange')
    plt.plot(x_axis, np.mean(cheetah_dac_array, 0), label='DA2C', color='cornflowerblue')
    plt.fill_between(x_axis, np.mean(cheetah_dac_array, 0)-np.std(cheetah_dac_array, 0)/2, np.mean(cheetah_dac_array, 0)+np.std(cheetah_dac_array, 0)/2, alpha=0.2, color='cornflowerblue')

    plt.legend()
    plt.savefig('cheetah.png')
    plt.clf()


def plot_mountaincar():
    mountaincar_doubledqn_array = []
    for exp in mountaincar_doubledqn:
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
        print(df.value.shape)
        mountaincar_doubledqn_array.append(df.value)
        
    mountaincar_doubledqn_array = np.array(mountaincar_doubledqn_array)

    mountaincar_ddqn_array = []
    for exp in mountaincar_ddqn:
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
        print(df.value.shape)
        mountaincar_ddqn_array.append(df.value)
        
    mountaincar_ddqn_array = np.array(mountaincar_ddqn_array)

    x_axis = np.arange(0, 5e5, 10000)
    
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ylabel('Return')
    plt.xlabel('Time steps')
    plt.plot(x_axis, np.mean(mountaincar_doubledqn_array, 0), label='DQN', color='darkorange')
    plt.fill_between(x_axis, np.mean(mountaincar_doubledqn_array, 0)-np.std(mountaincar_doubledqn_array, 0), np.mean(mountaincar_doubledqn_array, 0)+np.std(mountaincar_doubledqn_array, 0), alpha=0.2, color='darkorange')
    plt.plot(x_axis, np.mean(mountaincar_ddqn_array, 0), label='DDQN', color='cornflowerblue')
    plt.fill_between(x_axis, np.mean(mountaincar_ddqn_array, 0)-np.std(mountaincar_ddqn_array, 0)/2, np.mean(mountaincar_ddqn_array, 0)+np.std(mountaincar_ddqn_array, 0)/2, alpha=0.2, color='cornflowerblue')

    plt.legend()
    plt.savefig('mountaincar.png')
    plt.clf()

def plot_inverted():
    inverted_ac_array = []
    for exp in inverted_ac:
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

        events = ea.Reload()
        eval_ret = ea.Scalars('Eval_AverageReturn')
        df = pd.DataFrame(eval_ret)
        inverted_ac_array.append(df.value)
        
    inverted_ac_array = np.array(inverted_ac_array)

    inverted_dac_array = []
    for exp in inverted_dac:
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

        events = ea.Reload()
        eval_ret = ea.Scalars('Eval_AverageReturn')
        df = pd.DataFrame(eval_ret)
        inverted_dac_array.append(df.value)
        
    inverted_dac_array = np.array(inverted_dac_array)

    x_axis = np.arange(0, 100, 10)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ylabel('Return')
    plt.xlabel('Time steps')
    plt.plot(x_axis, np.mean(inverted_ac_array, 0), label='A2C', color='darkorange')
    plt.fill_between(x_axis, np.mean(inverted_ac_array, 0)-np.std(inverted_ac_array, 0), np.mean(inverted_ac_array, 0)+np.std(inverted_ac_array, 0), alpha=0.2, color='darkorange')
    plt.plot(x_axis, np.mean(inverted_dac_array, 0), label='DA2C', color='cornflowerblue')
    plt.fill_between(x_axis, np.mean(inverted_dac_array, 0)-np.std(inverted_dac_array, 0)/2, np.mean(inverted_dac_array, 0)+np.std(inverted_dac_array, 0)/2, alpha=0.2, color='cornflowerblue')

    plt.legend()
    plt.savefig('inverted.png')
    plt.clf()



def plot_cartpole_a2c():
    cartpole_a2c_array = []
    for exp in cartpole_a2c:
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

        events = ea.Reload()
        eval_ret = ea.Scalars('Eval_AverageReturn')
        df = pd.DataFrame(eval_ret)
        cartpole_a2c_array.append(df.value[:10])
        
    cartpole_a2c_array = np.array(cartpole_a2c_array)
    cartpole_dac_array = []
    for exp in cartpole_dac:
        ea = event_accumulator.EventAccumulator(exp,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

        events = ea.Reload()
        eval_ret = ea.Scalars('Eval_AverageReturn')
        df = pd.DataFrame(eval_ret)
        cartpole_dac_array.append(df.value[:10])
    print(cartpole_dac_array)
    cartpole_dac_array = np.array(cartpole_dac_array)
    
    x_axis = np.arange(0, 5e5, 10000)[:10]
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ylabel('Return')
    plt.xlabel('Time steps')
    plt.plot(x_axis, np.mean(cartpole_a2c_array, 0), label='A2C', color='darkorange')
    plt.fill_between(x_axis, np.mean(cartpole_a2c_array, 0)-np.std(cartpole_a2c_array, 0), np.mean(cartpole_a2c_array, 0)+np.std(cartpole_a2c_array, 0), alpha=0.2, color='darkorange')
    plt.plot(x_axis, np.mean(cartpole_dac_array, 0), label='DA2C', color='cornflowerblue')
    plt.fill_between(x_axis, np.mean(cartpole_dac_array, 0)-np.std(cartpole_dac_array, 0)/2, np.mean(cartpole_dac_array, 0)+np.std(cartpole_dac_array, 0)/2, alpha=0.2, color='cornflowerblue')

    plt.legend()
    plt.savefig('cartpole_a2c.png')
    plt.clf()


# plot_q1()
# plot_lunarlander()
# plot_q3()
# plot_cartpole()
plot_cartpole_a2c()
# plot_bonuscartpole()
plot_bonuscartpole_a2c()
# plot_cheetah()
# plot_inverted()
# plot_mountaincar()
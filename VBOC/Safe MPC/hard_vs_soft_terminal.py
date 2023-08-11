import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

label_size = 65
font_size = 60
legend_size = 50
mpl.rcdefaults()
mpl.rcParams['lines.linewidth'] = 10
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['patch.linewidth'] = 2
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.labelsize'] = label_size
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = font_size
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Bitstream Vera Serif', 'DejaVu Serif',
                                   'New Century Schoolbook',
                                   'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L',
                                   'Palatino',
                                   'Charter', 'serif']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['legend.fontsize'] = legend_size
mpl.rcParams['legend.loc'] = 'best'
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.figsize'] = 22, 14
mpl.rcParams['savefig.format'] = 'pdf'

# Load data
data_dir = "data_wo_noise/"
user_dir = os.path.expanduser('~')
pic_dir = user_dir + '/Pictures/Plots/'
data_no = np.load(data_dir + "results_no_constraint.npz")
data_hard = np.load(data_dir + "results_hardterm.npz")
data_soft = np.load(data_dir + "results_softterm.npz")

res_steps_no = data_no["res_steps"]
res_steps_hard = data_hard["res_steps_term"]
res_steps_soft = data_soft["res_steps_term"]
tests = np.arange(1, len(res_steps_no) + 1)

desc_indexes = np.argsort(res_steps_no)[::-1]

# Compare the three residual steps
plt.figure()
plt.plot(tests, res_steps_no[desc_indexes], label="naive MPC", color='red')
plt.plot(tests, res_steps_hard[desc_indexes], label="MPC + hard terminal", color='green')
plt.plot(tests, res_steps_soft[desc_indexes], label="MPC + soft terminal", color='blue')
plt.legend()
plt.xlabel(r"\# test")
plt.ylabel(r"MPC steps")
plt.savefig(pic_dir + "hard_vs_soft_terminal.pdf", bbox_inches='tight')

# Compare the receding cases
data_hardsoft = np.load(data_dir + "results_receiding_hardsoft.npz")
res_steps_hardsoft = data_hardsoft["res_steps_term"]
data_softsoft = np.load(data_dir + "results_receiding_softsoft.npz")
res_steps_softsoft = data_softsoft["res_steps_term"]

plt.figure()
plt.plot(tests, res_steps_no[desc_indexes], label="naive MPC", color='red')
plt.plot(tests, res_steps_hardsoft[desc_indexes], label="receding MPC (hard + soft)", color='green')
plt.plot(tests, res_steps_softsoft[desc_indexes], label="receding MPC (soft + soft)", color='blue')
plt.legend()
plt.xlabel(r"\# test")
plt.ylabel(r"MPC steps")
plt.savefig(pic_dir + "receding.pdf", bbox_inches='tight')

plt.show()
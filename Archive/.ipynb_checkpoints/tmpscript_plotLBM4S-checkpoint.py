from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML

import LBM4S_v0

outputs = LBM4S_v0.simulate(szing, None, 0.236, 30, 'V', inc=torch.as_tensor((0, -torch.pi/4+0.001))[None,None,])

fig, ax = plt.subplots(figsize=(3,3))

readf =  outputs[0].sum(dim=(-1,-2))#outputs[0][...,0,1] + outputs[0][...,2,1]
line = np.ones((50,50)); line[0,0] = readf.min(); line[1,1] = readf.max(); line = ax.imshow(line, animated=True)

t0 = 300
def animf(t): line.set_data((readf[0,0,t+t0,:,:]).cpu().numpy()); return line,
ani = FuncAnimation(fig, animf, frames=outputs[0].shape[2]-t0, interval=25, blit=False)
plt.close(fig)
ani.save("tmp.gif", writer=PillowWriter(fps=30))


HTML(ani.to_html5_video())

readf = outputs[1]
line = np.ones((50,50)); line[0,0] = readf.min(); line[1,1] = readf.max(); line = ax.imshow(line, animated=True)
ani = FuncAnimation(fig, animf, frames=outputs[0].shape[2]-t0, interval=25, blit=False)
plt.close(fig)
ani.save("tmp.gif", writer=PillowWriter(fps=30))
#HTML(ani.to_html5_video())
"""Bounded growth schedule and actual continuous-solve history initialization."""
import numpy as np
from research.fullstep_radau import fullstep_curve


def schedule(initial_step,duration,order,ratio):
    if (order not in (3,4) or not np.isfinite([initial_step,duration,ratio]).all()
            or initial_step<=0 or duration<=(order-1)*initial_step or not 1<=ratio<=2):
        raise ValueError('Orders 3/4, positive window beyond startup, growth in [1,2] required')
    times=[0.,(order-1)*initial_step];h=initial_step
    while times[-1]<duration:
        h=min(h*ratio,duration-times[-1]);nxt=times[-1]+h
        if nxt<=times[-1]:raise ValueError('Schedule below time resolution')
        times.append(min(nxt,duration))
    return np.array(times)


def bootstrap(case,initial,order,h,tolerance,screen,kernel):
    times=np.arange(order)*h
    trial=fullstep_curve(case,initial,float(times[-1]),tolerance,screen)
    states=trial['pieces'][0](times/times[-1])
    # Keep the solver endpoint exact rather than its rounded polynomial value.
    states[:,-1]=trial['fine']
    previous=np.column_stack([kernel.slow(z) for z in states[:,:-1].T])
    return trial,times,states,previous

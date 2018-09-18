#!/usr/bin/env python3

from __future__ import division

# get cpu layout

import os
import re
import json
import sys
from subprocess import call
from time import sleep

import logging
from logging import debug, info, warn, error, critical
logging.basicConfig(stream=sys.stderr,level=logging.DEBUG)

cputasks = [
    r'^primegrid_llr',
    r'setiathome'
    ]

gputasks = [
    r'cudaPPSsieve',
    ]

def slurp(*args):
    path = os.path.join(*args)
    return open(path, 'r').read().rstrip()

def dump(thing):
    debug(json.dumps(thing, indent=2, sort_keys=True))
    
def is_cputask(cmdline):
    for e in cputasks:
        if re.search(e, cmdline[0]) is not None:
            return True
        
def is_gputask(cmdline):
    for e in gputasks:
        if re.search(e, cmdline[0]) is not None:
            return True

def argmin(d):
    return min(d, key=d.get)

cores = {}
threads = {}
thread_siblings = {}

def get_topology():
    base = "/sys/devices/system/cpu"
    for i in os.listdir(base):
        m = re.match(r'cpu(\d+)', i)
        if m is not None:
            thread = int(m.group(1))
            core = int(slurp(base,i,"topology/core_id"))
            thread_siblings[thread] = slurp(base,i,"topology/thread_siblings_list")
            thread_d = {}
            threads[thread] = thread_d
            cores.setdefault(core, {})[thread] = thread_d
            thread_d['procs'] = {}
            thread_d['core'] = core

gpu_irq = None
irqs = "/proc/irq"

def get_gpu_irq():
    for i in os.listdir(irqs):
        if os.path.isdir(os.path.join(irqs, i)):
            if os.path.isdir(os.path.join(irqs, i, 'nvidia')):
                return i

def get_gpu_hw_thread(gpu_irq):
    affinity = slurp(irqs, gpu_irq,'smp_affinity_list')
    return int(re.split('[,-]', affinity)[0])

def set_gpu_core(gpu_irq, core):
    with open(os.path.join(irqs, gpu_irq, 'smp_affinity_list'), 'w') as affinity:
        affinity.write(','.join([str(i) for i in cores[core].keys()]))
        affinity.flush()

#gpu_core = first_cpu[min(first_cpu.keys())]
get_topology()
#gpu_irq = get_gpu_irq()
#gpu_thread = get_gpu_hw_thread(gpu_irq)
#gpu_core = threads[gpu_thread]['core']
#set_gpu_core(gpu_irq, gpu_core)
#other_cores = [core for core in cores.keys() if core != gpu_core]
#other_cores_threads = [thread for thread in threads.keys() if thread not in cores[gpu_core].keys()]
#other_all_threads = [thread for thread in threads.keys() if thread != gpu_thread]

def get_free_core_thread(core):
    return min(cores[core].keys(), key=lambda x: len(cores[core][x]['procs']))

def get_core_load(core):
    return sum([len(thread['procs']) for thread in cores[core].values()])

def get_free_cores_thread(cores):
    mincore = min(cores, key=get_core_load)
    return get_free_core_thread(mincore)

def smartish_thread_load(threadid):
    return len(threads[threadid]['procs']) + get_core_load(threads[threadid]['core'])/1000

def get_free_hw_thread(some_threads):
    return min(some_threads, key=smartish_thread_load)

def assign_process(pid, threadid, count=True):
    if count:
        threads[threadid]['procs'][pid] = True
    #call(['taskset', '-a', '-p', '-c', str(threadid), str(pid)])
    call(['taskset', '-a', '-p', '-c', thread_siblings[threadid], str(pid)])

def assign_thread(pid, threadid, count=True):
    if count:
        threads[threadid]['procs'][pid] = True
    #call(['taskset', '-p', '-c', str(threadid), str(pid)])
    call(['taskset', '-p', '-c', thread_siblings[threadid], str(pid)])

def get_process_threads(pid):
    return list(os.listdir(os.path.join('/proc', str(pid), 'task')))
    
def distribute_process_threads(pid, hw_threads, count=True):
    process_threads = get_process_threads(pid)
    if count:
        for pid in process_threads:
            assign_thread(pid, get_free_hw_thread(hw_threads), count)
    else:
        for i in range(0, len(process_threads)):
            pid = process_threads[i]
            hw_thread = hw_threads[i % len(hw_threads)]
            assign_thread(pid, hw_thread, count)


def go():
    gpuprocs = []
    cpuprocs = []
    otherprocs = []

    proc = "/proc"
    for i in os.listdir(proc):
        m = re.match(r'\d+$', i)
        if m is not None:
            pid = int(i)
            try:
                cmd = slurp(proc, i, 'cmdline').split('\0')
            except IOError as e:
                continue
            if is_gputask(cmd):
                debug("GPU proc: %d" % (pid))
                gpuprocs.append(pid)
            elif is_cputask(cmd):
                debug("CPU proc: %d" % (pid))
                cpuprocs.append(pid)
            else:
                otherprocs.append(pid)
    
    debug("-------")
    for t in threads.values():
        t['procs'] = {}
    debug(repr(gpuprocs))
    for p in sorted(gpuprocs):
        #assign_process(p, get_free_core_thread(gpu_core))
        #distribute_process_threads(p, [gpu_thread])
        call(['chrt', '-a', '-f', '-p', '1', str(p)])
        #distribute_process_threads(p, [gpu_thread])
        #call(['taskset', '-a', '-p', '-c', '1,5', str(pid)])

    for p in sorted(cpuprocs):
        #assign_process(p, get_free_cores_thread(other_cores))
        #call(['chrt', '-a', '-i', '-p', '0', str(p)])
        #call(['taskset', '-a', '-p', '-c', ','.join([str(t) for t in other_threads]), str(pid)])
        #call(['taskset', '-a', '-p', '-c', '4,5,6,7', str(pid)])
        #distribute_process_threads(p, other_all_threads)
        distribute_process_threads(p, threads.keys())
        pass


go()

#call(['taskset', '-p', '-c', str(cores[gpu_core].keys()[0]), str(os.getpid())])
#latency = open('/dev/cpu_dma_latency', 'wb', buffering=0)
#latency.write(b'\0\0\0\0')
#latency.flush()

while True:
    sleep(10)
    go()


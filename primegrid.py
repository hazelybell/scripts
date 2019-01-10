#!/usr/bin/env python3

import os
import re
import json
import sys
from subprocess import call
from time import sleep
import argparse

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

class NSet:
    __slots__ = (s,d)
    
    def __init__(self, i=None):
        if isinstance(i, set):
            self.s = set(i)
            self.d = {
                x.n:x for x in s
                }
        elif isinstance(i, dict):
            self.s = set(i.values())
            self.d = dict(i)
        elif i is None:
            s = set()
            d = dict()
        else:
            raise ValueError()
        
    def add(self, x):
        assert x not in self.s
        assert x not in self.d.values()
        self.s.add(x)
        self.d[x.n] = x
    
    def __getitem__(self, key):
        return self.d[key]
    
    def __len__(self, key):
        return len(self.s)
    
    def __setitem__(self, key, value):
        assert key == value.n
        self.add(value)
    
    def __contains__(self, item):
        return item in self.s or item in self.d
    
    def __delitem__(self, item):
        if item in self.d:
            self.s.remove(d[item])
            del self.d[item]
        else:
            raise KeyError()
    
    def remove(self, item):
        if item in self.s:
            self.s.remove(item)
            del self.d[item.n]
        else:
            raise KeyError()
    
    def without(self, item):
        new = self.__class__(self)
        if item in self.s:
            new.remove(item)
        elif item in self.d:
            del new[item]
        else:
            raise KeyError()
        return new
    
    def union(self, *others):
        return self.__class__(self.s.union(*[o.s for o in others]))
    
    def difference(self, *others):
        return self.__class__(self.s.difference(*[o.s for o in others]))
    
    def minimum(self):
        return min(self.s, key=lambda x: x.weight())

class ProcessorSet(NSet):
    def without_core(self, core):
        new = ProcessorSet(self)
        new.d = {
            n:p for n,p in new.d.items()
                if p.core != core
            }
        new.s = {
            p for p in p
                if p.core != core
            }
        return new
    
    def cores(self):
        return CoreSet({
            p.core for p in self.s
            })

    def weight(self):
        return sum([p.weight() for p in self.s])
    
    def as_string(self):
        return ','.join(map(str, new.d.keys()))

class CoreSet(NSet):
    def processors(self):
        ps = ProcessorSet()
        return ps.union(*[
                c.processors for c in self.s
            ])
    
    def minimum_core_processor(self):
        """ Return the freest processor on the freest core """
        return self.minimum().processors.minimum()

class Numbered:
    def __init__(self, n):
        self.n = n
    
    def __hash__(self):
        return hash((self.__class__, self.n))

class Processor(Numbered):
    def __init__(self, n, core, topology):
        super().__init__(n)
        self.topology = topology
        self.core = core
        self.threads = dict()
    
    def siblings(self):
        assert self in self.core.processors
        return {p for p in self.core.processors
                    if p is not self }
    
    def weight(self):
        return sum(self.threads.values())
    
    def assign(tid, weight, strict, process=False)
        self.threads[tid] = weight
        taskset = ['taskset', '-p', '-c'] 
        if process:
            taskset.insert(1, '-a')
        if strict:
            call(taskset + [str(self.n), str(tid)])
        else: # allow the thread to float between hyperthreads
            s = self.core.processors.as_string()
            call(taskset + [s, str(tid)])
 
    def assign_process(pid, weight, strict):
            self.assign(pid, weight, strict, True)
    
class Core:
    def __init__(self, n, topology):
        super().__init__(n)
        self.processors = ProcessorSet()
        self.topology = topology
        assert self in self.topology.cores
    
    def add_processor(self, processor):
        self.processors.add(processor)
    
    def weight(self):
        return self.processors.weight()

    def get_freeest_processor(self):
        return self.processors.minimum()

class Topology:
    """400-level maths"""

    self.base = "/sys/devices/system/cpu"

    def __init__(self):
        self.read_topology()
        
    def self.read_each(self, f):
        for i in os.listdir(self.base):
            m = re.match(r'cpu(\d+)', i)
            if m is not None:
                f(i)

    def read_topology(self):
        self.cores = CoreSet()
        self.processors = ProcessorSet()
        
        def processor(i):
            processor_n = int(i)
            core_n = int(slurp(base,i,"topology/core_id"))
            self.add_processor(processor_n, core_n)
        self.read_each(processor)
        
        def siblings(i):
            processor_siblings = slurp(base,i,"topology/processor_siblings_list")
        self.read_each(siblings)
    
    def add_core(self, n):
        assert not n in self.cores_by_number
        core = Core(n, self)
        self.cores.add(core)
    
    def add_processor(self, n, core_n):
        assert not n in self.processors_by_number
        if not core_n in self.cores:
            self.add_core(core_n)
        core = self.cores[core_n]
        processor = Processor(n, core, self)
        self.processors.add(processor)
        core.add_processor(processor)
    
    def get_freeest_processor(self):
        return self.cores.get_minimum_core_processor()
    
class GPU:
    irqs = "/proc/irq"
    affinity_file = "smp_affinity_list"

    def __init__(self, topology):
        self.read_irq()
        self.read_processors()
        self.topology = topology

    def read_irq(self):
        self.gpu_irq = None
        for i in os.listdir(irqs):
            if os.path.isdir(os.path.join(irqs, i)):
                if os.path.isdir(os.path.join(irqs, i, 'nvidia')):
                    self.gpu_irq = i

    def read_processors(self):
        affinity = slurp(self.irqs, self.gpu_irq, self.affinity_file)
        self.processors = ProcessorSet({
            self.topology.processors[int(n)] 
            for n in re.split('[,-]', affinity)
            })
    
    def cores(self):
        return self.processor().core

    def set_core(self, gpu_irq, core):
        with open(os.path.join(self.irqs, gpu_irq, self.affinity_file), 'w') as affinity:
            affinity.write(','.join([str(i) for i in core.processors.keys()]))
            affinity.flush()
    
    def non_gpu_cores(self):
        return self.topology.cores.difference(self.processors.cores())
    
    def non_gpu_processors(self):
        return self.non_gpu_cores().processors()
    
class Schedule:
        

def get_process_threads(pid):
    return list(os.listdir(os.path.join('/proc', str(pid), 'task')))
    
def distribute_process_threads(pid, hw_threads, stagger=0, count=True):
    process_threads = get_process_threads(pid)
    if count:
        for pid in process_threads:
            assign_thread(pid, get_free_hw_thread(hw_threads, stagger), count)
    else:
        for i in range(0, len(process_threads)):
            pid = process_threads[i]
            hw_thread = hw_threads[(i+stagger) % len(hw_threads)]
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

    for i in range(0, len(cpuprocs)):
        p = sorted(cpuprocs)[i]
        #assign_process(p, get_free_cores_thread(other_cores))
        #call(['chrt', '-a', '-i', '-p', '0', str(p)])
        #call(['taskset', '-a', '-p', '-c', ','.join([str(t) for t in other_threads]), str(pid)])
        #call(['taskset', '-a', '-p', '-c', '4,5,6,7', str(pid)])
        #distribute_process_threads(p, other_all_threads)
        distribute_process_threads(p, threads.keys(), i)
        pass


go()

#call(['taskset', '-p', '-c', str(cores[gpu_core].keys()[0]), str(os.getpid())])
#latency = open('/dev/cpu_dma_latency', 'wb', buffering=0)
#latency.write(b'\0\0\0\0')
#latency.flush()

def mainloop(interval):
    while True:
        sleep(interval)
        go()
    
def main():
    description = "Manage CPU affinities for primegrid."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--accross-cores", 
                        help="distribute process threads across cores"
                        )
    parser.add_argument("--interval", 
                        type=int,
                        default=10,
                        help="set polling interval"
                        )
    args = parser.parse_args()
    mainloop(args.interval)

main()

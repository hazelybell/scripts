#!/usr/bin/env python3

import os
import re
import json
import sys
import subprocess
import time
import argparse
import math

import logging
logger = logging.getLogger(__name__)
DEBUG = logger.debug
INFO = logger.info
WARNING = logger.warning
ERROR = logger.error
CRITICAL = logger.critical
logging.basicConfig(stream=sys.stderr,level=logging.DEBUG)

cputasks = [
    re.compile(r'^primegrid_llr'),
    re.compile(r'setiathome')
    ]

gputasks = [
    re.compile(r'cudaPPSsieve'),
    re.compile(r'primegrid_genefer'),
    ]

PROC = "/proc"

def call(*args, **kwargs):
    DEBUG(" ".join([a for arg in args for a in arg]))
    subprocess.call(*args, **kwargs)

def slurp(*args):
    path = os.path.join(*args)
    return open(path, 'r').read().rstrip()

def dump(thing):
    DEBUG(json.dumps(thing, indent=2, sort_keys=True))
    
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
    __slots__ = ('s','d')

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
            self.s = set()
            self.d = dict()
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
            #self.s.remove(d[item])
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
    
    def minimum(self, stagger=[0]):
        precision = 0.0001
        ub = 1+precision
        lb = 1-precision
        minweight = math.inf
        minprocs = self.s
        for i in self.s:
            w = i.weight()
            if w < (minweight * lb):
                minprocs = [i]
                minweight = w
            elif w > (minweight * lb) and (w < minweight * ub):
                minprocs.append(i)
        minprocs = sorted(minprocs, key=lambda x: x.n)
        i = stagger[0] % len(minprocs)
        stagger[0] = stagger[0] // len(minprocs)
        return minprocs[i]
    
    def __iter__(self):
        return self.s.__iter__()
    
    def __len__(self):
        return self.s.__len__()

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
    
    def minimum_core_processor(self, stagger=[0]):
        """ Return the freest processor on the freest core """
        return self.minimum(stagger).processors.minimum(stagger)

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
        r = {p for p in self.core.processors
                    if p is not self }
        return r
    
    def weight(self):
        return sum(self.threads.values())
    
    def assign(self, tid, weight):
        self.threads[tid] = weight
        #DEBUG("Processor " + str(self.n) + " weight " + str(self.weight()))
    
    def unassign(self, tid):
        del self.threads[tid]

class Core:
    def __init__(self, n, topology):
        self.n = n
        super().__init__()
        self.processors = ProcessorSet()
        self.topology = topology
    
    def add_processor(self, processor):
        self.processors.add(processor)
    
    def weight(self):
        return self.processors.weight()

    def get_freeest_processor(self):
        return self.processors.minimum()

class Topology:
    """400-level maths"""

    base = "/sys/devices/system/cpu"

    def __init__(self):
        self.read_topology()
        
    def read_each(self, f):
        for i in os.listdir(self.base):
            m = re.match(r'cpu(\d+)', i)
            if m is not None:
                f(m)

    def read_topology(self):
        self.cores = CoreSet()
        self.processors = ProcessorSet()
        self.processors_by_number = dict()
        self.cores_by_number = dict()
        
        def processor(i):
            processor_n = int(i[1])
            core_n = int(slurp(self.base,i[0],"topology/core_id"))
            self.add_processor(processor_n, core_n)
        self.read_each(processor)
        
        def siblings(i):
            processor_siblings = slurp(self.base,i[0],"topology/thread_siblings_list")
            #TODO: make sure this checks out
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
        self.processors_by_number[n] = processor
    
    def get_freeest_processor(self):
        return self.cores.minimum_core_processor()
    
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

class Thread:
    def __init__(self, job, tid, weight):
        self.job = job
        self.tid = tid
        self.weight = weight
        self.processors = None
    
    def assign(self, processor, ahs):
        if ahs:
            self.processors = processor.core.processors
        else:
            self.processors = {processor}
        for p in self.processors:
            p.assign(self.tid, self.weight/len(self.processors))
        self.affine()
    
    def affine(self):
        procs = ','.join([str(p.n) for p in self.processors])
        taskset = ['taskset', '-p', '-c'] 
        call(taskset + [procs, str(self.tid)])
    
    def unassign(self):
        for p in self.processors:
             p.unassign(self.tid)

class Job:
    kind = None
    
    def __init__(self, pid, schedule):
        self.pid = pid
        self.schedule = schedule
        self.state = 'new'
        self.threads = self.get_threads()
    
    def get_threads(self):
        tids = list(os.listdir(os.path.join(PROC, str(self.pid), 'task')))
        threads = []
        for i in range(0, len(tids)):
            if i == 0:
                weight = 1.0
            else:
                weight = 0.9
            threads.append(Thread(self, tids[i], weight))
        return threads

class GPUJob(Job):
    kind = 'gpu'
    
    def assign(self, processor, ahs):
        if ahs:
            self.processors = processor.siblings()
        else:
            self.processors = {processor}
        for p in self.processors:
            p.assign(self.pid, self.weight/len(self.processors))
        self.affine()
    
    def affine(self):
        procs = ','.join([str(p.n) for p in self.processors])
        taskset = ['taskset', '-a', '-p', '-c'] 
        call(taskset + [procs, str(self.pid)])
        if self.schedule.options.realtime_gpu:
            call(['chrt', '-a', '-f', '-p', '1', str(self.pid)])
    
    def assign_whole_process(self, cores, stagger):
        p = cores.minimum_core_processor(stagger)
        ahs = self.schedule.options.allow_hyperthread_swapping
        self.assign(p, ahs)
    
    distribute = assign_whole_process
    
    def undistribute(self):
        for p in self.processors:
            p.unassign(self.pid)

class CPUJob(Job):
    kind = 'cpu'
    
    def spread_process_threads(self, cores, stagger):
        ahs = self.schedule.options.allow_hyperthread_swapping
        for t in self.threads:
            p = cores.minimum_core_processor(stagger)
            t.assign(p, ahs)
    
    def distribute_process_threads(self, cores, stagger):
        layout = self.schedule.options.layout
        if layout == 'spread':
            self.spread_process_threads(cores, stagger)
        else:
            raise NotImplemented("Layout not implemented")
    
    distribute = distribute_process_threads
    
    def undistribute(self):
        for t in self.threads:
            t.unassign()

def matches_one_of(what, res):
    for r in res:
        if r.search(what) is not None:
            return True
    return False

class Schedule:
    digits = re.compile(r'\d+$')
    
    def __init__(self, options):
        self.jobs = set()
        self.alive = set()
        self.dead = set()
        self.new = set()
        self.options = options
    
    @property
    def cpu_jobs(self):
        return { j for j in self.jobs if j.kind=="cpu" }
    
    @property
    def gpu_jobs(self):
        return { j for j in self.jobs if j.kind=="gpu" }
    
    @property
    def jobs_by_pid(self):
        return {
            j.pid: j for j in self.jobs
            }
    
    def get_new_jobs(self):
        known = self.jobs_by_pid
        alive = dict()
        new = set()
        def maybe_new_job(pid, kind):
            if pid in known:
                alive[pid] = known[pid]
            else:
                DEBUG("NEW %s PROC: %d" % (kind, pid))
                new.add(kind(pid, self))
        
        for i in os.listdir(PROC):
            m = self.digits.match(i)
            if m is not None:
                pid = int(i)
                try:
                    cmd = (slurp(PROC, i, 'cmdline').split('\0'))[0]
                except IOError as e:
                    continue
                if matches_one_of(cmd, gputasks):
                    maybe_new_job(pid, GPUJob)
                elif matches_one_of(cmd, cputasks):
                    maybe_new_job(pid, CPUJob)
                else:
                    pass
        self.alive = set(alive.values())
        self.dead = self.jobs.difference(self.alive)
        self.jobs = self.alive.union(new)
        self.new = new
    
    def distribute(self, cores):
        stagger = 0
        
        for j in self.dead:
            j.undistribute()
        
        self.dead = set()
        
        for j in self.new:
            j.distribute(cores, [stagger])
            stagger = stagger + 1
        
        self.alive = self.jobs
        self.new = set()
        
class Gridder:
    def __init__(self, options):
        self.options = options
        self.topology = Topology()
        self.schedule = Schedule(options)
    
    def watch(self):
        while True:
            self.tick()
            time.sleep(self.options.interval)
            DEBUG("-------")
    
    def tick(self):
        self.schedule.get_new_jobs()
        self.schedule.distribute(self.topology.cores)
        
    def run(self):
        if self.options.zero_latency:
            latency = open('/dev/cpu_dma_latency', 'wb', buffering=0)
            latency.write(b'\0\0\0\0')
            latency.flush()
        self.watch()

def main():
    description = "Manage CPU affinities for primegrid."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--layout",
                        help="how to distribute process threads among cores. "
                        "Can be one of spread or clump. It's not recommended to use spread.",
                        default='spread',
                        )
    parser.add_argument("--interval", 
                        type=int,
                        default=10,
                        help="set polling interval."
                        )
    parser.add_argument("--zero-latency",
                        help="ask the kernel to provide zero latency. "
                        "Not recommended: this keeps the processors from idling which interferes with hyperthreading",
                        action='store_true',
                        )
    parser.add_argument("--realtime-gpu",
                        help="give GPU processes realtime priority on the CPU",
                        action='store_true',
                        )
    parser.add_argument("--disallow-hyperthread-swapping",
                        help="don't allow threads to jump between processors on the same core",
                        action='store_true',
                        )
    args = parser.parse_args()
    args.allow_hyperthread_swapping = not args.disallow_hyperthread_swapping
    gridder = Gridder(args)
    gridder.run()

main()

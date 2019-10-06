#!/usr/bin/env python3

import os
import re
import json
import sys
import subprocess
import time
import argparse
import math
import glob
import asyncio
import tempfile
import shutil
import signal
import traceback
import json

import numpy as np
import scipy.stats

import logging
logger = logging.getLogger(__name__)
DEBUG = logger.debug
INFO = logger.info
WARNING = logger.warning
ERROR = logger.error
CRITICAL = logger.critical

cputasks = [
    re.compile(r'^primegrid_llr'),
    re.compile(r'setiathome')
    ]

gputasks = [
    re.compile(r'cudaPPSsieve'),
    re.compile(r'primegrid_genefer'),
    ]

PROC = "/proc"

LLR_RE = re.compile('[\n\r\b]+')

original_sigint_handler = signal.getsignal(signal.SIGINT)

def call(*args, **kwargs):
    #DEBUG(" ".join([a for arg in args for a in arg]))
    subprocess.call(*args, **kwargs)

def acse(*args, **kwargs):
    #DEBUG(" ".join(args))
    return asyncio.create_subprocess_exec(*args, **kwargs)

def slurp(*args):
    path = os.path.join(*args)
    return open(path, 'r').read().rstrip()

def slurp_output(*args, **kwargs):
    r = subprocess.run(args, capture_output=True, **kwargs)
    return r.stdout.decode('ascii')

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
                x.n:x for x in i
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
        self.directory = self.topology.base + "/cpu" + str(self.n)
        self.max_freq_fn = self.directory + '/cpufreq/cpuinfo_max_freq'
        self.min_freq_fn = self.directory + '/cpufreq/cpuinfo_min_freq'
        self.lim_freq_fn = self.directory + '/cpufreq/scaling_max_freq'
        self.cur_freq_fn = self.directory + '/cpufreq/scaling_cur_freq'
        self.max_freq = self.get_max_freq()
        self.min_freq = self.get_min_freq()
        self.ctc = None
        self.ptc = None
        self.ctc_fn = (self.directory
            + '/thermal_throttle/core_throttle_count')
        self.ptc_fn = (self.directory
            + '/thermal_throttle/package_throttle_count')
        if os.path.isfile(self.ctc_fn):
            self.ctc = int(slurp(self.ctc_fn))
        if os.path.isfile(self.ptc_fn):
            self.ptc = int(slurp(self.ptc_fn))
    
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
    
    def get_max_freq(self):
        return int(slurp(self.max_freq_fn))

    def get_min_freq(self):
        return int(slurp(self.min_freq_fn))
    
    def get_lim_freq(self):
        return int(slurp(self.lim_freq_fn))
    
    def set_lim_freq(self, khz):
        with open(self.lim_freq_fn, 'w') as limit:
            limit.write(str(khz))
            limit.flush()
        if (self.get_lim_freq() != khz):
            ERROR("Couldn't set frequency limit to " + str(khz))

    def get_cur_freq(self):
        return int(slurp(self.cur_freq_fn))
    
    def is_idle(self):
        lines = slurp('/proc/stat').splitlines()
        for line in lines:
            if line.startswith('cpu' + str(self.n)):
                fields = line.split()
                fields = [int(f) for f in fields[1:]]
                total = sum(fields)
                idle = fields[3]
                if idle/total > 0.75:
                    return True
                else:
                    return False
        ERROR("Couldn't determin if processor is idle...")
    
    def detect_throttle(self):
        if self.ctc is not None:
            new_ctc = int(slurp(self.ctc_fn))
            if new_ctc > self.ctc:
                self.ctc = new_ctc
                return True
        if self.ptc is not None:
            new_ptc = int(slurp(self.ptc_fn))
            if new_ptc > self.ptc:
                self.ptc = new_ptc
                return True
        if self.ctc is None and self.ptc is None:
            if not self.is_idle():
                cur_freq = self.core.get_cur_freq()
                lim_freq = self.core.get_lim_freq()
                if cur_freq < lim_freq - 200000:
                    INFO("Core {} frequency is {:.2f}. It's probably throttling."
                            .format(self.core.n, cur_freq/1000000))
                    return True
            return False

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
    
    def get_cur_freq(self):
        fs = [p.get_cur_freq() for p in self.processors]
        return sum(fs)//len(fs)

    def get_min_freq(self):
        fs = [p.get_min_freq() for p in self.processors]
        return max(fs)

    def get_max_freq(self):
        fs = [p.get_max_freq() for p in self.processors]
        return min(fs)

    def get_lim_freq(self):
        fs = [p.get_lim_freq() for p in self.processors]
        return min(fs)
    
    def set_lim_freq(self, khz):
        for p in self.processors:
            p.set_lim_freq(khz)
    
    def detect_throttle(self):
        for p in self.processors:
            if p.detect_throttle():
                return True
        return False

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
    
    def assign_free(self, processors):
        self.processors = processors
        for p in self.processors:
            p.assign(self.tid, self.weight/len(self.processors))
        self.affine()

    def assign(self, processor, ahs):
        if ahs:
            self.processors = processor.core.processors
        else:
            self.processors = {processor}
        self.assign_free(self.processors)
    
    def affine(self):
        procs = ','.join([str(p.n) for p in self.processors])
        taskset = ['taskset', '-p', '-c'] 
        call(taskset + [procs, str(self.tid)], stdout=subprocess.DEVNULL)
        #call(taskset + [procs, str(self.tid)])
    
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
        #DEBUG(repr(tids))
        threads = []
        for i in range(0, len(tids)):
            if i == 0:
                weight = 1.0
            else:
                weight = 0.9
            threads.append(Thread(self, tids[i], weight))
        #DEBUG("Threads: " + str(len(threads)))
        return threads

class GPUJob(Job):
    kind = 'gpu'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = 0.2
    
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
        DEBUG("GPU job " + str(self.pid) + " finished")
        for p in self.processors:
            p.unassign(self.pid)

class CPUJob(Job):
    kind = 'cpu'
    
    def spread_process_threads(self, cores, stagger):
        ahs = self.schedule.options.allow_hyperthread_swapping
        for t in self.threads:
            p = cores.minimum_core_processor(stagger)
            t.assign(p, ahs)
    
    def clump_process_threads(self, cores, stagger):
        ahs = self.schedule.options.allow_hyperthread_swapping
        i = 0
        while i < len(self.threads):
            p = cores.minimum_core_processor(stagger)
            sibs = list(p.core.processors)
            j = 0
            while j < len(sibs) and i < len(self.threads):
                t = self.threads[i]
                t.assign(sibs[j], ahs)
                j = j + 1
                i = i + 1
    
    def free_process_threads(self, cores, stagger):
        for t in self.threads:
            t.assign_free(cores.processors())
    
    def distribute_process_threads(self, cores, stagger):
        layout = self.schedule.options.layout
        if layout == 'spread':
            self.spread_process_threads(cores, stagger)
        elif layout == 'clump':
            self.clump_process_threads(cores, stagger)
        elif layout == 'free':
            self.free_process_threads(cores, stagger)
        else:
            raise NotImplementedError("Bad layout name: " + layout)
    
    distribute = distribute_process_threads
    
    def undistribute(self):
        #DEBUG("CPU job " + str(self.pid) + " finished")
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
                #DEBUG("NEW %s PROC: %d" % (kind.__name__, pid))
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

class NoHardware(Exception):
    pass

class IntelCoreTemp:
    def __init__(self):
        self.files = "/sys/devices/platform/coretemp.*/hwmon/hwmon*/temp*_input"
        self.get_max_temp()
    
    def get_max_temp(self):
        file_list = glob.glob(self.files)
        temps = []
        for f in file_list:
            t = float(slurp(f)) / 1000.0
            temps.append(t)
        if len(temps) < 1:
            raise NoHardware()
        return max(temps)

class AcpiTz:
    def __init__(self):
        self.files = "/sys/devices/virtual/thermal/thermal_zone*/temp"
        self.get_max_temp()
    
    def get_max_temp(self):
        file_list = glob.glob(self.files)
        temps = []
        for f in file_list:
            t = float(slurp(f)) / 1000.0
            temps.append(t)
        if len(temps) < 1:
            raise NoHardware()
        return max(temps)

class Sensors:
    def __init__(self):
        self.exe = "/usr/bin/sensors"
    
    def get_max_temp(self):
        temps = []
        if not os.path.isfile(self.exe):
            raise NoHardware()
        read = json.loads(slurp_output(self.exe, '-j'))
        for device_name, device_data in read.items():
            for sensor_name, sensor_data in device_data.items():
                if not isinstance(sensor_data, dict):
                    continue
                for k, v in sensor_data.items():
                    if k.startswith("temp") and k.endswith("input"):
                        temps.append(v)
        if len(temps) < 1:
            raise NoHardware()
        return max(temps)

class PID:
    def __init__(self, core, options):
        self.core = core
        self.options = options
        if self.options.target_temp == 'max':
            self.auto_target = True
            self.target = 100.0
        else:
            self.auto_target = False
            self.target = float(self.options.target_temp)
        self.interval = self.options.interval
        self.i = 0.0
        self.prev_undershoot = 0.0
        gain = options.gain * 1000.0 # C / Khz
        self.p_scale = gain # C / Khz
        self.i_scale = gain / options.i_time
        self.d_scale = gain * options.d_time
        self.cur_freq = self.core.get_cur_freq()
        
    def detect_throttle(self):
        return self.core.detect_throttle()
    
    def read(self):
        self.cur_freq = self.core.get_cur_freq()
    
    def get_undershoot(self, cur_temp):
        return self.target - cur_temp
    
    def update(self, u):
        cur_freq = self.cur_freq
        max_freq = self.core.get_max_freq()
        min_freq = self.core.get_min_freq()
        new_freq = max(min(cur_freq + u, max_freq), min_freq)
        self.core.set_lim_freq(new_freq)
        if self.core.n == 0:
            INFO("Core {} Frequency {:.2f}->{:.2f}Ghz"
                .format(self.core.n,
                    cur_freq/1000000, new_freq/1000000))
        else:
            DEBUG("Core {} Frequency {:.2f}->{:.2f}Ghz"
                .format(self.core.n,
                    cur_freq/1000000, new_freq/1000000))
    
    def regulate(self, cur_temp):
        undershoot = self.get_undershoot(cur_temp)
        if self.detect_throttle():
            self.i = -100.0
            if self.auto_target:
                self.target -= 1 # reduce target by 1C
                INFO("Temp target reduced to {}C".format(self.target))
        p = undershoot
        diff = undershoot - self.prev_undershoot
        d = diff / self.interval # in degrees/second
        self.i += undershoot * self.interval # in degree.seconds
        u = (p * self.p_scale
             + self.i * self.i_scale
             + d * self.i_scale
             )
        if True or self.core.n == 0:
            DEBUG("T={}/{} P={} I={} D={} U={:.0f}Mhz".format(cur_temp, self.target, p, self.i, d, u/1000))
        self.update(u)
        self.prev_undershoot = undershoot

class SimpleRegulator:
    def __init__(self, core, options):
        self.core = core
        self.options = options
        if self.options.target_temp == 'max':
            self.auto_target = True
            self.target = 100.0
        else:
            self.auto_target = False
            self.target = float(self.options.target_temp)
        self.gain = options.gain * 1000.0 # C / Khz
        self.max_freq = self.core.get_max_freq()
        self.min_freq = self.core.get_min_freq()
        self.lim = self.core.get_cur_freq()
        
    def detect_throttle(self):
        return self.core.detect_throttle()
    
    def read(self):
        pass
    
    def get_undershoot(self, cur_temp):
        return self.target - cur_temp
    
    def update(self, u):
        cur_freq = self.lim
        new_freq = max(min(cur_freq + u, self.max_freq), self.min_freq)
        self.core.set_lim_freq(new_freq)
        self.lim = new_freq
        if self.core.n == 0:
            INFO("Core {} Frequency {:.2f}->{:.2f}Ghz"
                .format(self.core.n,
                    cur_freq/1000000, new_freq/1000000))
        else:
            DEBUG("Core {} Frequency {:.2f}->{:.2f}Ghz"
                .format(self.core.n,
                    cur_freq/1000000, new_freq/1000000))
    
    def regulate(self, cur_temp):
        undershoot = self.get_undershoot(cur_temp)
        if self.detect_throttle():
            if self.auto_target:
                self.target -= 1 # reduce target by 1C
                INFO("Temp target reduced to {}C".format(self.target))
            self.update(-self.gain)
            return
        if undershoot < 0:
            self.update(-self.gain)
        elif undershoot > 0:
            self.update(self.gain)

class Thermo:
    def pick_sensor(self):
        try_sensors = [
            IntelCoreTemp,
            AcpiTz,
            Sensors
            ]
        self.sensor = None
        for try_sensor in try_sensors:
            try:
                self.sensor = try_sensor()
            except NoHardware:
                continue
            break
        if self.sensor is None:
            ERROR("Couldn't find a CPU temperature sensor on your platform. "
                "Try loading the coretemp kernel module for Intel Core processors. "
                "You can also try setting up lm-sensors. "
                )
            if self.options.target_temp:
                exit(1)
    
    def __init__(self, options, topology):
        self.options = options
        self.topology = topology
        self.pick_sensor()
        if options.pid:
            regulator = PID
        else:
            regulator = SimpleRegulator
        if self.options.target_temp is not None:
            for core in self.topology.cores:
                core.pid = regulator(core, self.options)
        else:
            self.target = None
        
    def get_max_temp(self):
        return self.sensor.get_max_temp()
    
    def tick(self):
        cur_temp = self.get_max_temp()
        #INFO("Current temperature: {:.1f}".format(cur_temp))
        for core in self.topology.cores:
            core.pid.read()
        for core in self.topology.cores:
            core.pid.regulate(cur_temp)
        
        
class Gridder:
    def __init__(self, options):
        self.options = options
        self.topology = Topology()
        self.schedule = Schedule(options)
        self.thermo = Thermo(options, self.topology)
        self.always_big = self.options.interval >= 10.0
        self.c = 0
    
    def watch(self):
        while True:
            self.small_tick()
            if self.always_big or self.c % 10 == 0:
                self.big_tick()
            time.sleep(self.options.interval)
            self.c += 1
            DEBUG("-------")
    
    def big_tick(self):
        self.schedule.get_new_jobs()
        self.schedule.distribute(self.topology.cores)
    
    def small_tick(self):
        self.thermo.tick()
        
    def tick(self):
        self.small_tick()
        self.big_tick()
        
    def run(self):
        if self.options.zero_latency:
            latency = open('/dev/cpu_dma_latency', 'wb', buffering=0)
            latency.write(b'\0\0\0\0')
            latency.flush()
        self.watch()

BIT = re.compile(r'bit: (\d+) / (\d+).+?Time per bit: ([\d.]+) ms')
THUSFAR = re.compile(r'bit: (\d+) / (\d+).+?thusfar: ([\d.]+) sec')

class Llr:
    def __init__(self, threads, candidate, exe):
        self.threads = threads
        self.candidate = candidate
        self.exe = exe
        self.proc = None
        self.bufout = ""
        self.total = None
        self.secs = None
    
    async def start(self):
        assert self.proc is None
        self.temp = tempfile.mkdtemp(prefix="benchmark-llr-")
        assert os.path.isdir(self.temp)
        self.proc = await acse(
            self.exe,
            '-w' + self.temp,
            '-d',
            '-q' + self.candidate,
            '-oThreadsPerTest=' + str(self.threads),
            '-oCumulativeTiming=1',
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            )
        self.started = time.time()
    
    def eta(self):
        self.total = self.tpb * self.bits
        #DEBUG("Estimated total time: {:.1f}h ({:.0f}s)".format(self.total/3600, self.total))
    
    def parse(self, line):
        m = BIT.search(line)
        if m:
            self.bit = int(m.group(1))
            self.bits = int(m.group(2))
            self.tpb = float(m.group(3))/1000
            self.secs = time.time() - self.started
            self.eta()
            #DEBUG(str(self.bit/self.bits) + " " + str(self.tpb))
            return True
        m = THUSFAR.search(line)
        if m:
            self.bit = int(m.group(1))
            self.bits = int(m.group(2))
            #self.secs = float(m.group(3))
            self.secs = time.time() - self.started
            self.tpb = self.secs / self.bit
            self.eta()
            #DEBUG(str(self.bit/self.bits) + " " + str(self.tpb))
            #DEBUG(str(time.time()-self.started) + " " + str(self.secs))
            return True
        if line.startswith("Starting"):
            return True
        if line.startswith("Using"):
            return True
        if line.startswith("Caught"):
            return True
        return False
    
    async def watch_stdout(self):
        while self.proc != "over" and not self.proc.stdout.at_eof():
            data = await self.proc.stdout.read(100)
            data = data.decode()
            lines = LLR_RE.split(data)
            lines[0] = self.bufout + lines[0]
            for i in range(0, len(lines)-1):
                line = lines[i].strip()
                if len(line):
                    if self.parse(line):
                        DEBUG("LLR stdout: " + line)
                    else:
                        WARNING("LLR stdout: " + line)
            self.bufout = lines[len(lines)-1]
    
    async def watch_stderr(self):
        while not self.proc.stderr.at_eof():
            data = await self.proc.stderr.readline()
            line = data.decode('ascii').rstrip()
            DEBUG("LLR stderr: " + line)
    
    async def wait(self):
        await self.proc.wait()
        shutil.rmtree(self.temp)
        self.proc = "over"
    
    def coros(self):
        return [
            self.watch_stdout(),
            #self.watch_stderr(),
            self.wait()
            ]
    
    def stop(self):
        self.proc.terminate()

class LlrMark:
    def __init__(self, threads, processes, marker, layout):
        self.threads = threads
        self.processes = processes
        self.marker = marker
        self.layout = layout
        self.gridder = self.marker.gridder
        self.schedule = self.gridder.schedule
        self.processors = len(self.gridder.topology.processors)
        self.tasks = []
        self.aborting = False
        for p in range(0, processes):
            self.tasks.append(Llr(
                    self.threads,
                    self.marker.options.benchmark_llr,
                    self.marker.exe,
                ))
        
    def check_existing(self):
        self.gridder.tick()
        if len(self.schedule.new) + len(self.schedule.alive) > 0:
            ERROR("LLR is still running :(")
            ERROR("Did you stop BOINC?")
            sys.exit(1)
    
    def catch_sigint(self):
        def handler(signum, frame):
            signal.signal(signal.SIGINT, original_sigint_handler)
            traceback.print_stack(f=frame)
            ERROR("Caught SIGINT, exiting!")
            self.stop()
            self.aborting = True
        signal.signal(signal.SIGINT, handler)
    
    def uncatch_sigint(self):
        signal.signal(signal.SIGINT, original_sigint_handler)
        
    def maybe_done(self):
        for task in self.tasks:
            if task.secs is None:
                return
            elif task.secs < 60:
                return
        self.stop()
    
    def desc(self):
        return "Processors: {:.0f}% Threads: {} Layout: {}".format(
            self.processes*self.threads*100/self.processors, 
            self.threads,
            self.layout
            )
    
    def tick(self):
        #DEBUG("tick")
        self.gridder.tick()
        acc = 0
        for task in self.tasks:
            if task.total is None:
                return
            acc += task.total
        avg = acc/len(self.tasks)
        #DEBUG("Average time per task: {:.0f}".format(avg))
        rate = len(self.tasks)/avg # tasks per second
        tpd = rate * 60 * 60 * 24
        #DEBUG("Tasks/day: {:.2f}".format(tpd))
        self.tpd = tpd
        self.maybe_done()

    def run(self):
        self.schedule.options.layout = self.layout
        self.check_existing()
        self.catch_sigint()
        async def r():
            things = []
            for task in self.tasks:
                await task.start()
                things.extend(task.coros())
            time.sleep(1)
            self.gridder.tick()
            gathered = asyncio.gather(*things)
            while True:
                shielded = asyncio.shield(gathered)
                try:
                    return await asyncio.wait_for(shielded, timeout=1.0)
                except asyncio.TimeoutError:
                    self.tick()
        asyncio.run(r())
        self.uncatch_sigint()
        if self.aborting:
            ERROR("Aborted.")
            for task in self.tasks:
                assert task.proc == "over"
            exit(1)
            raise Exception("Unreachable")
    
    def stop(self):
        for task in self.tasks:
            task.stop()

class LlrSampler:
    def __init__(self, mark, ci, *argv):
        self.mark = mark
        self.argv = argv
        self.instance = None
        self.ci = ci
        self.samples = []
    
    def run(self):
        self.instance = self.mark(*self.argv)
        self.desc = self.instance.desc()
        self.instance.run()
        self.samples.append(self.instance.tpd)
        a = np.array(self.samples)
        n = len(a)
        self.mean = np.mean(a)
        if n > 1:
            sem = scipy.stats.sem(a)
            #DEBUG("sem: " + str(sem))
            self.error = sem * scipy.stats.t.ppf(
                (1 + self.ci) / 2,
                n - 1
                )
        else:
            self.error = float('inf')
        
    
    def summary(self):
        return (
            self.desc + " Tasks/day {:.2f}±{:.2f}".format(
                self.mean,
                self.error
                )
            )

class LlrMarker:
    def __init__(self, gridder):
        self.gridder = gridder
        self.options = gridder.options
        self.topology = gridder.topology
        self.ci = self.options.ci
        self.tests = []
        self.remaining = set()
        self.exe = None
        self.get_exe()
        self.plan()
    
    def add_layouts(self, threads, processes, no_aff=False):
        total = threads * processes
        cores = len(self.topology.cores)
        layouts = []
        #INFO("total: " + str(total) + " cores: " + str(cores))
        if len(self.options.layout) > 0:
            layouts = self.options.layout
        elif threads > cores or no_aff:
            layouts = ['free']
        elif total == cores:
            layouts = ['spread', 'free']
        elif processes > cores:
            layouts = ['spread', 'free']
        elif threads == 1:
            layouts = ['spread', 'free']
        else:
            layouts = ['spread', 'clump', 'free']
        for layout in layouts:
            INFO("Will test " + str(processes) + " tasks with "
                + str(threads) + " threads each in layout " + layout)
            self.tests.append(LlrSampler(
                LlrMark,
                self.ci,
                threads,
                processes,
                self,
                layout
                ))
    
    def add_threads(self, processors, force_threads=None, no_aff=False):
        if force_threads is not None:
            threads = [1]
        elif len(self.options.threads) > 0:
            threads = self.options.threads
        else:
            threads = list(range(1, processors+1))
        for t in threads:
            if processors % t == 0:
                processes = processors // t
                self.add_layouts(t, processes, no_aff)
    
    def add_processors(self):
        lps = len(self.topology.processors)
        cores = len(self.topology.cores)
        processors = self.options.processors
        hyper = lps//cores
        if len(processors) == 0:
            for hyperthreading in range(1, hyper+1):
                self.add_threads(cores * hyperthreading)
            self.add_threads(lps - 1, force_threads=lps-1, no_aff=True)
        else:
            for p in processors:
                self.add_threads(p)
    
    def plan(self):
        self.add_processors()
    
    def scan_files(self, directory):
        files = glob.glob(directory + "/*llr*")
        files = [i for i in files if not "wrapper" in i]
        files = [i for i in files if not ".ini" in i]
        files = [i for i in files if os.access(i, os.X_OK)]
        return files
    
    def find_exe(self):
        exes = self.scan_files(".")
        if len(exes) == 0:
            exes = self.scan_files(
                "/var/lib/boinc-client/projects/www.primegrid.com"
                )
        if len(exes) > 1:
            WARNING("Multiple candidate llr binaries: ")
            for i in exes:
                WARNING("    " + i)
            WARNING("Choose one with --llr-executable")
            exes = sorted(exes, reverse=True)
        elif len(exes) < 1:
            ERROR("Can't find llr binary. Choose one with --llr-executable")
            exit(1)
        INFO("Using " + exes[0])
        return exes[0]
    
    def get_exe(self):
        exe = None
        if self.options.llr_executable:
            exe = self.options.llr_executable
        else:
            exe = self.find_exe()
        assert os.access(exe, os.X_OK)
        cputasks.append(re.compile(re.escape(exe)))
        self.exe = exe
    
    def run1(self):
        INFO("Come back in " + str(len(self.remaining)) + " minutes")
        for test in self.remaining:
            test.run()
            INFO(test.summary())
    
    def results_ok(self):
        asc = sorted(self.tests, key=lambda test: test.mean)
        ok = True
        self.remaining = set()
        INFO("---------- Current results:")
        for test in asc:
            INFO(test.summary())
        INFO("----------")
        if self.options.benchmark_losers:
            for i in range(0, len(asc)):
                if i > 0:
                    diff = asc[i].mean - asc[i-1].mean
                    err = asc[i].error + asc[i-1].error
                    #DEBUG(str(diff) + " " + str(err))
                    if err > diff:
                        ok = False
                        self.remaining.update({asc[i], asc[i-1]})
                if i < len(asc)-1:
                    diff = asc[i+1].mean - asc[i].mean
                    err = asc[i+1].error + asc[i].error
                    #DEBUG(str(diff) + " " + str(err))
                    if err > diff:
                        ok = False
                        self.remaining.update({asc[i+1], asc[i]})
        else:
            for t in asc:
                if t is asc[-1]:
                    continue
                diff = asc[-1].mean - t.mean
                err = t.error + asc[-1].error
                if err > diff:
                    #INFO("diff=" + str(diff) +  " err=" + str(err))
                    ok = False
                    self.remaining.update({t, asc[-1]})
        return ok
    
    def run(self):
        done = False
        self.remaining = set(self.tests)
        while not done:
            self.run1()
            done = self.results_ok()
        for test in self.tests:
            print(test.summary())

def main():
    description = "Manage and benchmark CPU affinities for primegrid."
    epilog = ("")
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("--layout",
                        help="how to distribute process threads among cores. "
                        "Can be one of spread, clump, or free. "
                        "When benchmarking, specifies the layouts to benchmark.",
                        nargs='*',
                        )
    parser.add_argument("--interval", 
                        type=float,
                        default=10.0,
                        help="set polling interval."                        )
    parser.add_argument("--zero-latency",
                        help="ask the kernel to provide zero latency. "
                        "Not recommended: this keeps the processors from idling which interferes with hyperthreading",
                        action='store_true',
                        )
    parser.add_argument("--realtime-gpu",
                        help="give GPU processes realtime priority on the CPU. Not recommended: can interfere with system"
                        "function",
                        action='store_true',
                        )
    parser.add_argument("--disallow-hyperthread-swapping",
                        help="don't allow threads to jump between processors on the same core",
                        action='store_true',
                        )
    parser.add_argument("--benchmark-llr",
                        type=str,
                        metavar="CANDIDATE",
                        help="Benchmark LLR layouts")
    parser.add_argument("--llr-executable",
                        type=str,
                        metavar="PATH",
                        help="Path to llr executable")
    parser.add_argument("--ci",
                        type=float,
                        metavar="CONFIDENCE",
                        default=0.95,
                        help="Confidence target")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--benchmark-losers",
                        help="Keep benchmarking losing strategies.",
                        action='store_true',
                        )
    parser.add_argument("--processors",
                        help="Limit benchmark to a certain number of processors. "
                        "You can specify multiple values.",
                        metavar="N_CPUS",
                        type=int,
                        nargs='*')
    parser.add_argument("--threads",
                        help="Limit benchmark to a certain number of threads. "
                        "You can specify multiple values.",
                        metavar="N_THEADS",
                        type=int,
                        nargs='*')
    parser.add_argument("--target-temp",
                        help="Manage CPU temperature by restricting CPU Mhz. "
                        "Specify a temperature in °C or 'max'",
                        metavar="DEGREES_C",
                        default=None)
    parser.add_argument("--pid",
                        action='store_true',
                        help="Use PID control loop for temperature regulation. Not recommended.")
    parser.add_argument("--gain",
                        type=float,
                        metavar="MHZ",
                        default=50.0,
                        help="Proccessor speed step amount in MHz or Gain for PID loop in MHz / °C")
    parser.add_argument("--i-time",
                        type=float,
                        metavar="S",
                        default=10.0,
                        help="Integral time for PID loop in seconds")
    parser.add_argument("--d-time",
                        type=float,
                        metavar="S",
                        default=2.0,
                        help="Derivative time for PID loop in seconds")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(stream=sys.stderr,level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr,level=logging.INFO)
    args.allow_hyperthread_swapping = not args.disallow_hyperthread_swapping
    gridder = Gridder(args)
    if args.benchmark_llr:
        if args.processors == None:
            args.processors = []
        if args.threads == None:
            args.threads = []
        if args.layout == None:
            args.layout = []
        LlrMarker(gridder).run()
    else:
        if args.layout == []:
            args.layout = 'spread'
        else: 
            args.layout = args.layout[-1]
        gridder.run()

main()

"""
[size=18]Download[/size]

The latest version is here: [url]https://github.com/hazelybell/scripts/blob/master/primegrid.py[/url]

[size=18]Requirments[/size]

[list]
* Python 3.7+
* numpy
* scipy
* util-linux (with the taskset command)
[/list]

[size=18]Benchmarking[/size]

To run the benchmark, stop primegrid and call:
[code]python3.7 primegrid.py --benchmark-llr 'k*2^n+1'[/code]
For example:
[code]python3.7 primegrid.py --benchmark-llr '25*2^3962242+1'[/code]

You can get a relevant prime for the project you're interested in from the subproject status page.

This will then run the benchmark with various thread counts, using HT or not, etc. The benchmark will run until it's relatively confident that the best strategy (thread count, HT, affinity) is the best.

Example output:
[code]INFO:__main__:---------- Current results:
INFO:__main__:Processors: 88% Threads: 7 Layout: free Tasks/day 48.68±0.60
INFO:__main__:Processors: 50% Threads: 4 Layout: free Tasks/day 50.39±0.48
INFO:__main__:Processors: 100% Threads: 1 Layout: free Tasks/day 51.79±0.48
INFO:__main__:Processors: 100% Threads: 1 Layout: spread Tasks/day 51.98±0.20
INFO:__main__:Processors: 100% Threads: 8 Layout: free Tasks/day 52.24±0.28
INFO:__main__:Processors: 50% Threads: 4 Layout: spread Tasks/day 52.90±0.30
INFO:__main__:Processors: 50% Threads: 2 Layout: free Tasks/day 57.94±0.69
INFO:__main__:Processors: 50% Threads: 2 Layout: spread Tasks/day 58.86±0.16
INFO:__main__:Processors: 100% Threads: 4 Layout: spread Tasks/day 61.50±0.59
INFO:__main__:Processors: 100% Threads: 4 Layout: clump Tasks/day 61.80±0.16
INFO:__main__:Processors: 100% Threads: 4 Layout: free Tasks/day 61.83±0.21
INFO:__main__:Processors: 100% Threads: 2 Layout: spread Tasks/day 62.34±1.40
INFO:__main__:Processors: 100% Threads: 2 Layout: free Tasks/day 64.24±1.65
INFO:__main__:Processors: 100% Threads: 2 Layout: clump Tasks/day 65.65±1.27
INFO:__main__:Processors: 50% Threads: 1 Layout: spread Tasks/day 72.00±3.00
INFO:__main__:Processors: 50% Threads: 1 Layout: free Tasks/day 72.51±2.46
INFO:__main__:----------[/code]

This indicates that the benchmark has completed a round, and that using 50% of the processors with tasks that have one thread each and letting Linux manage thread affinity was the fastest. It will continue running until it can be confident in it's choice, but you can stop whenever the error bars (the number after the ±) gets small enough for you, or if you're just sick of waiting.

[size=18]Affinity[/size]

The tool currently supports 3 different affinity types:

'free': Let the OS manage affinity itself
'spread': Spread out the threads of a single task across different cores
'clump': Put the threads of a single task on the same cores

Example: Consider a 4-core CPU with hyperthreading. For a task with 2 threads, 'spread' will put the two threads on two different cores, and 'free' will put the two threads on the same core. For a task with 4 threads, 'spread' will put one thread on each core, while 'clump' will put all the threads on 2 cores.

[size=18]Managing Affinity[/size]

If you decide that you want to run primegrid with an affinity layout other than 'free' you can run the script with '--layout clump' or '--layout spread' as root. The script will watch for BOINC to start primegrid LLR tasks and manage their affinity using 'taskset'.

[size=18]Managing CPU Temperature[/size]

For some systems which thermal throttle, or are simply too loud, the script can manage the CPU temperature by specifying '--target-temp 95' or some other temperature in °C. The script will then change the maximum allowed CPU frequency until that temperature is met but not exceeded. Do not use this feature on overclocked systems.

Thermal throttling can degrade performance when the CPU runs too fast, overheats, and then runs very slowly until it cools down, then repeats this process over and over again. It is more efficient to run the CPU at a more consistent, intermediate speed.

[size=18]Advanced Options[/size]

'--processors N' Benchmark only using N processors. This includes logical processors. For example on a 4-core processor with hyperthreading, setting '--processors 4' will only benchmark equivalent to setting 50% of CPUs in BOINC. This will reduce the number of different benchmarks run.

'--threads N' Benchmark only using tasks with N threads. This will reduce the number of different benchmarks to run.

'--layout free|spread|clump' Benchmark only tasks using the specified affinity layout. This will reduce the number of benchmarks to run.

Example for a 6-core hyperthreading processor: '--processors 6 --layout free' will only benchmark 1x6, 2x3, 3x2 and 6x1 tasks x threads without worrying about CPU affinity.  

'--ci .90' Change the confidence interval used to compute the error bars. This will change the number of times the script re-runs benchmarks before it's "confident" in the results.

'--llr-executable path/to/llr' Specify a specific LLR executable.

Even more options: See '--help' output, but be wary, these can have unfortunate effects on your system.

[size=18]Known Bugs[/size]

Systems with complicated topologies are not modelled by the script. That includes systems with multiple CPU sockets, NUMA, and Ryzen 3000-series CPUs. For thread counts strictly more than 2 with hyperthreading or more than 1 without hyperthreading the CPU affinity may be handled poorly. It is better to let the OS manage CPU affinity in these situations. For example, consider the Ryzen 3600X, a 6-core processor with hyperthreading. On this CPU, cores are organized into groups called CCXs, for which communication inside a single CCX is much faster than communication between cores in different CCXs. Thus plans like 4 threads x 3 tasks (100%) may have very poor performance if 'clump' or 'thread' is chosen. I plan supporting these systems better, eventually, if I can get my hands on one.

The script may not have a good idea of the CPU temperature on some AMD systems.

[size=18]Future features[/size]

[list]
* Model NUMA/multisocket/CCX CPU topologies
* Manage C-states
* Manage GPU driver affinity
* Collect power usage and temperature stats
[/list]

[size=18]Example Results[/size]

PPS-DIV on a i7-7700K @ 4.7Ghz: use 1 thread, 50% CPUs, 'free' affinity
PPS-DIV on a i7-3770K @ 4.3Ghz: use 1 thread, 50% CPUs, 'free' affinity
PPS-DIV on a i7-8750H @ 45W: use 1 thread, 50% CPUs, 'spread' affinity
PPS-DIV on a i7-4700MQ @ 27W: use 1 thread, 50% CPUs, 'spread' affinity
PPS-DIV on a Xeon E3-1225 V2 @ stock: use 1 thread, 100% CPUs, 'spread' affinity
PPS-DIV on a i7-9700K @ 4.8Ghz: use 1 thread, 100% CPUs, 'free' affinity
SoB on a i7-7700K @ 4.7Ghz: use 4 threads, 50% CPUs, 'free' affinity

[size=18]Conclusions[/size]

From experimenting with this script I've come to the following conclusions. Note that these are only for linux. Other operating systems handle threads very differently.
[list]
* Using 1 thread on half the CPUs if hyperthreading or on all the CPUs if no hyperthreading is generally an okay choice. Even for SoB, using 1 thread isn't very much slower than 4 threads.
* Choosing whether or not to use all logical processors (threads) on a hyperthreading CPU matters.
* Setting thread affinity can improve performance on some systems.
* Best thread count changes depending on project. Both [i]k[/i] and [i]n[/i] can have an effect here and they can have different effects. 
[/list]
"""

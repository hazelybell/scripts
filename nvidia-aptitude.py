#!/usr/bin/env python3

import argparse
import subprocess
import re
import sys

import logging
logger = logging.getLogger(__name__)
DEBUG = logger.debug
INFO = logger.info
WARNING = logger.warning
ERROR = logger.error
CRITICAL = logger.critical

PSTART = re.compile(r"Package (\S+):")
VERS = re.compile(r"(\S+)\s+(\S+)")
PACKAGE = re.compile(r"(\S+)")

MAIN_PACKAGE = 'nvidia-driver'

SOURCE_PACKAGES = [
    'nvidia-graphics-drivers',
    'nvidia-settings',
    'nvidia-modprobe',
    'nvidia-persistenced'
    ]

class Version:
    def __init__(self, v, state):
        DEBUG("Version: " + v + " " + state)
        self.v = v
        self.state = state

    def __hash__(self):
        return hash(self.v)

class Package:
    def __init__(self, desc):
        if ':' in desc:
            p, a = desc.split(":")
            self.name = p
            self.arch = a
        else:
            self.name = desc
            self.arch = None
        self.versions = dict()
        self.installed_version = None
        self.wanted_version = None
    
    def __str__(self):
        if self.arch is None:
            return self.name
        else:
            return self.name + ":" + self.arch

    def __hash__(self):
        return hash(self.__str__())
    
    def add_version(self, v):
        self.versions[v.v] = v
        if v.state == 'installed':
            self.installed_version = v

class Packages:
    def __init__(self):
        self.s = set()
        self.by_name = dict()
        self.by_desc = dict()
    
    def add(self, pkg):
        self.s.add(pkg)
        self.by_name[pkg.name] = pkg
        self.by_desc[str(pkg)] = pkg
        assert(len(self.s) == len(self.by_desc))
    
    def remove(self, pkg):
        self.s.remove(pkg)
        if pkg.name in self.by_name and self.by_name[pkg.name] is pkg:
            del self.by_name[pkg.name]
        del self.by_desc[str(pkg)]
        assert(len(self.s) == len(self.by_desc))

class Aptitude:
    def __init__(self, options):
        self.path = "/usr/bin/aptitude"
    
    def arch_search(self, pkgs):
        filtered = list()
        for pkg in pkgs.s:
            if len(pkg.versions) != 0:
                continue # skip this one, we've already found versions for it
            if pkg.arch is None:
                filtered.append(pkg.name)
            else:
                filtered.append(pkg.name + " ?architecture(" + pkg.arch + ")")
        return filtered
    
    def get_versions(self, packages):
        results = dict()
        args = [self.path, "-F", "%p %C", "versions"]
        args.extend(self.arch_search(packages))
        output = subprocess.check_output(args, text=True)
        current_package = None
        for line in output.splitlines():
            #DEBUG("line: " +  line)
            match = PSTART.match(line)
            if match is not None:
                desc = match.group(1)
                if desc not in packages.by_desc:
                    packages.add(Package(desc))
                current_package = packages.by_desc[desc]
                continue
            match = VERS.match(line)
            if match is not None:
                version = Version(match.group(1), match.group(2))
                current_package.add_version(version)
    
    def get_packages(self, search, packages):
        args = [self.path, "-F", "%p", "search"]
        args.extend(search)
        output = subprocess.check_output(args, text=True)
        for line in output.splitlines():
            match = PACKAGE.match(line)
            if match is not None:
                desc = match.group(1)
                if desc not in packages.by_desc:
                    packages.add(Package(desc))
    
    def get_packages_by_installed_version(self, version, packages):
        pattern = "?installed ?version(" + version + ") ?not(?virtual)"
        self.get_packages([pattern], packages)
    
    def get_installed_packages_by_source(self, source, packages):
        pattern = "?installed ?source-package(" + source + ") ?not(?virtual)"
        self.get_packages([pattern], packages)
    
    def install(self, packages):
        args = [self.path, "install"]
        for p in packages.s:
            if p.wanted_version is not None:
                arg = str(p) + "=" + p.wanted_version
                args.append(arg)
        subprocess.run(args)

class Upgrader:
    def get_candidates(self):
        by_version = self.aptitude.get_packages_by_installed_version(
            self.current_version, self.packages)
        for p in SOURCE_PACKAGES:
            by_spkg = self.aptitude.get_installed_packages_by_source(p, self.packages)
    
    def available_versions(self):
        for v in self.packages.by_desc[MAIN_PACKAGE].versions:
            INFO("Available version: " + v)
    
    def remove_not_installed(self):
        to_remove = set()
        for p in self.packages.s:
            if p.installed_version is None:
                to_remove.add(p)
        INFO(str(len(to_remove)) + " packages aren't actually installed.")
        for p in to_remove:
            self.packages.remove(p)
    
    def __init__(self, options):
        self.aptitude = Aptitude(options)
        self.target_version = options.version
        self.packages = Packages()
        self.packages.add(Package(MAIN_PACKAGE))
        self.aptitude.get_versions(self.packages)
        self.current_version = self.packages.by_desc[MAIN_PACKAGE].installed_version.v
        INFO("Current version of {}: {}".format(MAIN_PACKAGE, self.current_version))
        self.available_versions()
        self.get_candidates()
        DEBUG("Candidates: " + "\n".join(sorted(self.packages.by_desc.keys())))
        INFO(str(len(self.packages.s)) + " candidates.")
        self.check_versions()
        self.remove_not_installed()
    
    def check_versions(self):
        lite = self.target_version
        if '-' in lite:
            lite, _ = self.target_version.rsplit('-', maxsplit=1)
        self.aptitude.get_versions(self.packages)
        checked = dict()
        for package in self.packages.s:
            for version in package.versions:
                if version == self.target_version:
                    checked[package] = version
                    package.wanted_version = version
        for package in self.packages.s:
            if package not in checked:
                for v in package.versions:
                    if v.startswith(lite):
                        checked[package] = v
                        package.wanted_version = v
        for package in self.packages.s:
            if package not in checked:
                INFO(str(package) + " has no matching version.")
                INFO("Available versions of " + str(package))
                for v in package.versions:
                    INFO("    " + v)
        INFO(str(len(self.packages.s)) + " candidates.")
        INFO(str(len(checked)) + " have matching versions.")

    def install(self):
        self.aptitude.install(self.packages)

def main():
    description = "Try to get aptitude to upgrade nvidia driver packages."
    epilog = ("")
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('version', type=str,
                    help='new nvidia driver version to install')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(stream=sys.stderr,level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr,level=logging.INFO)
    Upgrader(args).install()

if __name__=="__main__":
    main()

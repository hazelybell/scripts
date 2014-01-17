#!/usr/bin/perl

# Use like: bigpie.pl <file containing search string> <file containing replace string> <files to search and replace in>

# based on code by almut on Apr 08, 2010 at 15:32 UTC
# http://www.perlmonks.org/?node_id=833495, Jan 17, 2014

use strict;
use warnings;
use File::Slurp;

$/ = '';

sub inplace_edit {
    my ($callback, @files) = @_;
    return unless ref($callback) eq "CODE";
    local $^I = "";
    while (<>) {
        $callback->();
        print;
    }
}

my $f = read_file(shift @ARGV);
chomp $f;
my $r = read_file(shift @ARGV);
chomp $r;

inplace_edit(
    sub {
      s/\Q$f\E/$r/gs;
    }
);
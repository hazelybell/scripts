#!/usr/bin/perl

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
my $r = read_file(shift @ARGV);

inplace_edit(
    sub {
      s/$f/$r/gs;
    }
);
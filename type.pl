#!/usr/bin/perl

# Script for typing onto an android phone from my terminal easily.
# © Joshua Charles Campbell, 2014

use strict;
use warnings;

use String::ShellQuote;
use Term::UI;

my $term = Term::ReadLine->new('type.pl');

sub type {
  my ($text) = @_;

  my @text = split(/(?<=%)(?=s)/, $text);

  @text = map {shell_quote_best_effort $_;} @text;

  foreach (@text) {
    s/ /%s/g;
    system(qw(adb shell input text), $_);
  }
  
}

if (@ARGV > 0) {
  my $text = join(" ", @ARGV);
  type($text);
} else {
  while (my $reply = $term->get_reply( prompt => 'Keyboard> ')) {
    type($reply);
  }
}


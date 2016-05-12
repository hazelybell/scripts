use strict;
use warnings;

$\ = $/;

sub scanh {
	my ($x) = @_;
	return unpack("C4", pack("H8",$x));
}

sub printh {
	return unpack("H8",pack("C4",@_));
}

my ($a, $b, $steps) = @ARGV;

my @a = scanh($a);
my @b = scanh($b);

for (my $i = 0; $i < $steps; $i++) {
	my @g;
	for (my $j = 0; $j < 4; $j++) {
		$g[$j]=int($a[$j]+($b[$j]-$a[$j])*$i/$steps);
	}
	print printh(@g);
}
print printh(@b);

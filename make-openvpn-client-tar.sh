#!/bin/bash

echo "Build a client config tarball for openvpn."

if [[ -z "$1" ]]
then	echo "USAGE: $0 newopenvpnclientname"
	exit 1
fi

OUTDIR="$(pwd)"
NAME=$1

(
	cd /etc/openvpn/rsa
	source vars
	./pkitool $NAME
)

[[ -f /etc/openvpn/rsa/keys/$NAME.key ]] || exit 3
[[ -f /etc/openvpn/rsa/keys/$NAME.key ]] || exit 4

TEMPDIR=`mktemp -d`

mkdir -p $TEMPDIR/openvpn || exit 2
cd $TEMPDIR/openvpn || exit 5
chmod 0700 .

cp /etc/openvpn/rsa/keys/$NAME.key ./
cp /etc/openvpn/rsa/keys/$NAME.crt ./
cp /etc/openvpn/ca.pizza.crt ./
cp /etc/openvpn/ta.pizza.key ./

cat >pizza.conf <<EOF
client
proto udp
dev tun2
ca ca.pizza.crt
cert $NAME.crt
key $NAME.key
remote pizza.cs.ualberta.ca 1194
nobind
ns-cert-type server
tls-auth ta.pizza.key 1
cipher AES-256-CBC
comp-lzo
user nobody
group nogroup
persist-key
persist-tun
status openvpn-status.log
verb 4
ping 1
ping-restart 20
EOF

cat >instructions.txt <<EOF
Place these files in /etc/openvpn!

Then start openvpn.

systemctl start openvpn@pizza.service
systemctl enable openvpn@pizza.service

Take note of your IP address for tun2 which will be 192.168.2.x

ip addr dev tun2

To see logs:
journalctl -u openvpn@pizza.service

EOF

cd "$OUTDIR"
tar -cvaf $NAME.txz -C $TEMPDIR openvpn || exit 6
echo "Your client config is in $NAME.txz"


#!/bin/bash
firewall_stop() {
        bash -x <<EOF
        iptables -t filter --flush
        
        iptables -t filter --policy INPUT ACCEPT
        iptables -t filter --policy FORWARD ACCEPT
        iptables -t filter --policy OUTPUT ACCEPT
EOF
}

firewall_start() {
        bash -x <<EOF
        iptables -t filter --flush
        iptables -t filter --delete-chain
        
        iptables -t filter --policy INPUT DROP
        iptables -t filter --policy FORWARD ACCEPT
        iptables -t filter --policy OUTPUT ACCEPT
        
        iptables -t filter -A INPUT -i lo -j ACCEPT
        iptables -t filter -A INPUT -i tun0 -j ACCEPT
        iptables -t filter -A INPUT -i tun1 -j ACCEPT
        
        iptables -t filter -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
        
        # ssh
        iptables -t filter -A INPUT -p tcp --dport 22 -m state --state NEW -j ACCEPT
        # cups
        iptables -t filter -A INPUT -p udp --dport 631 -j ACCEPT
        # X11
        iptables -t filter -A INPUT -p tcp --dport 6000 -m state --state NEW -j ACCEPT
        # avahi mdns
        iptables -t filter -A INPUT -p udp --dport 5353 -j ACCEPT
        
        # Kill windows noise
        iptables -t filter -A INPUT -m addrtype --dst-type BROADCAST -j DROP
        iptables -t filter -A INPUT -m addrtype --dst-type MULTICAST -j DROP
#         iptables -t filter -A INPUT -p tcp -m multiport --destination-ports 135:139,1024:1030,1433:1434,2967:2968,5900
        iptables -t filter -A INPUT -p tcp -m multiport --destination-ports 135:139
        iptables -t filter -A INPUT -p udp -m multiport --destination-ports 135:139
        
        iptables -t filter -A INPUT -p icmp -j ACCEPT
        
        iptables -t filter -A INPUT -j LOG --log-prefix "Input packet/ died:"
EOF
}

case "$1" in
    start|restart|force-reload)
        firewall_stop
        firewall_start
    ;;
    stop)
        firewall_stop
    ;;
    *)
        echo "Usage: $0 {start|stop|restart|force-reload}"
        exit 1
    ;;
esac

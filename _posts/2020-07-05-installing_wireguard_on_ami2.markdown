---
layout: post
title:  "Installing WireGuard on an AWS AMI2 EC2 instance"
date:   2020-07-05 11:00:00 +0200
categories: "AWS WireGuard Linux AWS EC2"
---

Make sure you have updated your AMI2 instance with the latest updates and restarted (for the newest kernel to run).
Then type
```bash
sudo curl -Lo /etc/yum.repos.d/wireguard.repo https://copr.fedorainfracloud.org/coprs/jdoss/wireguard/repo/epel-7/jdoss-wireguard-epel-7.repo
sudo yum clean all
sudo yum install wireguard-dkms wireguard-tools iptables-services -y
sudo dkms autoinstall
sudo modprobe wireguard
sudo dmesg | grep wireguard
```
to install and load the WireGuard kernel module.

# TODO: Setup network interface, keys, etc.

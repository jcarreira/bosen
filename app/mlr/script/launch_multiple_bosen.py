#!/usr/bin/env python

from shell_command import shell_call
import subprocess
from time import sleep
import os
import sys 

init_lr = [0.005, 0.01, 0.001]
num_servers = [1, 2, 4]
num_batches_per_epoch = [20000]
num_app_threads = [8]

def update_learning_rate_config(lr):
    # change learning rate
    cmd = "perl -i -pe  's/init_lr\": .*/init_lr\": "\
             "{}/g' launch_criteo.py".format(lr)
    shell_call(cmd)
def update_num_servers(n_servers):
    if n_servers == 1:
        filename = "1server"
    elif n_servers == 2:
        filename = "2servers"
    elif n_servers == 4:
        filename = "4servers"

    # change learning rate
    cmd = "perl -i -pe  's/hostfile_name=.*/hostfile_name="\
             "\"{}\"/g' launch_criteo.py".format(filename)
    shell_call(cmd)
def update_num_batches_per_epoch(nbe):
    cmd = "perl -i -pe  's/num_batches_per_epoch\": .*/num_batches_per_epoch\": "\
             "{}/g' launch_criteo.py".format(nbe)
    shell_call(cmd)
def update_num_threads(n_threads):
    cmd = "perl -i -pe  's/num_app_threads\": .*/num_app_threads\": "\
             "{}/g' launch_criteo.py".format(nthreads)
    shell_call(cmd)

def kill_bosen():
    cmd = "/ebs/joao/petuum_test/bosen/app/mlr/script/kill.py ../../../machinefiles/4servers"
    shell_call(cmd)

def clean_logs():
    shell_call("rm -rf /ebs/joao/*log")
    shell_call("rm -rf ~/bosen_log_loss.txt")

def name_from_params(lr, n_servers, nbe, nthreads):
    return str(lr) + "-" + str(n_servers) + "-" + str(nbe) + "-" + str(nthreads)

def run_bosen(lr, n_servers, nbe, nthreads, sleep_time):
    update_learning_rate_config(lr)
    update_num_servers(n_servers)
    update_num_batches_per_epoch(nbe)
    update_num_threads(nthreads)

    p=subprocess.Popen(
            "/ebs/joao/petuum_test/bosen/app/mlr/script/launch_criteo.py")
    sleep(sleep_time)
    kill_bosen()

    run_name = name_from_params(lr, n_servers, nbe, nthreads)
    shell_call("mkdir {}/".format(run_name))
    # mv logs in /ebs/joao
    shell_call("mv /ebs/joao/*log {}/".format(run_name))
    shell_call("mv ~/bosen_log_loss.txt {}/".format(run_name))

run_time = 20 * 60 #20 * 60

for lr in init_lr:
    for n_servers in num_servers:
        for nbe in num_batches_per_epoch:
            for nthreads in num_app_threads:
                clean_logs()
                print "Running with params. lr: ", lr, "n_servers:",n_servers, \
                        "nthreads:", nthreads
                run_bosen(lr, n_servers, nbe, nthreads, run_time)
                sys.exit(-1)



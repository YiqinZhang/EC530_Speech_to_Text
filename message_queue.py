import multiprocessing
from multiprocessing import Process, Queue
import time
import os


def do_something(q):
    seconds = q.get()
    print(f'Sleeping {seconds} second(s)...')
    time.sleep(seconds)
    print(f'{multiprocessing.current_process().name} Done Sleeping in {seconds} second(s)')
    return seconds


if __name__ == "__main__":
    processes = []
    num_processes = os.cpu_count()
    # number of CPUs on the machine. Usually a good choise for the number of processes
    q = Queue()

    start = time.time()
    # create processes and asign a function for each process
    for i in range(num_processes):
        q.put(i)
        process = Process(target=do_something, args=(q,))
        processes.append(process)

    # start all processes
    for process in processes:
        process.start()
        print(f'{process.name} starts')

    # wait for all processes to finish
    # block the main programm until these processes are finished
    for process in processes:
        process.join()

    end = time.time()

    print(f'Finished in {round(end-start, 2)} second(s)')
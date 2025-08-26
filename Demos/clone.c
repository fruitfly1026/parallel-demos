#define _GNU_SOURCE // required to use clone
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>
#include <stdlib.h>

/* In some cases, if you use -std=c99 or -std=c11, it will disable GNU extensions. 
    Use -std=gnu99 or -std=gnu11 instead.
*/

int thread_func(void *arg) {
    printf("child: getpid()=%d, gettid()=%ld\n", getpid(), syscall(SYS_gettid));
    return 0;
}

int main() {
    const int STACK_SIZE = 1024 * 1024;
    void *stack = malloc(STACK_SIZE);

    int flags = CLONE_VM | CLONE_FS | CLONE_FILES |
                CLONE_SIGHAND | CLONE_THREAD;

    clone(thread_func, stack + STACK_SIZE, flags, NULL);

    printf("parent: getpid()=%d, gettid()=%ld\n", getpid(), syscall(SYS_gettid));
    sleep(1);
    return 0;
}

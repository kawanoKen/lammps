// Bounded MPI traffic generator for an explicitly allocated host set.
// This is a co-runner, not a benchmark of the site's network fabric.

#define _POSIX_C_SOURCE 200809L

#include <mpi.h>

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t stop_requested = 0;

static void stop_handler(int signal_number) {
  (void)signal_number;
  stop_requested = 1;
}

static void usage(const char *program) {
  fprintf(stderr,
          "usage: %s --seconds N --message-mb N --streams N --sleep-us N\n",
          program);
}

int main(int argc, char **argv) {
  double seconds = 30.0;
  long message_mb = 4;
  int streams = 1;
  int sleep_us = 0;
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) {
      usage(argv[0]);
      return 2;
    }
    if (strcmp(argv[index], "--seconds") == 0) {
      seconds = atof(argv[++index]);
    } else if (strcmp(argv[index], "--message-mb") == 0) {
      message_mb = atol(argv[++index]);
    } else if (strcmp(argv[index], "--streams") == 0) {
      streams = atoi(argv[++index]);
    } else if (strcmp(argv[index], "--sleep-us") == 0) {
      sleep_us = atoi(argv[++index]);
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (seconds <= 0.0 || message_mb < 1 || streams < 1 || sleep_us < 0) {
    usage(argv[0]);
    return 2;
  }

  MPI_Init(&argc, &argv);
  signal(SIGTERM, stop_handler);
  signal(SIGINT, stop_handler);
  int rank = 0;
  int world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  if (world < 2) {
    if (rank == 0) fprintf(stderr, "mpi_stream requires at least two MPI ranks\n");
    MPI_Finalize();
    return 3;
  }

  const size_t bytes = (size_t)message_mb * 1024ULL * 1024ULL;
  unsigned char *buffer = (unsigned char *)malloc(bytes);
  if (buffer == NULL) {
    fprintf(stderr, "rank %d: unable to allocate %ld MiB\n", rank, message_mb);
    MPI_Abort(MPI_COMM_WORLD, 4);
  }
  memset(buffer, rank & 0xff, bytes);
  const int previous = (rank + world - 1) % world;
  const int next = (rank + 1) % world;
  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();
  long long iterations = 0;
  while (!stop_requested && MPI_Wtime() - start < seconds) {
    for (int stream = 0; stream < streams; ++stream) {
      MPI_Sendrecv_replace(buffer, (int)bytes, MPI_BYTE, next, 7,
                           previous, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    ++iterations;
    if (sleep_us > 0) {
      struct timespec pause_time = {
          .tv_sec = sleep_us / 1000000,
          .tv_nsec = (long)(sleep_us % 1000000) * 1000L,
      };
      nanosleep(&pause_time, NULL);
    }
  }
  const double elapsed = MPI_Wtime() - start;
  long long total_iterations = 0;
  MPI_Reduce(&iterations, &total_iterations, 1, MPI_LONG_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  double max_elapsed = 0.0;
  MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  if (rank == 0) {
    const double bytes_per_second =
        (double)bytes * (double)streams * (double)total_iterations / max_elapsed;
    printf("mpi_stream ranks=%d seconds=%.3f message_mb=%ld streams=%d "
           "sleep_us=%d aggregate_gib_per_s=%.6f iterations_sum=%lld\n",
           world, max_elapsed, message_mb, streams, sleep_us,
           bytes_per_second / (1024.0 * 1024.0 * 1024.0), total_iterations);
    fflush(stdout);
  }
  free(buffer);
  MPI_Finalize();
  return 0;
}

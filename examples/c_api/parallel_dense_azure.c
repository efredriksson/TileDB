/**
 * @file   parallel_dense_azure.c
 *
 * @section DESCRIPTION
 *
 * Benchmark for parallel array operations on Azure Blob Storage.
 * Creates, writes, and reads NUM_ARRAYS dense 2D arrays concurrently using
 * pthreads. Each phase is run twice (cold then warm) to separate connection
 * setup cost from steady-state throughput.
 *
 * Usage: set URI_PREFIX_BASE and vfs.azure.storage_account_name below,
 * then run the binary. A unique timestamp suffix is appended to the prefix
 * on each run so no manual cleanup between runs is required.
 */

#define _POSIX_C_SOURCE 199309L
#include <tiledb/tiledb.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_ARRAYS     100
#define NUM_ARRAYS_MAX 300
#define NUM_FRAGMENTS  5
#define URI_PREFIX_BASE "azure://experimentation/parallel_array_"

static char g_uri_prefix[128];

typedef struct {
  int index;
  tiledb_ctx_t* ctx;
} thread_arg_t;

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void check_rc(int rc, tiledb_ctx_t* ctx, const char* msg) {
  if (rc != TILEDB_OK) {
    tiledb_error_t* err;
    tiledb_ctx_get_last_error(ctx, &err);
    const char* errmsg = NULL;
    tiledb_error_message(err, &errmsg);
    fprintf(stderr, "Error: %s: %s\n", msg, errmsg ? errmsg : "unknown");
    exit(1);
  }
}

static void* create_array_thread(void* arg) {
  thread_arg_t* a = (thread_arg_t*)arg;

  char uri[256];
  snprintf(uri, sizeof(uri), "%s%d", g_uri_prefix, a->index);

  tiledb_domain_t* domain;
  tiledb_domain_alloc(a->ctx, &domain);

  uint64_t row_dom[2] = {1, 500};
  uint64_t col_dom[2] = {1, 500};
  uint64_t tile_extent = 100;

  tiledb_dimension_t *dim_r, *dim_c;
  tiledb_dimension_alloc(a->ctx, "r", TILEDB_UINT64, row_dom, &tile_extent, &dim_r);
  tiledb_dimension_alloc(a->ctx, "c", TILEDB_UINT64, col_dom, &tile_extent, &dim_c);

  tiledb_domain_add_dimension(a->ctx, domain, dim_r);
  tiledb_domain_add_dimension(a->ctx, domain, dim_c);
  tiledb_dimension_free(&dim_r);
  tiledb_dimension_free(&dim_c);

  tiledb_attribute_t* attr;
  tiledb_attribute_alloc(a->ctx, "a", TILEDB_INT32, &attr);

  tiledb_array_schema_t* schema;
  tiledb_array_schema_alloc(a->ctx, TILEDB_DENSE, &schema);
  tiledb_array_schema_set_domain(a->ctx, schema, domain);
  tiledb_array_schema_add_attribute(a->ctx, schema, attr);
  tiledb_domain_free(&domain);
  tiledb_attribute_free(&attr);

  tiledb_array_schema_check(a->ctx, schema);
  int rc = tiledb_array_create(a->ctx, uri, schema);
  check_rc(rc, a->ctx, "tiledb_array_create");
  tiledb_array_schema_free(&schema);
  return NULL;
}

static void* write_array_thread(void* arg) {
  thread_arg_t* a = (thread_arg_t*)arg;

  char uri[256];
  snprintf(uri, sizeof(uri), "%s%d", g_uri_prefix, a->index);

  for (int f = 0; f < NUM_FRAGMENTS; f++) {
    uint64_t row_start = f * 100 + 1;
    uint64_t row_end = row_start + 100 - 1;

    uint64_t col_start = 1;
    uint64_t col_end = 500;

    uint64_t nrows = 100;
    uint64_t ncols = 500;
    uint64_t nvals = nrows * ncols;
    uint64_t buffer_size = nvals * sizeof(int32_t);

    int32_t* data = malloc(nvals * sizeof(int32_t));
    for (uint64_t i = 0; i < nvals; i++) {
      data[i] = (int32_t)(a->index * 1000 + f);
    }

    tiledb_array_t* array;
    tiledb_array_alloc(a->ctx, uri, &array);
    tiledb_array_open(a->ctx, array, TILEDB_WRITE);

    tiledb_query_t* query;
    tiledb_query_alloc(a->ctx, array, TILEDB_WRITE, &query);

    tiledb_subarray_t* sub;
    tiledb_subarray_alloc(a->ctx, array, &sub);
    tiledb_subarray_add_range(a->ctx, sub, 0, &row_start, &row_end, NULL);
    tiledb_subarray_add_range(a->ctx, sub, 1, &col_start, &col_end, NULL);
    tiledb_query_set_subarray_t(a->ctx, query, sub);
    tiledb_query_set_layout(a->ctx, query, TILEDB_ROW_MAJOR);
    tiledb_query_set_data_buffer(a->ctx, query, "a", data, &buffer_size);

    int rc = tiledb_query_submit(a->ctx, query);
    check_rc(rc, a->ctx, "tiledb_query_submit (write)");

    tiledb_query_free(&query);
    tiledb_subarray_free(&sub);
    tiledb_array_close(a->ctx, array);
    tiledb_array_free(&array);
    free(data);
  }
  return NULL;
}

static void* read_array_thread(void* arg) {
  thread_arg_t* a = (thread_arg_t*)arg;

  char uri[256];
  snprintf(uri, sizeof(uri), "%s%d", g_uri_prefix, a->index);

  uint64_t row_start = 1;
  uint64_t row_end = 500;
  uint64_t col_start = 1;
  uint64_t col_end = 1;

  uint64_t nvals = 500;
  int32_t* buffer = malloc(nvals * sizeof(int32_t));

  tiledb_array_t* array;
  tiledb_array_alloc(a->ctx, uri, &array);
  tiledb_array_open(a->ctx, array, TILEDB_READ);

  tiledb_subarray_t* sub;
  tiledb_subarray_alloc(a->ctx, array, &sub);
  tiledb_subarray_add_range(a->ctx, sub, 0, &row_start, &row_end, NULL);
  tiledb_subarray_add_range(a->ctx, sub, 1, &col_start, &col_end, NULL);

  tiledb_query_t* query;
  tiledb_query_alloc(a->ctx, array, TILEDB_READ, &query);
  tiledb_query_set_subarray_t(a->ctx, query, sub);
  tiledb_query_set_layout(a->ctx, query, TILEDB_ROW_MAJOR);
  tiledb_query_set_data_buffer(a->ctx, query, "a", buffer, &nvals);

  int rc = tiledb_query_submit(a->ctx, query);
  check_rc(rc, a->ctx, "tiledb_query_submit (read)");

  tiledb_query_free(&query);
  tiledb_subarray_free(&sub);
  tiledb_array_close(a->ctx, array);
  tiledb_array_free(&array);
  free(buffer);
  return NULL;
}

/* Run a phase: spawn NUM_ARRAYS threads starting at index offset, join all. */
static double run_phase(
    pthread_t* threads,
    thread_arg_t* args,
    int offset,
    void* (*fn)(void*),
    tiledb_ctx_t* ctx) {
  double t0 = now();
  
  for (int i = 0; i < NUM_ARRAYS; i++) {
    args[offset + i].index = offset + i;
    args[offset + i].ctx = ctx;
    pthread_create(&threads[i], NULL, fn, &args[offset + i]);
  }
  
  for (int i = 0; i < NUM_ARRAYS; i++)
    pthread_join(threads[i], NULL);
  return now() - t0;
}

int main(void) {
  tiledb_config_t* cfg;
  tiledb_error_t* err;
  tiledb_config_alloc(&cfg, &err);

  /* Azure Blob Storage */
  tiledb_config_set(cfg, "vfs.azure.storage_account_name", "blobfortiledb", &err);

  /* Parallelism */
  tiledb_config_set(cfg, "sm.compute_concurrency_level", "16", &err);
  tiledb_config_set(cfg, "sm.io_concurrency_level", "100", &err);

  /* Set false to skip legacy pre-v10 HTTP checks (requires all arrays >= v10) */
  tiledb_config_set(cfg, "sm.legacy_compatibility", "false", &err);

  tiledb_ctx_t* ctx;
  tiledb_ctx_alloc(cfg, &ctx);
  tiledb_config_free(&cfg);

  snprintf(g_uri_prefix, sizeof(g_uri_prefix), "%s%ld_", URI_PREFIX_BASE, (long)time(NULL));
  printf("URI prefix: %s\n", g_uri_prefix);

  pthread_t threads[NUM_ARRAYS_MAX];
  thread_arg_t args[NUM_ARRAYS_MAX];

  /* Warm up the connection pool with a single array before timing. */
  thread_arg_t warmup = {.index = 1000000, .ctx = ctx};
  create_array_thread(&warmup);

  double elapsed;

  elapsed = run_phase(threads, args, 0, create_array_thread, ctx);
  printf("=====> Create arrays cold elapsed: %.6f seconds\n", elapsed);

  elapsed = run_phase(threads, args, NUM_ARRAYS, create_array_thread, ctx);
  printf("=====> Create arrays warm elapsed: %.6f seconds\n", elapsed);

  elapsed = run_phase(threads, args, 0, write_array_thread, ctx);
  printf("=====> Writing arrays cold elapsed: %.6f seconds\n", elapsed);

  elapsed = run_phase(threads, args, NUM_ARRAYS, write_array_thread, ctx);
  printf("=====> Writing arrays warm elapsed: %.6f seconds\n", elapsed);

  elapsed = run_phase(threads, args, 0, read_array_thread, ctx);
  printf("=====> Reading arrays cold elapsed: %.6f seconds\n", elapsed);

  elapsed = run_phase(threads, args, NUM_ARRAYS, read_array_thread, ctx);
  printf("=====> Reading arrays warm elapsed: %.6f seconds\n", elapsed);

  tiledb_ctx_free(&ctx);
  return 0;
}

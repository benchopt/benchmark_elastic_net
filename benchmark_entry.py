# init a benchmark object with root directory
from benchopt.benchmark import Benchmark
from benchopt.runner import run_benchmark


benchmark_obj = Benchmark("./")
df_results = run_benchmark(benchmark_obj, plot_result=True)

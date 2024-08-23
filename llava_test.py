from unibench import Evaluator

ev = Evaluator(
    models=["llava_1_5_7b"],
    benchmarks=['kitti_distance'],
    benchmarks_dir="/fsx-checkpoints/haideraltahan/.cache/unibench/data/",
)

ev.evaluate()
